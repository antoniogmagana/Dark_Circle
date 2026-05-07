"""
Replay a paired (audio, seismic) recording as a live ROS2 sensor source
for end-to-end inference-engine testing.

Publishes one bundled-channel std_msgs/String JSON message per tick on
the configured per-array topic — the same wire format the customer's
real publisher uses. Optionally subscribes back to the configured
inference output topic to score predictions against ground truth and
measure end-to-end latency.

Supported file formats: parquet (.parquet), CSV (.csv), WAV (.wav).
Format is inferred from the file extension.

Recordings are streamed one window at a time. An initial metadata pass
determines total sample count and (when available) sample rate, but the
file body is never fully read into memory.

Required columns (.parquet / .csv; defaults shown — override via
``--column-map``):
    amplitude     raw ADC sample (int or float)
    time_stamp    seconds-since-start (float) — used to infer sample rate
    present       bool (optional; ground truth for presence scoring)

WAV files carry no `present` column, so per-window precision/recall
scoring is skipped automatically. Sample rate comes from the WAV
header. Mono PCM only (8 / 16 / 24 / 32-bit).

Either ``time_stamp`` OR ``--audio-rate`` / ``--seismic-rate`` must
provide a sample rate. If both are present, the explicit rate wins.

Usage examples:

    # Parquet with the default Dark Circle schema
    replay_publisher.py --audio rec_audio.parquet --seismic rec_seismic.parquet

    # CSV with custom column names, explicit rates
    replay_publisher.py --audio rec.csv --seismic rec_geo.csv \\
        --audio-rate 16000 --seismic-rate 100 \\
        --column-map "amplitude=value,time_stamp=t,present=label"

    # No ground truth, just exercise the pipeline
    replay_publisher.py --audio rec.csv --seismic rec_geo.csv \\
        --audio-rate 16000 --seismic-rate 100 --no-subscribe

ROS2 must be sourced before invocation, and the ``ros2_interfaces``
package containing ``InferenceResult`` must be on AMENT_PREFIX_PATH.
See chart/README.md "Testing on pre-recorded data" or run
``scripts/install_replay.sh`` for a guided setup.
"""

import argparse
import csv as _csv
import datetime as _dt
import json
import statistics
import struct
import sys
import threading
import time
import wave
from pathlib import Path

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros2_interfaces.msg import InferenceResult
from std_msgs.msg import String

WINDOW_SEC = 1.0          # ingestor's window size
TICK_HZ = 10              # tick rate; matches fake-publisher to keep the
                          # ingestor's holding-pen logic happy.

DEFAULT_COLUMNS = {
    "amplitude": "amplitude",
    "time_stamp": "time_stamp",
    "present": "present",
}


# -----------------------------------------------------------------------------
# Column-map helpers
# -----------------------------------------------------------------------------


def parse_column_map(raw: str | None) -> dict[str, str]:
    """``"amplitude=value,present=label"`` -> {amplitude: value, present: label}.

    Customer-friendly: only override the columns whose names differ. Keys
    not mentioned fall back to the canonical name in DEFAULT_COLUMNS.
    """
    mapping = dict(DEFAULT_COLUMNS)
    if not raw:
        return mapping
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"--column-map entry '{pair}' is not key=value")
        canonical, actual = (s.strip() for s in pair.split("=", 1))
        if canonical not in DEFAULT_COLUMNS:
            raise ValueError(
                f"--column-map: unknown canonical column '{canonical}'; "
                f"valid: {sorted(DEFAULT_COLUMNS)}"
            )
        mapping[canonical] = actual
    return mapping


def _parse_bool(raw: str) -> bool:
    s = raw.strip().lower()
    if s in ("1", "true", "t", "yes", "y", "present", "p"):
        return True
    if s in ("0", "false", "f", "no", "n", "absent", "a", ""):
        return False
    raise ValueError(f"cannot parse bool from {raw!r}")


# -----------------------------------------------------------------------------
# Streaming readers
#
# Each reader exposes:
#   .total_samples : int         (cheap metadata-only count)
#   .sample_rate   : float|None  (None when the format can't tell us)
#   .has_present   : bool
#   .seek(idx)                   (advance to absolute sample index `idx`)
#   .read(n) -> (amp_list, pres_list_or_None)
#                                (returns up to `n` samples; short read at EOF)
#   .close()
#
# All readers are designed for sequential forward reads after an
# initial seek. There is no random-access requirement once playback
# has started.
# -----------------------------------------------------------------------------


class ParquetReader:
    """Streams a parquet file via `iter_batches`. Pulls one row group at
    a time off disk; converts batches to Python lists lazily."""

    def __init__(self, path: Path, columns: dict[str, str]):
        import pyarrow.parquet as pq

        self._path = path
        self._pq = pq
        self._pf = pq.ParquetFile(path)
        self._amp_col = columns["amplitude"]
        self._ts_col = columns["time_stamp"]
        self._pres_col = columns["present"]

        schema_names = set(self._pf.schema.names)
        if self._amp_col not in schema_names:
            raise ValueError(
                f"{path}: missing required column '{self._amp_col}'; "
                f"schema has {sorted(schema_names)}"
            )
        self._needed = [self._amp_col]
        self._has_ts = self._ts_col in schema_names
        if self._has_ts:
            self._needed.append(self._ts_col)
        self.has_present = self._pres_col in schema_names
        if self.has_present:
            self._needed.append(self._pres_col)

        self.total_samples = self._pf.metadata.num_rows
        self.sample_rate = self._infer_rate_from_head() if self._has_ts else None

        self._batch_iter = None
        self._buf_amp: list = []
        self._buf_pres: list | None = [] if self.has_present else None
        self._cursor_in_buf = 0
        self._abs_pos = 0
        self._open_iter()

    def _infer_rate_from_head(self) -> float | None:
        # Pull just the first batch to estimate sample rate. A few
        # thousand rows is plenty; we don't need the whole file.
        head = next(
            self._pf.iter_batches(batch_size=2048, columns=[self._ts_col]), None
        )
        if head is None:
            return None
        ts = head.column(self._ts_col).to_pylist()
        return _infer_rate(ts)

    def _open_iter(self):
        # batch_size of 64k rows is comfortable for both audio (16kHz =
        # ~4s of data per batch) and seismic (100Hz = ~10min per batch);
        # in either case we never hold more than one batch + the unused
        # tail of the previous batch in memory.
        self._batch_iter = self._pf.iter_batches(
            batch_size=65536, columns=self._needed
        )
        self._buf_amp = []
        self._buf_pres = [] if self.has_present else None
        self._cursor_in_buf = 0

    def _refill(self) -> bool:
        try:
            batch = next(self._batch_iter)
        except StopIteration:
            return False
        self._buf_amp = batch.column(self._amp_col).to_pylist()
        if self.has_present:
            self._buf_pres = batch.column(self._pres_col).to_pylist()
        self._cursor_in_buf = 0
        return True

    def seek(self, idx: int) -> None:
        if idx < 0 or idx > self.total_samples:
            raise ValueError(f"seek out of range: {idx}/{self.total_samples}")
        # Re-open the iterator and skip whole batches until we land in
        # the batch containing `idx`. Skipping a batch only inspects
        # `batch.num_rows` (cheap metadata); we only pay `to_pylist`
        # cost on the partial final batch we're going to start reading
        # from. Without this short-circuit, --start-second 300 at
        # 16kHz would run to_pylist on ~4.8M discarded samples.
        self._open_iter()
        remaining = idx
        while remaining > 0:
            try:
                batch = next(self._batch_iter)
            except StopIteration:
                raise ValueError(
                    f"seek({idx}) past end of file ({self.total_samples} rows)"
                )
            if batch.num_rows <= remaining:
                remaining -= batch.num_rows
                continue
            # Land inside this batch — materialize and position cursor.
            self._buf_amp = batch.column(self._amp_col).to_pylist()
            if self.has_present:
                self._buf_pres = batch.column(self._pres_col).to_pylist()
            self._cursor_in_buf = remaining
            remaining = 0
        self._abs_pos = idx

    def read(self, n: int) -> tuple[list, list | None]:
        amps: list = []
        pres: list | None = [] if self.has_present else None
        while len(amps) < n:
            if self._cursor_in_buf >= len(self._buf_amp):
                if not self._refill():
                    break
            avail = len(self._buf_amp) - self._cursor_in_buf
            take = min(n - len(amps), avail)
            end = self._cursor_in_buf + take
            amps.extend(self._buf_amp[self._cursor_in_buf : end])
            if self.has_present:
                pres.extend(self._buf_pres[self._cursor_in_buf : end])
            self._cursor_in_buf = end
        self._abs_pos += len(amps)
        return amps, pres

    def close(self):
        self._batch_iter = None
        self._pf = None


class CsvReader:
    """Streams a CSV file row-by-row. The metadata pass counts rows
    (cheap line-iteration; no float parsing) and infers sample rate
    from the first ~1000 timestamps."""

    def __init__(self, path: Path, columns: dict[str, str], no_header: bool):
        self._path = path
        self._columns = columns
        self._no_header = no_header

        self._amp_idx, self._ts_idx, self._pres_idx, has_pres_in_header = (
            self._resolve_indices()
        )
        self.has_present = has_pres_in_header
        self.total_samples = self._count_rows()
        self.sample_rate = self._infer_rate_from_head() if self._ts_idx is not None else None

        self._fh = None
        self._reader = None
        self._abs_pos = 0
        self._open_reader()

    def _resolve_indices(self) -> tuple[int, int | None, int | None, bool]:
        if self._no_header:
            # Headerless: amplitude=col0, time_stamp=col1, present=col2.
            return 0, 1, 2, True
        with self._path.open(newline="") as f:
            reader = _csv.reader(f)
            header = next(reader, None)
        if header is None:
            raise ValueError(f"{self._path}: empty CSV")
        try:
            amp_idx = header.index(self._columns["amplitude"])
        except ValueError:
            raise ValueError(
                f"{self._path}: missing required column '{self._columns['amplitude']}'; "
                f"header was {header}"
            ) from None
        ts_idx = (
            header.index(self._columns["time_stamp"])
            if self._columns["time_stamp"] in header
            else None
        )
        pres_idx = (
            header.index(self._columns["present"])
            if self._columns["present"] in header
            else None
        )
        return amp_idx, ts_idx, pres_idx, pres_idx is not None

    def _count_rows(self) -> int:
        # Line-iterate; far cheaper than csv.reader because it avoids
        # field-splitting and quote-handling. Off-by-one for the header
        # is corrected when we subtract. Note: blank lines are counted
        # here but skipped by read()/seek(), so total_samples is an
        # upper bound when the input has empty rows.
        with self._path.open("rb") as f:
            n = sum(1 for _ in f)
        return n - (0 if self._no_header else 1)

    def _infer_rate_from_head(self) -> float | None:
        timestamps: list[float] = []
        with self._path.open(newline="") as f:
            reader = _csv.reader(f)
            if not self._no_header:
                next(reader, None)
            for row in reader:
                if not row or self._ts_idx >= len(row):
                    continue
                try:
                    timestamps.append(float(row[self._ts_idx]))
                except ValueError:
                    continue
                if len(timestamps) >= 1024:
                    break
        return _infer_rate(timestamps) if timestamps else None

    def _open_reader(self):
        if self._fh is not None:
            self._fh.close()
        self._fh = self._path.open(newline="")
        self._reader = _csv.reader(self._fh)
        if not self._no_header:
            next(self._reader, None)

    def seek(self, idx: int) -> None:
        if idx < 0 or idx > self.total_samples:
            raise ValueError(f"seek out of range: {idx}/{self.total_samples}")
        self._open_reader()
        remaining = idx
        for row in self._reader:
            if not row:
                continue
            remaining -= 1
            if remaining <= 0:
                break
        self._abs_pos = idx

    def read(self, n: int) -> tuple[list, list | None]:
        amps: list = []
        pres: list | None = [] if self.has_present else None
        for row in self._reader:
            if not row:
                continue
            amps.append(float(row[self._amp_idx]))
            if (
                self.has_present
                and self._pres_idx is not None
                and self._pres_idx < len(row)
            ):
                pres.append(_parse_bool(row[self._pres_idx]))
            if len(amps) >= n:
                break
        self._abs_pos += len(amps)
        return amps, pres

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._reader = None


class WavReader:
    """Streams a WAV file via `wave.readframes`. Mono PCM only."""

    def __init__(self, path: Path):
        self._path = path
        self._wf = wave.open(str(path), "rb")
        n_channels = self._wf.getnchannels()
        if n_channels != 1:
            self._wf.close()
            raise ValueError(
                f"{path}: expected mono WAV, got {n_channels} channels. "
                f"Mix down or split channels before replay."
            )
        self._sample_width = self._wf.getsampwidth()
        if self._sample_width not in (1, 2, 3, 4):
            self._wf.close()
            raise ValueError(
                f"{path}: unsupported PCM sample width {self._sample_width} bytes "
                f"({self._sample_width * 8} bits). Supported: 8, 16, 24, 32-bit PCM."
            )
        self.sample_rate = float(self._wf.getframerate())
        self.total_samples = self._wf.getnframes()
        self.has_present = False
        self._abs_pos = 0

    def seek(self, idx: int) -> None:
        if idx < 0 or idx > self.total_samples:
            raise ValueError(f"seek out of range: {idx}/{self.total_samples}")
        self._wf.setpos(idx)
        self._abs_pos = idx

    def read(self, n: int) -> tuple[list, list | None]:
        raw = self._wf.readframes(n)
        sw = self._sample_width
        if sw == 1:
            amps = [int(s) - 128 for s in struct.unpack(f"<{len(raw)}B", raw)]
        elif sw == 2:
            amps = list(struct.unpack(f"<{len(raw) // 2}h", raw))
        elif sw == 4:
            amps = list(struct.unpack(f"<{len(raw) // 4}i", raw))
        else:  # sw == 3, 24-bit
            amps = []
            for i in range(0, len(raw), 3):
                b0, b1, b2 = raw[i], raw[i + 1], raw[i + 2]
                v = b0 | (b1 << 8) | (b2 << 16)
                if v & 0x800000:
                    v -= 0x1000000
                amps.append(v)
        amps = [float(a) for a in amps]
        self._abs_pos += len(amps)
        return amps, None

    def close(self):
        if self._wf is not None:
            self._wf.close()
            self._wf = None


def _infer_rate(timestamps) -> float | None:
    """Median 1/dt across the first ~1000 rows, or None if can't infer."""
    n = min(1000, len(timestamps) - 1)
    if n <= 0:
        return None
    deltas = [timestamps[i + 1] - timestamps[i] for i in range(n)]
    deltas = [d for d in deltas if d > 0]
    if not deltas:
        return None
    return 1.0 / sorted(deltas)[len(deltas) // 2]


def open_recording(path: Path, columns: dict[str, str], no_header: bool):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return ParquetReader(path, columns)
    if suffix == ".csv":
        return CsvReader(path, columns, no_header)
    if suffix == ".wav":
        return WavReader(path)
    raise ValueError(
        f"unsupported file format {suffix!r} for {path}; "
        f"supported: .parquet, .csv, .wav"
    )


def majority_presence(window_presence: list[bool]) -> bool:
    if not window_presence:
        return False
    return sum(window_presence) * 2 > len(window_presence)


# -----------------------------------------------------------------------------
# ROS2 node
# -----------------------------------------------------------------------------


class ReplayNode(Node):
    def __init__(self, args, audio_reader, seis_reader):
        super().__init__("replay_publisher")
        self.args = args

        self.audio_reader = audio_reader
        self.seis_reader = seis_reader
        self.has_gt = audio_reader.has_present or seis_reader.has_present

        # Single bundled-channel publisher: one JSON message per tick
        # carries both audio and seismic readings on args.topic.
        self.pub = self.create_publisher(String, args.topic, 10)

        self.audio_per_tick = max(1, int(args.audio_rate / TICK_HZ))
        self.seismic_per_tick = max(1, int(args.seismic_rate / TICK_HZ))

        # Seek both readers to the configured start. Validated and
        # tick-aligned by main() before reaching this constructor.
        self.start_audio_idx = int(args.start_second * args.audio_rate)
        self.start_seismic_idx = int(args.start_second * args.seismic_rate)
        self.audio_reader.seek(self.start_audio_idx)
        self.seis_reader.seek(self.start_seismic_idx)

        # Per-window presence accumulators. The Ingestor's window is
        # WINDOW_SEC long (= TICK_HZ ticks). We collect presence flags
        # tick-by-tick and apply majority vote at window-first-tick to
        # match the previous behavior, except now we read the slice
        # incrementally as it streams in instead of indexing into
        # pre-loaded arrays.
        self._win_audio_pres: list[bool] = []
        self._win_seis_pres: list[bool] = []

        self.tick_count = 0
        self.start_wall = time.time()

        self.lock = threading.Lock()
        self.window_state: dict[float, dict] = {}
        self.results: list[dict] = []

        # Two callback groups so the timer and the subscription can run
        # concurrently under MultiThreadedExecutor. The default
        # SingleThreadedExecutor (rclpy.spin) serializes everything: the
        # 10Hz timer's per-tick work (publishing 1600 audio + 10 seismic
        # samples) starves the subscription, which then never sees
        # /inference_result messages even though they're being delivered
        # by DDS. Putting the subscription on its own group lets it run
        # in parallel with the timer.
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()
        self._sub_cb_group = ReentrantCallbackGroup()

        if not args.no_subscribe:
            # Explicit QoS matching egress's publisher (RELIABLE / VOLATILE,
            # depth=10). Avoids reliance on rclpy default-QoS behavior,
            # which differs subtly between ROS2 distros.
            sub_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            self.create_subscription(
                InferenceResult,
                args.inference_topic,
                self._on_inference,
                sub_qos,
                callback_group=self._sub_cb_group,
            )

        self.timer = self.create_timer(
            1.0 / TICK_HZ, self._tick, callback_group=self._timer_cb_group
        )
        max_secs = self._max_duration()
        if args.duration:
            max_secs = min(max_secs, float(args.duration))
        self.deadline = self.start_wall + max_secs
        self.get_logger().info(
            f"replaying {max_secs:.1f}s @ {TICK_HZ}Hz "
            f"(audio {self.audio_per_tick}/tick, seismic {self.seismic_per_tick}/tick, "
            f"GT={'yes' if self.has_gt else 'no'})"
        )

    def _max_duration(self) -> float:
        # Remaining duration from the configured start point — not the
        # absolute file length. --duration caps replay length *from*
        # --start-second, so this method has to do the same.
        audio_remaining = (
            self.audio_reader.total_samples - self.start_audio_idx
        ) / self.args.audio_rate
        seis_remaining = (
            self.seis_reader.total_samples - self.start_seismic_idx
        ) / self.args.seismic_rate
        return min(audio_remaining, seis_remaining)

    def _tick(self):
        now = time.time()
        if now >= self.deadline:
            self.get_logger().info("replay complete; spinning down")
            self.summarize()
            self.timer.cancel()
            rclpy.shutdown()
            return

        a_chunk, a_pres = self.audio_reader.read(self.audio_per_tick)
        s_chunk, s_pres = self.seis_reader.read(self.seismic_per_tick)
        if len(a_chunk) < self.audio_per_tick or len(s_chunk) < self.seismic_per_tick:
            self.get_logger().info("recording exhausted; spinning down")
            self.summarize()
            self.timer.cancel()
            rclpy.shutdown()
            return

        # Accumulate presence for the current window. The pre-streaming
        # version computed GT at window-FIRST-tick by indexing into a
        # pre-loaded array, which is impossible without the full file
        # in memory. Now we collect presence flags as they stream in
        # and apply the majority vote at window-LAST-tick — covers the
        # exact same ticks (0..TICK_HZ-1) of the same window, just
        # finalized ~1s later within that window.
        if a_pres is not None:
            self._win_audio_pres.extend(a_pres)
        if s_pres is not None:
            self._win_seis_pres.extend(s_pres)

        is_window_first_tick = (self.tick_count % TICK_HZ) == 0
        is_window_last_tick = (self.tick_count % TICK_HZ) == TICK_HZ - 1

        if is_window_first_tick:
            with self.lock:
                self.window_state[round(now, 3)] = {
                    "publish_wall": now,
                    "last_tick_wall": None,  # filled at window-last-tick
                    "gt_present": None,      # filled at window-last-tick
                    "matched": False,
                }
        # One bundled JSON message per tick: matches the customer's wire
        # format and the cluster's channels.yaml tags (default MIC / EHZ).
        utc = _dt.datetime.fromtimestamp(now, tz=_dt.timezone.utc).isoformat()
        doc = {
            "sensor_id": self.args.sensor_id,
            "state": "background",
            "timestamp_unix": now,
            "timestamp_utc": utc,
            "channels": [
                {
                    "channel": self.args.audio_tag,
                    "sampling_rate": int(self.args.audio_rate),
                    "dt": 1.0 / self.args.audio_rate,
                    "readings": [float(x) for x in a_chunk],
                },
                {
                    "channel": self.args.seismic_tag,
                    "sampling_rate": int(self.args.seismic_rate),
                    "dt": 1.0 / self.args.seismic_rate,
                    "readings": [float(x) for x in s_chunk],
                },
            ],
        }
        msg = String()
        msg.data = json.dumps(doc)
        self.pub.publish(msg)

        if is_window_last_tick:
            # The window's last tick — Ingestor has all the samples it
            # needs to emit. Latency from here to result arrival is the
            # true end-to-end pipeline latency.
            window_idx = self.tick_count // TICK_HZ
            window_first_wall = self.start_wall + window_idx * WINDOW_SEC
            gt = None
            if self.has_gt:
                gt = (
                    majority_presence(self._win_audio_pres)
                    or majority_presence(self._win_seis_pres)
                )
            self._win_audio_pres = []
            self._win_seis_pres = []
            with self.lock:
                # Find the most recent unfinalized window-state entry
                # whose publish_wall is closest to window_first_wall.
                best_k, best_dt = None, float("inf")
                for k, v in self.window_state.items():
                    if v["last_tick_wall"] is not None:
                        continue
                    dt = abs(v["publish_wall"] - window_first_wall)
                    if dt < best_dt:
                        best_k, best_dt = k, dt
                if best_k is not None:
                    self.window_state[best_k]["last_tick_wall"] = now
                    self.window_state[best_k]["gt_present"] = gt

        self.tick_count += 1

    def _on_inference(self, msg: InferenceResult):
        recv_wall = time.time()
        key = round(msg.timestamp, 3)
        with self.lock:
            entry = self.window_state.get(key)
            if entry is None:
                # Inference for a window we don't have a record of —
                # could be a stale message from a prior run still in
                # flight on JetStream (60s max-age) or sub-millisecond
                # rounding. Try a near-neighbor lookup; the timestamps
                # round-trip through several time.time() calls, so allow
                # ±100ms before giving up.
                best = None
                best_dt = float("inf")
                for k in list(self.window_state.keys()):
                    if self.window_state[k]["matched"]:
                        continue
                    dt = abs(k - key)
                    if dt < best_dt and dt < 0.1:
                        best = k
                        best_dt = dt
                if best is not None:
                    entry = self.window_state[best]
                    key = best
            if entry is None or entry["matched"]:
                return
            entry["matched"] = True
            # End-to-end pipeline latency: time from when the last
            # sample of the window was published (Ingestor's earliest
            # possible window-emit moment) to when the result arrived
            # back at the replay subscriber. If the last-tick-wall
            # wasn't recorded (e.g. last window of a truncated replay),
            # fall back to publish_wall + WINDOW_SEC as the reference.
            ref_wall = (
                entry["last_tick_wall"]
                if entry["last_tick_wall"] is not None
                else entry["publish_wall"] + WINDOW_SEC
            )
            latency = recv_wall - ref_wall
            # Egress stamps msg.latency_seconds = publish_time - capture_time,
            # which excludes the ROS2 delivery hop back to this replay node.
            # Older egress builds default it to 0.0; treat that as unavailable.
            cluster_latency = float(msg.latency_seconds) if msg.latency_seconds > 0 else None
            self.results.append(
                {
                    "publish_wall": entry["publish_wall"],
                    "gt": entry["gt_present"],
                    "pred": bool(msg.vehicle_detected),
                    "det_conf": float(msg.detection_confidence),
                    "cls": str(msg.vehicle_class),
                    "cls_conf": float(msg.classification_confidence),
                    "latency_s": latency,
                    "cluster_latency_s": cluster_latency,
                }
            )
        rel_t = entry["publish_wall"] - self.start_wall
        cls_str = msg.vehicle_class if msg.vehicle_detected else "—"
        gt_str = "?"
        if entry["gt_present"] is True:
            gt_str = "P"
        elif entry["gt_present"] is False:
            gt_str = "A"
        cluster_str = (
            f" cluster={cluster_latency * 1000:6.1f}ms"
            if cluster_latency is not None
            else ""
        )
        self.get_logger().info(
            f"t={rel_t:6.1f}s GT={gt_str} "
            f"pred={'P' if msg.vehicle_detected else 'A'} "
            f"det={msg.detection_confidence:.2f} cls={cls_str:>10s} "
            f"latency={latency * 1000:6.1f}ms{cluster_str}"
        )

    def summarize(self):
        with self.lock:
            results = list(self.results)
            unmatched = sum(1 for v in self.window_state.values() if not v["matched"])

        n = len(results)
        if n == 0:
            self.get_logger().warning("no inference results received")
            return

        def _stats(values_ms):
            sv = sorted(values_ms)
            return {
                "mean": statistics.mean(sv),
                "p50": sv[len(sv) // 2],
                "p95": sv[int(len(sv) * 0.95)],
                "p99": sv[int(len(sv) * 0.99)],
            }

        client_stats = _stats([r["latency_s"] * 1000 for r in results])
        cluster_values = [
            r["cluster_latency_s"] * 1000
            for r in results
            if r["cluster_latency_s"] is not None
        ]
        cluster_stats = _stats(cluster_values) if cluster_values else None

        print()
        print("=" * 60)
        print(f"replay summary  ({n} matched windows, {unmatched} unmatched)")
        print("=" * 60)
        if self.has_gt:
            scored = [r for r in results if r["gt"] is not None]
            tp = sum(1 for r in scored if r["gt"] and r["pred"])
            fp = sum(1 for r in scored if not r["gt"] and r["pred"])
            fn = sum(1 for r in scored if r["gt"] and not r["pred"])
            tn = sum(1 for r in scored if not r["gt"] and not r["pred"])
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / (tp + fn) if (tp + fn) else float("nan")
            print("presence (per-window)")
            print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
            print(f"  precision={prec:.3f}  recall={rec:.3f}")
            print()
        else:
            print("(no ground-truth column found; presence scoring skipped)")
            print()
        print("end-to-end latency (publish-window-end → inference receive)")
        print(f"  client : mean={client_stats['mean']:6.1f}ms  "
              f"p50={client_stats['p50']:6.1f}ms  "
              f"p95={client_stats['p95']:6.1f}ms  "
              f"p99={client_stats['p99']:6.1f}ms")
        if cluster_stats is not None:
            print("cluster latency (capture timestamp → egress publish, "
                  "excludes ROS2 return hop)")
            print(f"  cluster: mean={cluster_stats['mean']:6.1f}ms  "
                  f"p50={cluster_stats['p50']:6.1f}ms  "
                  f"p95={cluster_stats['p95']:6.1f}ms  "
                  f"p99={cluster_stats['p99']:6.1f}ms  "
                  f"(n={len(cluster_values)})")
        else:
            print("(cluster latency unavailable — egress build pre-dates "
                  "InferenceResult.latency_seconds)")
        print("=" * 60)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Paired audio recording (.parquet | .csv | .wav)",
    )
    p.add_argument(
        "--seismic",
        type=Path,
        required=True,
        help="Paired seismic recording (.parquet | .csv | .wav)",
    )
    p.add_argument(
        "--start-second",
        type=float,
        default=0.0,
        help=(
            "Start replay at this many seconds into the recording "
            "(default: 0). Validated against both files' lengths; the "
            "script errors out before publishing if the offset is "
            "negative or >= the shorter recording's duration."
        ),
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Cap replay length to this many seconds *from --start-second* (default: play to end).",
    )
    p.add_argument("--audio-rate", type=int, default=16000,
                   help="Audio sample rate in Hz. Used as authoritative when set; otherwise inferred from time_stamp.")
    p.add_argument("--seismic-rate", type=int, default=100, help="Seismic sample rate in Hz.")
    p.add_argument("--topic", default="/shake_001/data",
                   help="Bundled-channel topic the customer's array publishes on. "
                        "Must match expected-sensors.yaml.")
    p.add_argument("--audio-tag", default="MIC",
                   help="Channel tag for audio in the JSON payload. Must match a key "
                        "in the cluster's channels.yaml.")
    p.add_argument("--seismic-tag", default="EHZ",
                   help="Channel tag for seismic in the JSON payload. Must match a key "
                        "in the cluster's channels.yaml.")
    p.add_argument("--sensor-id", default="shake_001",
                   help="Per-msg sensor_id; logged for diagnostics. The Ingestor "
                        "uses SENSOR_ARRAY (set by Discovery) for actual routing.")
    p.add_argument("--inference-topic", default="/inference_result",
                   help="ROS2 topic to subscribe to for InferenceResult scoring.")
    p.add_argument("--no-subscribe", action="store_true",
                   help="Skip InferenceResult subscription and scoring.")
    p.add_argument("--column-map", default=None,
                   help="Comma-separated overrides: 'amplitude=foo,time_stamp=bar,present=baz'. "
                        "Only required when input files use non-default column names.")
    p.add_argument("--no-header", action="store_true",
                   help="CSV only: file has no header row; columns are positional "
                        "(amplitude, time_stamp, present).")
    return p.parse_args()


def main():
    args = parse_args()
    columns = parse_column_map(args.column_map)

    # Open streaming readers. Each reader's __init__ does a cheap
    # metadata pass (parquet: ParquetFile.metadata.num_rows; CSV:
    # line-count; WAV: getnframes()) and stops there — the file body is
    # not loaded into memory.
    print(f"[replay] opening {args.audio.name}", flush=True)
    audio_reader = open_recording(args.audio, columns, args.no_header)
    print(f"[replay] opening {args.seismic.name}", flush=True)
    seis_reader = open_recording(args.seismic, columns, args.no_header)

    if (
        audio_reader.sample_rate is not None
        and abs(audio_reader.sample_rate - args.audio_rate) > 1
    ):
        print(f"WARN: audio inferred rate {audio_reader.sample_rate:.1f} Hz != "
              f"declared {args.audio_rate} Hz; using declared", file=sys.stderr)
    if (
        seis_reader.sample_rate is not None
        and abs(seis_reader.sample_rate - args.seismic_rate) > 1
    ):
        print(f"WARN: seismic inferred rate {seis_reader.sample_rate:.1f} Hz != "
              f"declared {args.seismic_rate} Hz; using declared", file=sys.stderr)

    # Validate --start-second against both files' lengths via the
    # metadata-only sample counts. Done before any seek so we error out
    # before doing any cluster work.
    audio_secs = audio_reader.total_samples / args.audio_rate
    seis_secs = seis_reader.total_samples / args.seismic_rate
    shortest_secs = min(audio_secs, seis_secs)
    if args.start_second < 0:
        print(
            f"ERROR: --start-second {args.start_second} is negative.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.start_second >= shortest_secs:
        which = "audio" if audio_secs <= seis_secs else "seismic"
        print(
            f"ERROR: --start-second {args.start_second:.3f} is past the "
            f"shorter recording ({which}: {shortest_secs:.3f}s). "
            f"Pick a value in [0, {shortest_secs:.3f}).",
            file=sys.stderr,
        )
        sys.exit(2)
    # Snap to the nearest whole tick so audio + seismic stay aligned
    # at TICK_HZ resolution. The two streams have different sample
    # rates, so an arbitrary float start could put them on different
    # ticks and skew the GT-window mapping.
    snapped = round(args.start_second * TICK_HZ) / TICK_HZ
    if abs(snapped - args.start_second) > 1e-6:
        print(
            f"NOTE: snapping --start-second {args.start_second:.3f} to "
            f"{snapped:.3f} (nearest 1/{TICK_HZ}s tick).",
            file=sys.stderr,
        )
    args.start_second = snapped
    if args.start_second > 0:
        remaining = shortest_secs - args.start_second
        print(
            f"[replay] starting {args.start_second:.3f}s into recording "
            f"({remaining:.1f}s remaining)",
            flush=True,
        )

    rclpy.init()
    node = ReplayNode(args, audio_reader, seis_reader)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.summarize()
    finally:
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        audio_reader.close()
        seis_reader.close()


if __name__ == "__main__":
    main()
