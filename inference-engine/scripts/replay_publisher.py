"""
Replay a paired (audio, seismic) recording as a live ROS2 sensor source
for end-to-end inference-engine testing.

The pipeline cannot tell this apart from a real Raspberry Shake — same
RawSensorReading messages, same topics, same rates. Optionally subscribes
back to the configured inference output topic to score predictions
against ground truth and measure end-to-end latency.

Supported file formats: parquet (.parquet), CSV (.csv). Format is
inferred from the file extension.

Required columns (defaults; override via ``--column-map``):
    amplitude     raw ADC sample (int or float)
    time_stamp    seconds-since-start (float) — used to infer sample rate
    present       bool (optional; ground truth for presence scoring)

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
package containing ``RawSensorReading`` and ``InferenceResult`` must be
on AMENT_PREFIX_PATH. See chart/README.md "Testing on pre-recorded
data" or run ``scripts/install_replay.sh`` for a guided setup.
"""

import argparse
import csv as _csv
import statistics
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros2_interfaces.msg import InferenceResult, RawSensorReading

WINDOW_SEC = 1.0          # ingestor's window size
TICK_HZ = 10              # tick rate; matches fake-publisher to keep the
                          # ingestor's holding-pen logic happy.

DEFAULT_COLUMNS = {
    "amplitude": "amplitude",
    "time_stamp": "time_stamp",
    "present": "present",
}


# -----------------------------------------------------------------------------
# Loaders
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


def load_parquet(path: Path, columns: dict[str, str]) -> tuple[list, list, float | None]:
    """Return (amplitudes, presence-per-row-or-None, inferred_sample_rate)."""
    import pyarrow.parquet as pq

    needed = [columns["amplitude"]]
    schema_names = set(pq.ParquetFile(path).schema.names)
    if columns["time_stamp"] in schema_names:
        needed.append(columns["time_stamp"])
    if columns["present"] in schema_names:
        needed.append(columns["present"])

    table = pq.read_table(path, columns=needed)
    amp = table.column(columns["amplitude"]).to_pylist()
    pres = (
        table.column(columns["present"]).to_pylist()
        if columns["present"] in needed
        else None
    )
    rate = (
        _infer_rate(table.column(columns["time_stamp"]).to_numpy())
        if columns["time_stamp"] in needed
        else None
    )
    return amp, pres, rate


def load_csv(path: Path, columns: dict[str, str], no_header: bool) -> tuple[list, list, float | None]:
    """Stream-friendly CSV reader. Avoids pandas to keep the customer's
    install footprint to ``rclpy + pyarrow``."""
    if no_header:
        # Headerless: amplitude is column 0, time_stamp column 1,
        # present column 2 (if present). Customer is responsible for ordering.
        amp_idx, ts_idx, pres_idx = 0, 1, 2
    else:
        amp_idx = ts_idx = pres_idx = None

    amplitudes: list = []
    timestamps: list = []
    presence: list = []
    has_pres = False

    with path.open(newline="") as f:
        reader = _csv.reader(f)
        if not no_header:
            header = next(reader, None)
            if header is None:
                raise ValueError(f"{path}: empty CSV")
            try:
                amp_idx = header.index(columns["amplitude"])
            except ValueError:
                raise ValueError(
                    f"{path}: missing required column '{columns['amplitude']}'; "
                    f"header was {header}"
                ) from None
            ts_idx = header.index(columns["time_stamp"]) if columns["time_stamp"] in header else None
            pres_idx = header.index(columns["present"]) if columns["present"] in header else None
            has_pres = pres_idx is not None
        else:
            has_pres = True  # headerless assumes all three are present

        for row in reader:
            if not row:
                continue
            amplitudes.append(float(row[amp_idx]))
            if ts_idx is not None and ts_idx < len(row):
                timestamps.append(float(row[ts_idx]))
            if has_pres and pres_idx is not None and pres_idx < len(row):
                presence.append(_parse_bool(row[pres_idx]))

    rate = _infer_rate(timestamps) if timestamps else None
    return amplitudes, (presence if has_pres else None), rate


def _parse_bool(raw: str) -> bool:
    s = raw.strip().lower()
    if s in ("1", "true", "t", "yes", "y", "present", "p"):
        return True
    if s in ("0", "false", "f", "no", "n", "absent", "a", ""):
        return False
    raise ValueError(f"cannot parse bool from {raw!r}")


def load_recording(
    path: Path, columns: dict[str, str], no_header: bool
) -> tuple[list, list | None, float | None]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return load_parquet(path, columns)
    if suffix == ".csv":
        return load_csv(path, columns, no_header)
    raise ValueError(
        f"unsupported file format {suffix!r} for {path}; "
        f"supported: .parquet, .csv"
    )


def majority_presence(window_presence: list[bool]) -> bool:
    if not window_presence:
        return False
    return sum(window_presence) * 2 > len(window_presence)


# -----------------------------------------------------------------------------
# ROS2 node
# -----------------------------------------------------------------------------


class ReplayNode(Node):
    def __init__(self, args, audio_amp, audio_pres, seis_amp, seis_pres):
        super().__init__("replay_publisher")
        self.args = args

        self.audio_amp = audio_amp
        self.audio_pres = audio_pres  # may be None
        self.seis_amp = seis_amp
        self.seis_pres = seis_pres    # may be None
        self.has_gt = audio_pres is not None or seis_pres is not None

        self.audio_pub = self.create_publisher(RawSensorReading, args.audio_topic, 10)
        self.seismic_pub = self.create_publisher(RawSensorReading, args.seismic_topic, 10)

        self.audio_per_tick = max(1, int(args.audio_rate / TICK_HZ))
        self.seismic_per_tick = max(1, int(args.seismic_rate / TICK_HZ))

        self.audio_idx = 0
        self.seismic_idx = 0
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
        return min(
            len(self.audio_amp) / self.args.audio_rate,
            len(self.seis_amp) / self.args.seismic_rate,
        )

    def _tick(self):
        now = time.time()
        if now >= self.deadline:
            self.get_logger().info("replay complete; spinning down")
            self.summarize()
            self.timer.cancel()
            rclpy.shutdown()
            return

        a_end = self.audio_idx + self.audio_per_tick
        s_end = self.seismic_idx + self.seismic_per_tick
        if a_end > len(self.audio_amp) or s_end > len(self.seis_amp):
            self.get_logger().info("recording exhausted; spinning down")
            self.summarize()
            self.timer.cancel()
            rclpy.shutdown()
            return

        a_chunk = self.audio_amp[self.audio_idx : a_end]
        s_chunk = self.seis_amp[self.seismic_idx : s_end]

        msg = RawSensorReading()
        msg.sensor_id = self.args.sensor_id
        msg.start_time = now
        msg.amplitude_readings = [int(x) for x in a_chunk]
        self.audio_pub.publish(msg)

        smsg = RawSensorReading()
        smsg.sensor_id = self.args.sensor_id.replace("aud", "ehz")
        smsg.start_time = now
        smsg.amplitude_readings = [int(x) for x in s_chunk]
        self.seismic_pub.publish(smsg)

        self.audio_idx = a_end
        self.seismic_idx = s_end
        self.tick_count += 1

        # Once per window (10 ticks at TICK_HZ=10), record the window's
        # ground truth + publish wall-clock so we can match it to an
        # InferenceResult later.
        if self.tick_count % TICK_HZ == 0:
            window_idx = (self.tick_count // TICK_HZ) - 1
            gt = None
            if self.has_gt:
                window_first_audio = window_idx * self.audio_per_tick * TICK_HZ
                window_first_seis = window_idx * self.seismic_per_tick * TICK_HZ
                a_pres = (
                    self.audio_pres[
                        window_first_audio : window_first_audio + int(self.args.audio_rate)
                    ]
                    if self.audio_pres
                    else []
                )
                s_pres = (
                    self.seis_pres[
                        window_first_seis : window_first_seis + int(self.args.seismic_rate)
                    ]
                    if self.seis_pres
                    else []
                )
                gt = majority_presence(a_pres) or majority_presence(s_pres)
            window_start_wall = self.start_wall + window_idx * WINDOW_SEC
            with self.lock:
                self.window_state[round(window_start_wall, 3)] = {
                    "publish_wall": window_start_wall,
                    "gt_present": gt,
                    "matched": False,
                }

    def _on_inference(self, msg: InferenceResult):
        # TEMP DEBUG: confirm callback is firing at all
        self.get_logger().info(
            f"[debug] callback fired! ts={msg.timestamp} "
            f"detected={msg.vehicle_detected} det_conf={msg.detection_confidence:.3f}"
        )
        recv_wall = time.time()
        key = round(msg.timestamp, 3)
        with self.lock:
            entry = self.window_state.get(key)
            if entry is None:
                # Inference for an unknown window — usually a slight
                # rounding mismatch. Try near-neighbor lookup within 10ms.
                for k in list(self.window_state.keys()):
                    if abs(k - key) < 0.01:
                        entry = self.window_state[k]
                        key = k
                        break
            if entry is None or entry["matched"]:
                return
            entry["matched"] = True
            latency = recv_wall - entry["publish_wall"] - WINDOW_SEC
            self.results.append(
                {
                    "publish_wall": entry["publish_wall"],
                    "gt": entry["gt_present"],
                    "pred": bool(msg.vehicle_detected),
                    "det_conf": float(msg.detection_confidence),
                    "cls": str(msg.vehicle_class),
                    "cls_conf": float(msg.classification_confidence),
                    "latency_s": latency,
                }
            )
        rel_t = entry["publish_wall"] - self.start_wall
        cls_str = msg.vehicle_class if msg.vehicle_detected else "—"
        gt_str = "?"
        if entry["gt_present"] is True:
            gt_str = "P"
        elif entry["gt_present"] is False:
            gt_str = "A"
        self.get_logger().info(
            f"t={rel_t:6.1f}s GT={gt_str} "
            f"pred={'P' if msg.vehicle_detected else 'A'} "
            f"det={msg.detection_confidence:.2f} cls={cls_str:>10s} "
            f"latency={latency * 1000:6.1f}ms"
        )

    def summarize(self):
        with self.lock:
            results = list(self.results)
            unmatched = sum(1 for v in self.window_state.values() if not v["matched"])

        n = len(results)
        if n == 0:
            self.get_logger().warning("no inference results received")
            return

        latencies_ms = sorted(r["latency_s"] * 1000 for r in results)
        p50 = latencies_ms[len(latencies_ms) // 2]
        p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
        p99 = latencies_ms[int(len(latencies_ms) * 0.99)]
        mean = statistics.mean(latencies_ms)

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
        print(f"  mean={mean:6.1f}ms  p50={p50:6.1f}ms  "
              f"p95={p95:6.1f}ms  p99={p99:6.1f}ms")
        print("=" * 60)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--audio", type=Path, required=True, help="Paired audio recording (.parquet | .csv)")
    p.add_argument("--seismic", type=Path, required=True, help="Paired seismic recording (.parquet | .csv)")
    p.add_argument("--duration", type=float, default=None, help="Cap replay length (seconds)")
    p.add_argument("--audio-rate", type=int, default=16000,
                   help="Audio sample rate in Hz. Used as authoritative when set; otherwise inferred from time_stamp.")
    p.add_argument("--seismic-rate", type=int, default=100, help="Seismic sample rate in Hz.")
    p.add_argument("--audio-topic", default="/shake_001/aud")
    p.add_argument("--seismic-topic", default="/shake_001/ehz")
    p.add_argument("--sensor-id", default="shake_001.aud",
                   help="Per-msg sensor_id; the Ingestor uses Discovery's "
                        "role-map binding instead, so this is informational.")
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

    print(f"[replay] loading {args.audio.name}", flush=True)
    audio_amp, audio_pres, audio_rate_actual = load_recording(args.audio, columns, args.no_header)
    print(f"[replay] loading {args.seismic.name}", flush=True)
    seis_amp, seis_pres, seis_rate_actual = load_recording(args.seismic, columns, args.no_header)

    if audio_rate_actual is not None and abs(audio_rate_actual - args.audio_rate) > 1:
        print(f"WARN: audio inferred rate {audio_rate_actual:.1f} Hz != "
              f"declared {args.audio_rate} Hz; using declared", file=sys.stderr)
    if seis_rate_actual is not None and abs(seis_rate_actual - args.seismic_rate) > 1:
        print(f"WARN: seismic inferred rate {seis_rate_actual:.1f} Hz != "
              f"declared {args.seismic_rate} Hz; using declared", file=sys.stderr)

    rclpy.init()
    node = ReplayNode(args, audio_amp, audio_pres, seis_amp, seis_pres)
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


if __name__ == "__main__":
    main()
