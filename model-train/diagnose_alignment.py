"""
diagnose_alignment.py — Standalone diagnostic for sample count issues.

Replays _get_table_max_time() and _align_max_time() from dataset.py
and reports data retained vs lost when aligning sensor timestamps.

Run from the model-train/ directory:
    DB_PASSWORD=<pw> TRAINING_MODE=detection MODEL_NAME=TCN \
        poetry run python diagnose_alignment.py
"""

import math
from collections import defaultdict

import config
from db_utils import db_connect, db_close, get_time_bounds


# Flag groups where overlap < 90% of any individual sensor's duration.
SHRINK_THRESHOLD = 0.90


def fetch_all_tables(cursor, datasets):
    tables = []
    for dataset in datasets:
        cursor.execute(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname='public' AND tablename LIKE %s;",
            (f"{dataset}_%",),
        )
        tables.extend(row[0] for row in cursor.fetchall())
    return tables


def get_run_ids(cursor, table):
    """
    Return [None] for non-m3nvc tables, or the list of
    distinct run_ids for m3nvc tables.
    """
    if table.startswith("m3nvc_"):
        try:
            cursor.execute(
                "SELECT DISTINCT run_id FROM "
                f"{table} WHERE run_id IS NOT NULL;"
            )
            runs = [row[0] for row in cursor.fetchall()]
            return runs if runs else [None]
        except Exception:
            cursor.connection.rollback()
            return [None]
    return [None]


def build_time_bounds(cursor, tables):
    """Returns {(table, run_id): (min_t, max_t)} for all tables."""
    bounds = {}
    total = len(tables)
    for i, table in enumerate(tables, 1):
        run_ids = get_run_ids(cursor, table)
        for run_id in run_ids:
            min_t, max_t = get_time_bounds(cursor, table, run_id=run_id)
            bounds[(table, run_id)] = (min_t, max_t)
        if i % 20 == 0 or i == total:
            print(f"  Queried {i}/{total} tables...", flush=True)
    return bounds


def group_bounds(bounds):
    """
    Group (table, run_id) entries by
    (dataset, instance, sensor_node, run_id).
    Returns {group_key: [(table, run_id, signal), ...]}
    """
    groups = defaultdict(list)
    for (table, run_id), _ in bounds.items():
        parts = table.split("_")
        dataset = parts[0]
        signal = parts[1]
        instance = "_".join(parts[2:-1])
        sensor_node = parts[-1]
        group_key = (dataset, instance, sensor_node, run_id)
        groups[group_key].append((table, run_id, signal))
    return groups


def count_windows(duration_sec, sample_seconds=1):
    return math.floor(duration_sec / sample_seconds)


def est_test(total_windows, block_size, usable_size, split_test):
    """
    Rough estimate of test samples: usable fraction × split fraction.
    No block-boundary effects accounted for.
    """
    usable = total_windows * (usable_size / block_size)
    return int(usable * split_test)


def main():
    sample_seconds = config.SAMPLE_SECONDS
    train_datasets = config.TRAIN_DATASETS
    train_sensors = config.TRAIN_SENSORS
    block_size = config.BLOCK_SIZE
    usable_size = config.USABLE_SIZE
    split_test = config.SPLIT_TEST

    print("\nDiagnostic Config:")
    print(f"  TRAIN_DATASETS  : {train_datasets}")
    print(f"  TRAIN_SENSORS   : {train_sensors}")
    print(f"  BLOCK_SIZE      : {block_size}")
    print(f"  USABLE_SIZE     : {usable_size}")
    print(f"  SPLIT_TEST      : {split_test}")
    print(f"  SAMPLE_SECONDS  : {sample_seconds}")
    print()

    print("Connecting to database...")
    conn, cursor = db_connect(config.DB_CONN_PARAMS)

    print("Fetching table list...")
    tables = fetch_all_tables(cursor, train_datasets)
    print(f"Found {len(tables)} tables total.\n")

    print("Querying time bounds for all tables...")
    bounds = build_time_bounds(cursor, tables)
    print()

    groups = group_bounds(bounds)

    # -------------------------------------------------------------------
    # Per-group alignment report
    # -------------------------------------------------------------------
    print("=" * 70)
    print("ALIGNMENT REPORT (groups with shrinkage flagged)")
    print("=" * 70)

    # Keys: "intersect", and one per sensor name
    dataset_windows = defaultdict(lambda: defaultdict(int))
    dataset_group_counts = defaultdict(int)
    n_shrunk = 0

    for group_key, members in sorted(groups.items()):
        dataset, instance, sensor_node, run_id = group_key
        dataset_group_counts[dataset] += 1

        signal_bounds = {}
        for table, member_run_id, signal in members:
            if signal not in train_sensors:
                continue
            min_t, max_t = bounds[(table, member_run_id)]
            signal_bounds[signal] = (min_t, max_t)

        present_signals = list(signal_bounds.keys())
        has_all = all(s in present_signals for s in train_sensors)

        if not has_all:
            missing = [s for s in train_sensors if s not in present_signals]
            print(f"\n[{dataset} / {instance} / {sensor_node} / {run_id}]")
            print(f"  EXCLUDED: missing signals {missing}")
            continue

        intersect_min = max(v[0] for v in signal_bounds.values())
        intersect_max = min(v[1] for v in signal_bounds.values())
        intersect_dur = max(0.0, intersect_max - intersect_min)
        intersect_w = count_windows(intersect_dur, sample_seconds)

        sensor_windows = {}
        for sig, (mn, mx) in signal_bounds.items():
            dur = max(0.0, mx - mn)
            sensor_windows[sig] = (mn, mx, count_windows(dur, sample_seconds))

        shrink_msgs = []
        for sig, (mn, mx, ind_w) in sensor_windows.items():
            if ind_w > 0 and intersect_w < ind_w * SHRINK_THRESHOLD:
                pct_lost = 100.0 * (ind_w - intersect_w) / ind_w
                shrink_msgs.append(
                    f"{sig} loses {pct_lost:.1f}% of windows"
                )

        if shrink_msgs:
            n_shrunk += 1
            print(
                f"\n[{dataset} / {instance} / {sensor_node} / {run_id}]"
            )
            for sig, (mn, mx, ind_w) in sorted(sensor_windows.items()):
                print(
                    f"  {sig:<8}: t={mn:.1f} → {mx:.1f}"
                    f"  ({ind_w} windows)"
                )
            print(
                f"  overlap : t={intersect_min:.1f} → {intersect_max:.1f}"
                f"  ({intersect_w} windows)"
            )
            for msg in shrink_msgs:
                print(f"  *** SHRINKAGE: {msg} ***")

        dataset_windows[dataset]["intersect"] += intersect_w
        for sig, (mn, mx, ind_w) in sensor_windows.items():
            dataset_windows[dataset][sig] += ind_w

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Groups with significant timestamp shrinkage: {n_shrunk}")
    print()

    all_sensors = sorted(train_sensors)

    header = f"{'Dataset':<10} {'Groups':>6} {'Windows(intersect)':>18}"
    for s in all_sensors:
        header += f"  {'Windows(' + s + ')':>16}"
    header += f"  {'Est.Test(intersect)':>20}"
    for s in all_sensors:
        header += f"  {'Est.Test(' + s + ')':>14}"
    print(header)
    print("-" * len(header))

    totals = defaultdict(int)
    for dataset in sorted(dataset_windows.keys()):
        dw = dataset_windows[dataset]
        intersect_w = dw["intersect"]
        n_groups = dataset_group_counts[dataset]
        row = f"{dataset:<10} {n_groups:>6} {intersect_w:>18,}"
        for s in all_sensors:
            row += f"  {dw.get(s, 0):>16,}"
        est_i = est_test(intersect_w, block_size, usable_size, split_test)
        row += f"  {est_i:>20,}"
        for s in all_sensors:
            est_s = est_test(
                dw.get(s, 0), block_size, usable_size, split_test
            )
            row += f"  {est_s:>14,}"
        print(row)
        totals["intersect"] += intersect_w
        for s in all_sensors:
            totals[s] += dw.get(s, 0)

    print("-" * len(header))
    total_groups = sum(dataset_group_counts.values())
    row = (
        f"{'TOTAL':<10} {total_groups:>6}"
        f" {totals['intersect']:>18,}"
    )
    for s in all_sensors:
        row += f"  {totals.get(s, 0):>16,}"
    est_i = est_test(
        totals["intersect"], block_size, usable_size, split_test
    )
    row += f"  {est_i:>20,}"
    for s in all_sensors:
        est_s = est_test(
            totals.get(s, 0), block_size, usable_size, split_test
        )
        row += f"  {est_s:>14,}"
    print(row)

    print()
    print("Note: 'Est.Test' assumes uniform block split distribution.")
    print("      Actual counts vary due to deterministic random assignment.")

    db_close(conn, cursor)


if __name__ == "__main__":
    main()
