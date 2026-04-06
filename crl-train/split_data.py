import os
import random
import shutil
from pathlib import Path


def split_dataset(parsed_dir="./parsed", train_ratio=0.85):
    """
    Moves parquet files into causal splits:
    - test_iobt: All unseen zero-shot domains.
    - train: 85% of focal and m3nvc domains.
    - val: 15% of focal and m3nvc domains.
    """
    base_path = Path(parsed_dir)

    # Create the split directories
    train_dir = base_path / "train"
    val_dir = base_path / "val"
    test_iobt_dir = base_path / "test_iobt"

    for d in [train_dir, val_dir, test_iobt_dir]:
        d.mkdir(exist_ok=True)

    # Grab all parquet files currently in the root of the parsed directory
    all_files = [f for f in base_path.glob("*.parquet") if f.is_file()]

    iobt_files = []
    crl_training_files = []

    # Separate IOBT from FOCAL/M3NVC
    for f in all_files:
        if f.name.startswith("iobt_"):
            iobt_files.append(f)
        elif f.name.startswith("focal_") or f.name.startswith("m3nvc_"):
            crl_training_files.append(f)

    # 1. Move all IOBT files directly to the test folder
    print(f"Moving {len(iobt_files)} IOBT files to test_iobt...")
    for f in iobt_files:
        shutil.move(str(f), str(test_iobt_dir / f.name))

    # 2. Shuffle and split the Focal/M3NVC files for CRL training/validation
    # We use a fixed seed so the random split is identical if you ever re-run this

    # Group files by their base event (dataset + everything except sensor modality)
    # e.g. "m3nvc_audio_cx30_rs1" -> "m3nvc_cx30_rs1"
    groups = {}
    for f in crl_training_files:
        parts = f.stem.split("_", 2)
        if len(parts) == 3:
            dataset, sensor, rest = parts
            group_key = f"{dataset}_{rest}"
        else:
            group_key = f.stem
        groups.setdefault(group_key, []).append(f)

    group_keys = list(groups.keys())
    random.seed(42)
    random.shuffle(group_keys)

    split_idx = int(len(group_keys) * train_ratio)
    train_keys = group_keys[:split_idx]
    val_keys = group_keys[split_idx:]

    train_files = [f for k in train_keys for f in groups[k]]
    val_files = [f for k in val_keys for f in groups[k]]

    print(f"Moving {len(train_files)} files to train...")
    for f in train_files:
        shutil.move(str(f), str(train_dir / f.name))

    print(f"Moving {len(val_files)} files to val...")
    for f in val_files:
        shutil.move(str(f), str(val_dir / f.name))

    print("\nSplit Complete!")
    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files")
    print(f"Test (IOBT): {len(iobt_files)} files")


if __name__ == "__main__":
    # Point this to your parsed data directory
    split_dataset(parsed_dir="../data/parsed")
