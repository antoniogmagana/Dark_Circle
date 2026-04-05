import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class AcousticSeismicCausalDataset(Dataset):
    def __init__(self, parquet_dir, sample_rate=16000, is_audio=True):
        self.sample_rate = sample_rate
        self.is_audio = is_audio
        self.window_size = sample_rate  # 1 second windows
        self.files = list(Path(parquet_dir).glob("*.parquet"))

        # Build an index of all valid consecutive 1-second windows
        self.index = []
        self.sensor_to_id = {}

        print("Indexing parquet files...")
        for file_path in self.files:
            sensor_name = file_path.stem  # e.g., 'focal_audio_mustang2_rs7'
            if sensor_name not in self.sensor_to_id:
                self.sensor_to_id[sensor_name] = len(self.sensor_to_id)

            u_id = self.sensor_to_id[sensor_name]

            # Read just metadata/length if possible, or load to RAM if small enough
            df = pd.read_parquet(file_path, columns=["present"])
            n_samples = len(df)
            n_windows = n_samples // self.window_size

            # We need pairs of (t, t+1), so we go up to n_windows - 1
            for w in range(n_windows - 1):
                start_idx = w * self.window_size
                # Store: (file_path, start_idx, sensor_id, weak_label)
                self.index.append(
                    (file_path, start_idx, u_id, df["present"].iloc[start_idx])
                )

        print(f"Total valid (t, t+1) window pairs indexed: {len(self.index)}")
        print(f"Total unique domains (sensors): {len(self.sensor_to_id)}")

        # Spectrogram converter (Runs on GPU if available later)
        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024, hop_length=512, n_mels=64
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_path, start_idx, u_id, label = self.index[idx]

        # Read the 2-second block (t and t+1 combined to save read time)
        # Note: In production, memory-mapping NumPy arrays is faster than reading parquets on the fly.
        df = pd.read_parquet(file_path).iloc[
            start_idx : start_idx + (self.window_size * 2)
        ]

        amp_col = "amplitude" if "amplitude" in df.columns else df.columns[2]
        signal = torch.tensor(df[amp_col].values, dtype=torch.float32)

        # Split into t and t+1
        sig_t = signal[: self.window_size].unsqueeze(0)
        sig_next = signal[self.window_size :].unsqueeze(0)

        # Convert to Spectrograms and add channel dim -> shape: (1, 64, 32)
        spec_t = self.spectrogram(sig_t)
        spec_next = self.spectrogram(sig_next)

        # Log scaling for better neural network processing
        spec_t = torch.log(spec_t + 1e-9)
        spec_next = torch.log(spec_next + 1e-9)

        return (
            spec_t,
            spec_next,
            torch.tensor(u_id, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )
