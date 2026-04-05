import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class AcousticSeismicCausalDataset(Dataset):
    def __init__(self, parquet_dir, sample_rate=16000, is_audio=True):
        self.sample_rate = sample_rate
        self.is_audio = is_audio
        self.window_size = sample_rate  # 1 second windows
        self.files = list(Path(parquet_dir).rglob("*.parquet"))
        
        self.index = []
        self.sensor_to_id = {}
        
        # --- THE NEW CACHE ---
        # This will hold the raw audio arrays in RAM so we never hit the hard drive during training
        self.audio_cache = {} 
        
        print(f"Loading {len(self.files)} files into RAM cache. This might take a minute...")
        
        for file_path in self.files:
            file_str = str(file_path)
            sensor_name = file_path.stem
            
            if sensor_name not in self.sensor_to_id:
                self.sensor_to_id[sensor_name] = len(self.sensor_to_id)
            
            u_id = self.sensor_to_id[sensor_name]
            
            # Read the file ONCE
            df = pd.read_parquet(file_path)
            amp_col = 'amplitude' if 'amplitude' in df.columns else df.columns[2]
            
            # Extract raw numpy array and store it in the RAM cache
            self.audio_cache[file_str] = df[amp_col].values.astype(np.float32)
            
            # Build the index
            n_samples = len(df)
            n_windows = n_samples // self.window_size
            
            for w in range(n_windows - 1):
                start_idx = w * self.window_size
                # Store: (file_path_str, start_idx, sensor_id, weak_label)
                self.index.append((file_str, start_idx, u_id, df['present'].iloc[start_idx]))

        print(f"Total valid (t, t+1) window pairs indexed: {len(self.index)}")
        print(f"Total unique domains (sensors): {len(self.sensor_to_id)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_str, start_idx, u_id, label = self.index[idx]
        
        # Instantly slice the data straight from RAM instead of reading the parquet!
        cached_signal = self.audio_cache[file_str]
        signal = torch.tensor(cached_signal[start_idx : start_idx + (self.window_size * 2)])
        
        # Split into t and t+1
        sig_t = signal[:self.window_size].unsqueeze(0)
        sig_next = signal[self.window_size:].unsqueeze(0)
        
        return sig_t, sig_next, torch.tensor(u_id, dtype=torch.long), torch.tensor(label, dtype=torch.float32)