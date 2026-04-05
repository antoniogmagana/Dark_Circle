import argparse
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from causal_dataset import AcousticSeismicCausalDataset
from causal_vae import CausalVAE

def parse_args():
    parser = argparse.ArgumentParser(description="Train Causal Representation Learning Model")
    parser.add_argument("--data-dir", type=str, default="./parsed/train", help="Directory containing parquet files")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda-slow", type=float, default=5.0, help="Weight for temporal slowness")
    parser.add_argument("--beta-kl", type=float, default=1.0, help="Weight for KL divergence")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()

def main():
    args = parse_args()

    # Hardware Detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU (CUDA): {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected, using CPU.")

    # 1. Load Data (Cached in RAM)
    dataset = AcousticSeismicCausalDataset(parquet_dir=args.data_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False 
    )
    num_sensors = len(dataset.sensor_to_id)

    # 2. Init Model
    model = CausalVAE(num_sensors=num_sensors, z_veh_dim=16, z_env_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. Init GPU Audio Processor
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    ).to(device)

    # 4. Training Loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        
        for batch_idx, (sig_t, sig_next, u, labels) in enumerate(dataloader):
            # Move raw audio to GPU
            sig_t, sig_next, u = sig_t.to(device), sig_next.to(device), u.to(device)
            
            # Transform to Spectrograms ON THE GPU
            x_t = spectrogram_transform(sig_t)
            x_next = spectrogram_transform(sig_next)
            
            # Log scale for neural network stability
            x_t = torch.log(x_t + 1e-9)
            x_next = torch.log(x_next + 1e-9)
            
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, mu_t, logvar_t, z_t, z_next = model(x_t, x_next, u)
            
            # 1. Reconstruction Loss
            loss_recon = F.mse_loss(x_recon, x_t, reduction='sum') / x_t.size(0)
            
            # Partition the latent space
            mu_veh, mu_env = torch.split(mu_t, [model.z_veh_dim, model.z_env_dim], dim=1)
            logvar_veh, logvar_env = torch.split(logvar_t, [model.z_veh_dim, model.z_env_dim], dim=1)
            
            # 2. KL Divergence for Vehicle
            loss_kl_veh = -0.5 * torch.sum(1 + logvar_veh - mu_veh.pow(2) - logvar_veh.exp()) / x_t.size(0)
            
            # 3. KL Divergence for Environment
            prior_mu_env = model.env_prior_mu(u)
            prior_logvar_env = model.env_prior_logvar(u)
            
            logvar_diff = prior_logvar_env - logvar_env
            loss_kl_env = 0.5 * torch.sum(
                logvar_diff - 1 + (logvar_env.exp() + (mu_env - prior_mu_env).pow(2)) / prior_logvar_env.exp()
            ) / x_t.size(0)
            
            # 4. Slow Loss
            z_env_t = z_t[:, model.z_veh_dim:]
            z_env_next = z_next[:, model.z_veh_dim:]
            loss_slow = F.mse_loss(z_env_t, z_env_next) * args.lambda_slow
            
            # Combine and Backprop
            loss = loss_recon + args.beta_kl * (loss_kl_veh + loss_kl_env) + loss_slow
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Recon: {loss_recon:.2f} | KL_Veh: {loss_kl_veh:.2f} | KL_Env: {loss_kl_env:.2f} | Slow: {loss_slow:.2f}")

        print(f"==== Epoch {epoch} Avg Loss: {total_loss/len(dataloader):.4f} ====")

    # Save the weights
    torch.save(model.state_dict(), "causal_vehicle_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()