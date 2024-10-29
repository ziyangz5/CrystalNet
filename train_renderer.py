import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import logging
from sklearn.model_selection import train_test_split
from ext import pytorch_ssim
from utils import DssimL1Loss, save_checkpoint, load_checkpoint
from net_models import CrystalRenderer
import torch.optim.lr_scheduler

def load_data(ds_name, batch_size, res=256, test_size=0.03, random_seed=42):
    """Loads and splits the dataset into train and validation sets."""
    # Load data
    Xnpz = np.load(f"./datasets/{ds_name}/{ds_name}_X_{res}.npz")
    Gnpz = np.load(f"./datasets/{ds_name}/{ds_name}_Xg_{res}.npz")
    Ynpz = np.load(f"./datasets/{ds_name}/{ds_name}_Y_{res}.npz")
    
    # Preprocess and combine data
    Xnp = np.rollaxis(Xnpz["X"], 3, 1)
    Gnp = Gnpz["Xg"]
    Ynp = np.rollaxis(Ynpz["Y"], 3, 1) / 255
    mask = np.expand_dims(np.clip(np.round(np.sum(np.abs(Gnp[:, :, 8, :, :]), axis=1)), 0, 1), axis=1)
    Xnp = np.concatenate((Xnp, mask), axis=1)
    
    # Train-validation split
    X_train, X_val, G_train, G_val, Y_train, Y_val = train_test_split(Xnp, Gnp, Ynp, test_size=test_size, random_state=random_seed)
    
    # Convert to tensors
    dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(G_train), torch.Tensor(Y_train))
    dataset_val = TensorDataset(torch.Tensor(X_val), torch.Tensor(G_val), torch.Tensor(Y_val))
    
    # DataLoader creation
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    
    return dataloader_train, dataloader_val

def train_model(net, dataloader_train, dataloader_val, optimizer, scheduler, criterion, num_epochs, device, ds_name):
    """Main training loop."""
    net.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x_raw, g, y in dataloader_train:
            x, g, y = x_raw.to(device), g.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            x_recon = net(x, g)
            loss = criterion(x_recon, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader_train)
        if epoch % 2 == 0:
            logging.info(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Validation and Checkpointing
        if epoch % 10 == 0:
            val_loss = evaluate_model(net, dataloader_val, criterion, device)
            logging.info(f"Validation Loss: {val_loss:.4f}")

        # Save model checkpoints
        if epoch % 20 == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_dir="./models", filename=f"{ds_name}_checkpoint_epoch_{epoch}.pth")

        # Step the scheduler
        scheduler.step()

def evaluate_model(net, dataloader_val, criterion, device):
    """Evaluates the model on the validation set."""
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, g_val, y_val in dataloader_val:
            x_val, g_val, y_val = x_val.to(device), g_val.to(device), y_val.to(device)
            x_recon = net(x_val, g_val)
            loss = criterion(x_recon, y_val)
            val_loss += loss.item()
    return val_loss / len(dataloader_val)

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name', type=str, required=True)
    parser.add_argument('--single_batch_size', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()
    
    # Logging Configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device: {device}')
    
    # Load Data
    batch_size = torch.cuda.device_count() * args.single_batch_size
    dataloader_train, dataloader_val = load_data(args.scene_name, batch_size)

    # Model, Optimizer, Scheduler, and Criterion
    net = CrystalRenderer().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1.0e-4, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.9)
    criterion = DssimL1Loss()
    
    # Training Loop
    train_model(net, dataloader_train, dataloader_val, optimizer, scheduler, criterion, args.num_epochs, device, args.scene_name)

if __name__ == "__main__":
    main()
