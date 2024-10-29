import torch
import argparse
import logging
from torch.utils.data import DataLoader, TensorDataset
from utils import DssimL1Loss, load_checkpoint
from net_models import CrystalRenderer

def load_data(ds_name, batch_size, res=256):
    """Load validation data for evaluation."""
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
    
    # Convert to tensors
    dataset_val = TensorDataset(torch.Tensor(Xnp), torch.Tensor(Gnp), torch.Tensor(Ynp))
    
    # DataLoader creation
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    
    return dataloader_val

def evaluate_checkpoint(checkpoint_path, dataloader_val, device):
    """Load model checkpoint and evaluate."""
    # Load model and criterion
    net = CrystalRenderer().to(device)
    criterion = DssimL1Loss()
    
    # Load checkpoint
    _, _ = load_checkpoint(checkpoint_path, net, None)
    
    # Evaluate model
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, g_val, y_val in dataloader_val:
            x_val, g_val, y_val = x_val.to(device), g_val.to(device), y_val.to(device)
            x_recon = net(x_val, g_val)
            loss = criterion(x_recon, y_val)
            val_loss += loss.item()
    
    logging.info(f"Validation Loss from checkpoint: {val_loss / len(dataloader_val):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to model checkpoint in ./models/")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device: {device}')
    
    # Load Validation Data
    dataloader_val = load_data(args.scene_name, args.batch_size)

    # Run Evaluation
    evaluate_checkpoint(args.checkpoint_path, dataloader_val, device)

if __name__ == "__main__":
    main()

