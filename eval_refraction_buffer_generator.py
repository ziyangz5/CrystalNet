import torch
import numpy as np
import argparse
import logging
from torch.utils.data import DataLoader, TensorDataset
from utils import DL1Combine, load_checkpoint
from net_models import CrystalNet as RBufferGenerator

def load_data(ds_name, obj_num, batch_size, res=256):
    """Load validation data for evaluation."""
    # Load data
    G = torch.load(f"./datasets/{ds_name}/{ds_name}_RAOV_Xg_{res}.pt")
    X = torch.load(f"./datasets/{ds_name}/{ds_name}_RAOV_X_{res}.pt")
    X = torch.moveaxis(X, 3, 1)
    
    Rnpz = np.load(f"./datasets/{ds_name}/{ds_name}_RAOV_{res}.npz")
    Rnp = np.rollaxis(Rnpz["RAOV"], 3, 1)
    Rnp[:, 0, ...] = np.round(Rnp[:, 0, ...])
    Rnp[:, 0, ...] += 1
    mask = np.expand_dims(np.clip(np.round(torch.sum(torch.abs(G[:, :, 8, :, :]), dim=1).numpy()), 0, 1), axis=1)
    Rnp = (Rnp * mask)
    Rnp[:, 0, ...][Rnp[:, 0, ...] > obj_num] = 0
    Rnp[:, 0, ...][Rnp[:, 0, ...] < 0] = 0

    Xva = X
    Gva = G
    Rva = torch.Tensor(Rnp)

    # Convert to tensors
    dataset_val = TensorDataset(Xva, Gva, Rva[:, -3:, ...], Rva[:, 0, ...], Rva[:, 1:3, ...])
    
    # DataLoader creation
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    
    return dataloader_val

def evaluate_checkpoint(checkpoint_path, dataloader_val, obj_num, device):
    """Load model checkpoint and evaluate."""
    # Load model and criterion
    net = RBufferGenerator(obj_num + 1).to(device)
    criterion = DL1Combine()
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, net, None)
    
    # Evaluate model
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for x_raw, g_val, sn_val, oi_val, uv_val in dataloader_val:
            x_val, g_val = x_raw.to(device), g_val.to(device)
            oi_val = oi_val.to(device).long()
            uv_val = uv_val.to(device)
            sn_val = sn_val.to(device)
            
            # Forward pass
            pred_sn, pred_oi, pred_uv = net(x_val, g_val)
            
            # Calculate loss
            loss = criterion(pred_sn, pred_oi, pred_uv, sn_val, oi_val, uv_val)
            val_loss += loss.item()
    
    logging.info(f"Validation Loss from checkpoint: {val_loss / len(dataloader_val):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name', type=str, required=True)
    parser.add_argument('--obj_num', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to model checkpoint in ./models/")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device: {device}')
    
    # Load Validation Data
    dataloader_val = load_data(args.scene_name, args.obj_num, args.batch_size)

    # Run Evaluation
    evaluate_checkpoint(args.checkpoint_path, dataloader_val, args.obj_num, device)

if __name__ == "__main__":
    main()
