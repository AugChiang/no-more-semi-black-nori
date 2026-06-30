import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from dataset import MangaStripeDataset
from model import DualDomainNAFNet
import torchvision.transforms as v2
from focal_frequency_loss import FocalFrequencyLoss
import os
from tqdm import tqdm
from torchvision.utils import save_image


def parse_args():
    parser = ArgumentParser(description="Train DualDomainNAFNet for Image Restoration")
    parser.add_argument("--image_dir", type=str, default="data", help="Path to the input image directory")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches to extract from the image")
    parser.add_argument("--num_patches", type=int, default=16, help="Number of patches to extract for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    return parser.parse_args()

def train():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    lr = 1e-3
    args = parse_args()
    batch_size = args.batch_size
    image_dir = args.image_dir
    patch_size = args.patch_size
    num_patches = args.num_patches
    epochs = args.epochs
    # transform
    transform = v2.Compose([
        v2.RandomCrop(256),
        v2.ToTensor(),
    ])
    
    # Dataset and Loader
    dataset = MangaStripeDataset(image_dir, patch_size=patch_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Model
    model = DualDomainNAFNet(width=32).to(device)
    
    # Loss and Optimizer
    criterion_l1 = nn.L1Loss()
    criterion_ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0) # Parameters for FFL
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_loss = float('inf')
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for i, (input_img, target) in enumerate(pbar):
            input_img = input_img.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(input_img)
            
            loss_l1 = criterion_l1(output, target)
            loss_ffl = criterion_ffl(output, target)
            
            loss = loss_l1 + 0.1 * loss_ffl # Balance spatial and frequency loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
            # Save periodic samples
            if i == 0 and epoch % 10 == 0:
                with torch.no_grad():
                    sample = torch.cat([input_img[0:1], output[0:1], target[0:1]], dim=0)
                    save_image(sample, f"samples/epoch_{epoch}.png")
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("Saved Best Model")
            
    print("Training Complete.")

if __name__ == "__main__":
    train()
