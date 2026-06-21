import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MangaStripeDataset
from model import DualDomainNAFNet
from focal_frequency_loss import FocalFrequencyLoss
import os
from tqdm import tqdm
from torchvision.utils import save_image

def train():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 4
    epochs = 500
    lr = 1e-3
    image_path = "sample.webp"
    patch_size = 256
    
    # Dataset and Loader
    dataset = MangaStripeDataset(image_path, patch_size=patch_size, num_patches=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
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
