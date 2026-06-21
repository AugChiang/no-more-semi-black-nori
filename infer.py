import torch
import numpy as np
from PIL import Image
from model import DualDomainNAFNet
from dataset import StripeAugmentor
import os
from torchvision import transforms
import cv2

def restore_image(model, image_path, device, add_stripes=False):
    model.eval()
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    
    if add_stripes:
        augmentor = StripeAugmentor()
        img_np = augmentor.add_stripes(img_np)
        Image.fromarray(img_np).save("stained_input.png")
        print("Saved stained input as stained_input.png")

    # To Tensor
    input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.clamp(output, 0, 1)
        
    # Back to PIL
    output_np = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(output_np)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "checkpoints/best_model.pth"
    image_path = "sample.webp"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return

    model = DualDomainNAFNet(width=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # Restore with demo stripes
    restored_img = restore_image(model, image_path, device, add_stripes=True)
    restored_img.save("restored_output.png")
    print("Saved restored output as restored_output.png")

if __name__ == "__main__":
    main()
