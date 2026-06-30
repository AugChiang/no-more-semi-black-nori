import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import torchvision.transforms as v2
from model import DualDomainNAFNet
from dataset import StripeAugmentor
import os
from torchvision import transforms
import cv2

def parse_args():
    parser = ArgumentParser(description="Image Restoration Inference")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to the trained model")
    parser.add_argument("--image_path", type=str, default="sample.webp", help="Path to the input image")
    parser.add_argument("--output_path", type=str, default="restored_output.png", help="Path to save the restored image")
    parser.add_argument("--add_stripes", action='store_true', help="Whether to add demo stripes to the input image")
    return parser.parse_args()

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
    # transform
    transform = v2.Compose([
        v2.RandomCrop(256),
        v2.ToTensor(),
    ])

    # To Tensor
    input_tensor = transform(Image.fromarray(img_np)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.clamp(output, 0, 1)
        
    # Back to PIL
    output_np = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(output_np)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path
    image_path = args.image_path
    output_path = args.output_path
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return

    model = DualDomainNAFNet(width=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # Restore with demo stripes
    restored_img = restore_image(model, image_path, device, add_stripes=args.add_stripes)
    restored_img.save(output_path)
    print(f"Saved restored output as {output_path}")

if __name__ == "__main__":
    main()
