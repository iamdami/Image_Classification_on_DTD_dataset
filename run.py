import argparse
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import EfficientNetModel
import re

def numerical_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

class TestImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def inference(args, data_loader, model):
    model.eval()
    preds = []

    with torch.no_grad():
        for inputs in tqdm(data_loader):
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            preds.extend(predicted.cpu().tolist())

    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EfficientNet Inference for DTD')
    parser.add_argument('--load-model', default='checkpoints/model.pth', help="Path to trained model")
    parser.add_argument('--data', default='data/', help='Path to folder containing test images (no subfolders)')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    args.device = torch.device("cuda:1")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load dataset
    test_dataset = TestImageDataset(args.data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = EfficientNetModel()
    model.load_state_dict(torch.load(args.load_model, map_location=args.device))
    model.to(args.device)

    # Run inference
    predictions = inference(args, test_loader, model)

    from collections import Counter
    print("🧮 Prediction Distribution:", Counter(predictions))

    # Save results
    with open("no_s_result.txt", "w") as f:
        f.writelines('\n'.join(map(str, predictions)))

    print("Inference complete. Saved to result.txt")
