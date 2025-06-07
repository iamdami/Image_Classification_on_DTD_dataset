import argparse
import torch
from tqdm import tqdm
from model import EfficientNetModel
from utils._utils import make_data_loader

def acc(pred, label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def test(args, data_loader, model):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            correct += acc(outputs, labels)
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test EfficientNet on DTD')
    parser.add_argument('--load-model', default='checkpoints/model.pth', help="Path to trained model state_dict")
    parser.add_argument('--data', default='data/', help='Path to test dataset folder')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    args.device = torch.device("cuda:1")

    # Load data
    test_loader = make_data_loader(args, mode='test')

    # Load model
    model = EfficientNetModel()
    model.load_state_dict(torch.load(args.load_model, map_location=args.device))
    model.to(args.device)

    # Evaluate
    test(args, test_loader, model)
