import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import EfficientNetModel  # EfficientNet 기반
import torch.nn.functional as F
import os

def acc(pred, label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader, model):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        for images, labels in tqdm(data_loader):
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += acc(outputs, labels)
            total_samples += labels.size(0)

        scheduler.step()

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples

        print(f"Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc*100:.2f}%")

        # 모델 저장
        os.makedirs(args.save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{args.save_path}/model.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2025 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Directory to save model.pth")
    parser.add_argument('--data', default='data/', help="Path to training data directory")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    args.device = torch.device("cuda:1")

    print("===================================")
    print("Save path      :", args.save_path)
    print("Device         :", args.device)
    print("Batch size     :", args.batch_size)
    print("Learning rate  :", args.learning_rate)
    print("Epochs         :", args.epochs)
    print("===================================")

    train_loader = make_data_loader(args, mode='train')

    model = EfficientNetModel().to(args.device)

    train(args, train_loader, model)
