from model import EfficientNetModel
from utils._utils import make_data_loader
from train import train
import torch

class Args:
    save_path = 'checkpoints'
    data = 'data'  # 폴더 구조: data/class_name/image.jpg
    epochs = 30
    learning_rate = 1e-3
    batch_size = 32
    device = torch.device("cuda:1")

args = Args()

# 데이터 로더 및 모델 생성
train_loader = make_data_loader(args, mode='train')
model = EfficientNetModel().to(args.device)

# 학습 실행
train(args, train_loader, model)
