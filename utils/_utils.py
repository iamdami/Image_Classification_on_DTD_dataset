from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader

def make_data_loader(args, mode='train'):
    
    # mode: 'train' or 'test'
    if mode == 'train':
        custom_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])
        shuffle = True

    else:  # test or validation
        custom_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        shuffle = False

    dataset_ = datasets.ImageFolder(root=args.data, transform=custom_transforms)
    data_loader = DataLoader(dataset_, batch_size=args.batch_size, shuffle=shuffle)

    return data_loader
