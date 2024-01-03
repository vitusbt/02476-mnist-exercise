import torch
from torch.utils.data import TensorDataset, DataLoader


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784)
    train_images_0 = torch.load('data/corruptmnist/train_images_0.pt')
    train_images_1 = torch.load('data/corruptmnist/train_images_1.pt')
    train_images_2 = torch.load('data/corruptmnist/train_images_2.pt')
    train_images_3 = torch.load('data/corruptmnist/train_images_3.pt')
    train_images_4 = torch.load('data/corruptmnist/train_images_4.pt')
    train_images_5 = torch.load('data/corruptmnist/train_images_5.pt')
    train_target_0 = torch.load('data/corruptmnist/train_target_0.pt')
    train_target_1 = torch.load('data/corruptmnist/train_target_1.pt')
    train_target_2 = torch.load('data/corruptmnist/train_target_2.pt')
    train_target_3 = torch.load('data/corruptmnist/train_target_3.pt')
    train_target_4 = torch.load('data/corruptmnist/train_target_4.pt')
    train_target_5 = torch.load('data/corruptmnist/train_target_5.pt')

    test_images = torch.load('data/corruptmnist/test_images.pt').unsqueeze(1)
    test_target = torch.load('data/corruptmnist/test_target.pt')

    train_images = torch.concat([
        train_images_0,
        train_images_1,
        train_images_2,
        train_images_3,
        train_images_4,
        train_images_5
    ]).unsqueeze(1)

    train_images = train_images*2 - 1
    test_images = test_images*2 - 1

    train_target = torch.concat([
        train_target_0,
        train_target_1,
        train_target_2,
        train_target_3,
        train_target_4,
        train_target_5
    ])

    train_dataset = TensorDataset(train_images, train_target)
    test_dataset = TensorDataset(test_images, test_target)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return train_loader, test_loader
