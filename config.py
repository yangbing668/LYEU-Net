from torchvision import transforms


trainLoaderConfig = {
    'transforms': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.15),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.ToTensor(),
    ]),
    'batch_size': 8,
    'shuffle': True,
    'num_workers': 2
}

valLoaderConfig = {
    'transforms': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    'batch_size': 4,
    'shuffle': False,
    'num_workers': 2
}