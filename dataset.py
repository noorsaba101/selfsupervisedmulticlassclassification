from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class CIFAR10Pair:
    def __init__(self, train, aug_mode):
        self.dataset = CIFAR10(root='.', train=train, download=True)
        self.aug_mode = aug_mode
        self.augmentation = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2)])

        self.to_tensor = ToTensor()

    def __getitem__(self, item):
        x = self.dataset.data[item]
        label = self.dataset.targets[item]

        if self.aug_mode:
            x1 = self.augmentation(self.to_tensor(x))
            x2 = self.augmentation(self.to_tensor(x))

            return x1, x2, label
        else:
            x = self.to_tensor(x)
            return x, label

    def __len__(self):
        return len(self.dataset.data)