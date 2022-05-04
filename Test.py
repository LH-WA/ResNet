import torch
from models import ResNet24
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

val_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
valid_dataset = CIFAR10(root='CIFAR10', train=False, download=False, transform=val_transform)
valDataLoader = DataLoader(valid_dataset, num_workers=2, batch_size=128, shuffle=False)

net = ResNet24().cuda()
net.load_state_dict(torch.load('checkpoints/ResNet24_weight.pth')['state_dict'])


def main():
    net.eval()
    val_acc = 0.0
    with torch.no_grad():
        for data in valDataLoader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')
            predicted = torch.argmax(net(images), dim=1)
            val_acc += predicted.eq(labels).cpu().sum().item()

    print('Accuracy in test dataset: ', 100 * val_acc / len(valid_dataset))


if __name__ == '__main__':
    main()
