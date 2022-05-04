import torch
import numpy as np
from torch import load
from models import ResNet24
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10
from utils import GradCAM, show_cam_on_image

ori_transform = transforms.Compose([transforms.ToTensor()])
ori_dataset = CIFAR10(root='CIFAR10', train=False, download=False, transform=ori_transform)
val_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
valid_dataset = CIFAR10(root='CIFAR10', train=False, download=False, transform=val_transform)

num = 331
pic = ori_dataset[num][0]
img = np.transpose(pic.numpy(), (1, 2, 0))
plt.subplot(1, 5, 1)
plt.imshow(img)
ax = plt.gca()
ax.set_title('Origin')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

model = ResNet24().cuda()
model.load_state_dict(load('checkpoints/ResNet24_weight.pth')['state_dict'])
for i in range(4):
    cam = GradCAM(model=model, target_layers=[model.layers[i]])
    grayscale_cam = cam(input_tensor=torch.unsqueeze(valid_dataset[num][0], dim=0).cuda(), target_category=None)[0, :]
    plt.subplot(1, 5, i + 2)
    ax = plt.gca()
    ax.set_title(r'$Layer_{}$'.format(i + 1))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(show_cam_on_image(img.astype(dtype=np.float32) / np.max(img), grayscale_cam, use_rgb=True))
print('Authentic Category Id: ', ori_dataset[num][1])
plt.savefig('Grad_Cam.png')
