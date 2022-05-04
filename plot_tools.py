import random
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from utils import DataTransformer, Train_Transform

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def plotAcc_Loss(tr_acc_his, val_acc_his, tr_loss_his, val_loss_his, epochs, prefix_name, test_acc):
    plt.figure(figsize=[12, 7])
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), tr_acc_his, '-', linewidth=3, label='Train Acc')
    plt.plot(range(1, epochs + 1), val_acc_his, '-', linewidth=3, label='Val Acc')
    plt.ylabel('Acc(%)', fontsize=13)
    plt.title('Test Acc {}%'.format(94.43), fontsize=15)
    plt.grid(True), plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), tr_loss_his, '-', linewidth=3, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_loss_his, '-', linewidth=3, label='Val Loss')
    plt.xlabel('Epoch', fontsize=13), plt.ylabel('Loss', fontsize=13)
    plt.grid(True), plt.legend()
    ax = plt.gca()
    ax.set_yscale('log')
    pic_name = prefix_name + '_best.png'
    plt.savefig('./plots/' + pic_name)
    print('{} Saved !'.format(pic_name))


def plotConfuseMatrix(targets, predictions, prefix_name):
    cf_matrix = confusion_matrix(targets, predictions)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(7, 7))
    seaborn.heatmap(df_cm, annot=True, fmt="d", linewidths=.5, cmap=["#FDBFBF", '#74BDFF'], cbar=False)
    plt.savefig('./plots/' + prefix_name + '_confusion_matrix.png')
    print('{}_confusion_matrix.png Saved !'.format(prefix_name))


def plot_Img_Transformed(ori_data, trans_data, rng, name):
    fig = plt.figure(figsize=(12, 3))
    idx_ls = random.sample(range(0, 1000), rng)
    ori_images = [np.copy(ori_data.data)[idx] for idx in idx_ls]
    for i, idx in enumerate(idx_ls):
        plt.subplot(2, 5, i + 1)
        plt.imshow(ori_images[i])
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.3)
        plt.subplot(2, 5, i + 6)
        image, label = trans_data[idx]
        image = image.numpy()
        plt.imshow(np.transpose((image - np.min(image)) / (np.max(image) - np.min(image)), (1, 2, 0)))

        ax = plt.gca()
        ax.set_title(classes[label])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle(name)
    plt.savefig('./plots/' + name + ".png")


def plot_trans_process():
    def Train_Transform_1():
        transform = transforms.Compose([
            # GaussianNoise(mean=0, variance=1, amplitude=10),
            transforms.ToTensor(),
        ])
        return transform

    def Train_Transform_2():
        crop_size = 32
        crop_padding = 4
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(crop_size, padding=crop_padding),
        ])
        return transform

    def Train_Transform_3():
        flip_prob = 0.5
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.RandomVerticalFlip(p=flip_prob),
        ])
        return transform

    def Train_Transform_4():
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        return transform

    class Data_Transformer(Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def forward(self):
            image = self.dataset
            if self.transform:
                image = self.transform(image)
            return image

    image0 = np.copy(trainData.data)[4]  # 43, 4
    plt.subplot(1, 5, 1)
    plt.imshow(image0)
    ax = plt.gca()
    ax.set_title('Origin')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    title = ['Noise', 'Crop', 'Flip', 'Normalize']
    for idx in range(4):
        plt.subplot(1, 5, idx + 2)
        exec('Data{} = Data_Transformer(image{}, Train_Transform_{}())'.format(idx, idx, idx + 1))
        exec('image{} = np.transpose(np.array(Data{}.forward()), (1, 2, 0))'.format(idx + 1, idx))
        exec('plt.imshow(image{})'.format(idx + 1))
        ax = plt.gca()
        ax.set_title(title[idx])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


trainData = torchvision.datasets.CIFAR10('CIFAR10/', train=True, download=False, transform=None)
# plot_Img_Transformed(trainData, DataTransformer(trainData, Train_Transform()), 5, "Train")
# plot_trans_process()
