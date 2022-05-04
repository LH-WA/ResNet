import os
import cv2
import torch
import numpy as np
from argparse import ArgumentParser as Args
from torchvision import transforms
from torch.utils.data import Dataset
# from PIL import Image


def params_setting():
    parser = Args()
    parser.add_argument("-dp", "--data_path", type=str, default="CIFAR10/",
                        metavar="<path>", help="Dataset storage path | default: CIFAR10/")
    parser.add_argument("-trb", "--train_batch_size", type=int, default=128,
                        metavar="<int>", help="Batch size for training | default: 128")
    parser.add_argument("-vab", "--val_batch_size", type=int, default=128,
                        metavar="<int>", help="Batch size for validation | default: 128")
    parser.add_argument("-teb", "--test_batch_size", type=int, default=128,
                        metavar="<int>", help="Batch size for testing | default: 128")
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        metavar="<int>", help="Number of epochs for training | default: 1")
    parser.add_argument("-m", "--model", type=str, default="ResNet24",
                        choices=['ResNet10', 'ResNet16', 'ResNet24', 'ResNet48'],
                        metavar="<str>", help="ResNets Model | default: ResNet24")
    parser.add_argument("-o", "--optimizer", type=str, default='SGD',
                        choices=['SGD', 'SGDN', 'Adagrad', 'Adadelta'],
                        metavar="<str>", help="Optimizer : SGD, SGDN, Adagrad, Adadelta | default: SGD")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1,
                        metavar="<float>", help="Learning Rate of optimizer | default: 0.1")
    parser.add_argument("-lrm", "--lr_mode", type=str, default="Exp",
                        choices=['Step', 'Exp', 'Cosine', 'Dynamic'],
                        help="Switch Learning Mode | default: Exp")
    parser.add_argument("-lp", "--load_path", type=str, default=None,
                        metavar="<path>", help="Path to Model weight and params | default: None")
    parser.add_argument("-mo", "--momentum", type=float, default=0.9,
                        metavar="<float>", help="Momentum of optimizer | default: 0.9")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-3,
                        metavar="<float>", help="Weight Decay of optimizer | default: 1e-3")
    parser.add_argument("-act", "--activation", type=str, default='ReLU',
                        choices=['ReLU', 'ReLU6', 'LRU', 'ELU', 'Swish'],
                        metavar="<str>", help="Activations: ReLU, ReLU6, LRU, ELU, Swish| default: ReLU")
    parser.add_argument("-coe", "--activation_coe", type=float, default=0.1,
                        help="Activation coefficient | default: 0.1")
    parser.add_argument("-mix", "--mix_up", action="store_true",
                        help="Enable Mix up  | default: false")
    parser.add_argument("-ns", "--net_summary", action="store_true",
                        help="Print Network Summary | default: false")
    parser.add_argument("-sm", "--save_model", action="store_true",
                        help="Save Params of Best Model | default: false")
    parser.add_argument("-sw", "--save_weight", action="store_true",
                        help="Save Weight of Best Model | default: false")
    parser.add_argument("-st", "--save_tbd", action="store_true",
                        help="Save plots and net graph in Tensorboard | default: false")
    parser.add_argument("-shis", "--save_his", action="store_true",
                        help="Save Loss, Acc and Lr history | default: false")
    parser.add_argument("-scm", "--save_conf_mat", action="store_true",
                        help="Save confusion matrix | default: false")

    return parser.parse_args()


args = params_setting()
if (args.save_model or args.save_weight) and not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
if (args.save_his or args.save_conf_mat) and not os.path.isdir('plots'):
    os.mkdir('plots')
if args.save_tensorboard and not os.path.isdir('logs'):
    os.mkdir('logs')
print("\nRun Config: \n------------------------------")
for key in args.__dict__:
    print("{:<18} : {}".format(key, args.__dict__[key]))


def Test_Transform():
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transform


def Train_Transform():
    crop_size = 32
    crop_padding = 4
    flip_prob = 0.5
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        # GaussianNoise(mean=0.0, variance=1.0, amplitude=1.0),
        transforms.RandomCrop(crop_size, padding=crop_padding),
        transforms.RandomHorizontalFlip(p=flip_prob),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transform


# class GaussianNoise(object):
#
#     def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
#         self.mean = mean
#         self.variance = variance
#         self.amplitude = amplitude
#
#     def __call__(self, img):
#         img = np.array(img)
#         h, w, c = img.shape
#         N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
#         N = np.repeat(N, c, axis=2)
#         img = N + img
#         img[img > 255] = 255
#         img = Image.fromarray(img.astype('uint8')).convert('RGB')
#         return img


class DataTransformer(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)


def mix_up_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    index = torch.randperm(x.size()[0]).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def l1_regularization(model, alpha):
    loss = torch.tensor(0, dtype=torch.float32).cuda()
    for name, param in model.named_parameters():
        if 'weight' in name:
            loss += torch.norm(param, 1)
    return alpha * loss


class Activations_gradients:
    def __init__(self, model, target_layers):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(target_layer.register_full_backward_hook(self.save_gradient))
            else:
                self.handles.append(target_layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations.append(output.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = [grad_output[0].cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self, model, target_layers):
        self.model = model.eval()
        self.target_layers = target_layers
        self.activations_grads = Activations_gradients(self.model, target_layers)

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)
        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy() for a in self.activations_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # ReLU
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):
        output = self.activations_grads(input_tensor)

        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"Predict Category Id: {target_category}")

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_grads.release()
        if isinstance(exc_value, IndexError):
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap * 0.5 + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
