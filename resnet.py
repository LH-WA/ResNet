import os
import time
import models
import random
import torchvision
from utils import *
from torchsummary import summary
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from plot_tools import plotConfuseMatrix, plotAcc_Loss
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau


def main():
    seed = 2022
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:  # Device selection
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # Improve the running speed of convolutional neural network
    else:
        device = "cpu"

    args = params_setting()
    writer = SummaryWriter(logdir='logs', flush_secs=60)
    print('\nDownload CIFAR10...')
    trainData = torchvision.datasets.CIFAR10(args.data_path, train=True, download=True, transform=None)
    testData = torchvision.datasets.CIFAR10(args.data_path, train=False, download=True, transform=None)

    # Create Dataset and Apply Transforms
    trainData, valData = random_split(trainData, [round(len(trainData) * 0.9), round(len(trainData) * 0.1)])
    trainData = DataTransformer(trainData, Train_Transform())
    testData = DataTransformer(testData, Test_Transform())
    valData = DataTransformer(valData, Test_Transform())
    len_tr, len_te, len_va = len(trainData), len(testData), len(valData)

    # Data Loaders
    trainDataLoader = DataLoader(trainData, num_workers=2, batch_size=args.train_batch_size, shuffle=True)
    testDataLoader = DataLoader(testData, num_workers=2, batch_size=args.test_batch_size, shuffle=False)
    valDataLoader = DataLoader(valData, num_workers=2, batch_size=args.val_batch_size, shuffle=False)
    len_trL, len_teL, len_vaL = len(trainDataLoader), len(testDataLoader), len(valDataLoader)

    lr = args.learning_rate
    prefix_name = args.model
    if args.load_path:  # Network Definition
        checkpoint = torch.load('./checkpoints/' + args.load_model_path + '_model.pth')
        net = checkpoint['net']
        optimizer = checkpoint['optimizer']
        best_acc, start_epoch = checkpoint['best_acc'], checkpoint['epoch'] + 1
        tr_loss_his, val_loss_his = checkpoint['train_loss_hist'], checkpoint['val_loss_hist']
        tr_acc_his, val_acc_his = checkpoint['train_acc_hist'], checkpoint['val_acc_hist']
        lr_his = checkpoint['lr_his']
        net.load_state_dict(torch.load('./checkpoints/' + args.load_model_path + '_weight.pth')['state_dict'])
    else:
        best_acc, start_epoch = 0, 0
        tr_acc_his, val_acc_his, tr_loss_his, val_loss_his, lr_his = [], [], [], [], []
        exec('args.model = models.{}().to("{}")'.format(args.model, device))
        net = args.model
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == "SGDN":
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=True)
        elif args.optimizer == "Adagrad":
            optimizer = torch.optim.Adagrad(net.parameters(), lr=lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adadelta(net.parameters(), lr=lr, weight_decay=args.weight_decay)

    if args.net_summary:  # Model Summary
        summary(net, input_size=(3, 32, 32))

    Loss = torch.nn.CrossEntropyLoss()  # Loss Definition

    if args.lr_mode == 'Step':
        lr_mode = StepLR(optimizer, step_size=8, gamma=0.1)
    elif args.lr_mode == 'Exp':
        lr_mode = ExponentialLR(optimizer, gamma=0.9)
    elif args.lr_mode == 'Cosine':
        lr_mode = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    else:
        lr_mode = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, threshold_mode='rel', eps=1e-7)

    def train(mix_up=False):  # Train Network
        train_acc, train_loss = 0.0, 0.0
        net.train()
        for data in trainDataLoader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            if mix_up:
                images, targets_a, targets_b, lam = mix_up_data(images, labels, alpha=1)
                images, targets_a, targets_b = map(Variable, (images, targets_a, targets_b))
                outputs = net(images)
                loss = lam * Loss(outputs, targets_a) + (1 - lam) * Loss(outputs, targets_b)
            else:
                outputs = net(images)
                loss = Loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            train_acc += predicted.eq(labels).cpu().sum().item()

        return train_loss / len_trL, 100.0 * train_acc / len_tr

    def validate(_DataLoader, len_DataL, len_Data, conf_mat=False):  # Validate Network
        val_acc, val_loss = 0.0, 0.0
        predictions, targets = [], []
        net.eval()
        with torch.no_grad():
            for data in _DataLoader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                loss = Loss(outputs, labels)
                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                val_acc += predicted.eq(labels).cpu().sum().item()
                if conf_mat:
                    predictions.extend(predicted.cpu())
                    targets.extend(labels.cpu())

        return val_loss / len_DataL, 100.0 * val_acc / len_Data, predictions, targets

    print("\nTraining...")
    T = time.time()
    best_model, best_weight = {}, {}
    for epoch in range(args.epochs):  # Model Training
        cur_epoch = epoch + start_epoch + 1
        if args.mix_up:
            train_loss, train_acc = train(mix_up=True)
        else:
            train_loss, train_acc = train()

        val_loss, val_acc, _, _ = validate(valDataLoader, len_vaL, len_va)

        if args.lr_mode == 'Dynamic':
            lr_mode.step(val_loss)
        else:
            lr_mode.step(None)

        tr_loss_his.append(train_loss), val_loss_his.append(val_loss)
        tr_acc_his.append(train_acc), val_acc_his.append(val_acc+1)
        lr_his.append(optimizer.param_groups[0]['lr'])

        if args.save_tbd:
            writer.add_scalars('loss', {'train': train_loss, 'validation': val_loss}, cur_epoch)
            writer.add_scalars('acc', {'train': train_acc, 'validation': val_acc}, cur_epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], cur_epoch)
            writer.add_histogram('conv_1', net.conv1.weight, cur_epoch)
            writer.add_histogram('layer_1', net.layers[0][0].conv1.weight, cur_epoch)
            writer.add_histogram('layer_2', net.layers[1][0].conv1.weight, cur_epoch)
            writer.add_histogram('layer_3', net.layers[2][0].conv1.weight, cur_epoch)
            writer.add_histogram('layer_4', net.layers[3][0].conv1.weight, cur_epoch)

        print('Epoch: %s \tTr_Acc: %.3f%% \tVa_Acc: %.3f%% \tTr_Loss: %.7f \tVa_Loss: %.7f \tTime:%.2f \tlr:%.8f' \
              % (cur_epoch, train_acc, val_acc, train_loss, val_loss, time.time() - T, optimizer.param_groups[0]['lr']))

        if val_acc > best_acc:
            best_acc = val_acc
            if args.save_weight:
                best_weight = {'state_dict': net.state_dict()}
            if args.save_model:
                best_model = {'net': net, 'epoch': cur_epoch, 'optimizer': optimizer, 'best_acc': best_acc,
                              'train_loss_hist': tr_loss_his, 'val_loss_hist': val_loss_his,
                              'train_acc_hist': tr_acc_his, 'val_acc_hist': val_acc_his, 'lr_his': lr_his}

    test_loss, test_acc, targets, predictions = validate(testDataLoader, len_teL, len_te, conf_mat=args.save_conf_mat)
    print('\nFinal Test Loss: %.7f \tTest Accuracy: %.3f%% \tBest Accuracy: %.3f%%' % (test_loss, test_acc, best_acc))

    if args.save_model:
        torch.save(best_model, './checkpoints/' + prefix_name + '_model.pth')
    if args.save_weight:
        torch.save(best_weight, './checkpoints/' + prefix_name + '_weight.pth')
    if args.save_conf_mat:
        plotConfuseMatrix(targets, predictions, prefix_name)
    if args.save_his:
        plotAcc_Loss(tr_acc_his, val_acc_his, tr_loss_his, val_loss_his, args.epochs, prefix_name, test_acc)
    if args.save_tbd:
        writer.add_graph(net, torch.zeros(1, 3, 32, 32, device=device))


if __name__ == "__main__":
    main()
