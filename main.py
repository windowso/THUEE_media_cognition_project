import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import TinyCaltech35
from resnet import resnet
from visualize import Visualize


def main(config):
    transform_train, transform_test = data_enhance(config.image_size)

    train_dataset = TinyCaltech35(transform=transform_train, used_data=['train', 'val'])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    test_dataset = TinyCaltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    model = resnet(class_num=config.class_num)

    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum,
                          weight_decay=config.weight_decay,
                          # nesterov=True
                          )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, 0.2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer, mode='max', factor=0.75, patience=5, verbose=True)
    creiteron = torch.nn.CrossEntropyLoss()

    # you may need train_numbers and train_losses to visualize something
    train_numbers, train_losses, accuracies = train(config, train_loader, model, optimizer, scheduler, creiteron,
                                                    test_loader)

    # you can use validation dataset to adjust hyper-parameters
    test_accuracy, features, labels = test(test_loader, model)
    print('===========================')
    print("test accuracy:{}%".format(test_accuracy * 100))
    visualize = Visualize(train_numbers, train_losses, accuracies, features, labels)
    visualize.loss_accuracy_visualize()
    visualize.features_visualize()


def train(config, data_loader, model, optimizer, scheduler, creiteron, test_loader):
    model.train()
    model.cuda()
    train_losses = []
    train_numbers = []
    accuracies = []
    counter = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.cuda(), label.cuda()
            output = model(data)
            loss = creiteron(output, label).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter += data.shape[0]
            accuracy = (label == output.argmax(dim=1)).sum() * 1.0 / output.shape[0]
            if batch_idx % 50 == 0:
                print('Train Epoch: {} / {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.6f}'.format(
                    epoch, config.epochs, batch_idx * len(data), len(data_loader.dataset),
                                          100. * batch_idx / len(data_loader), loss.item(), accuracy.item()))
                train_losses.append(loss.item())
                train_numbers.append(counter)
                accuracies.append(accuracy.item())
        scheduler.step()
        torch.save(model.state_dict(), './model.pth')
    return train_numbers, train_losses, accuracies


def test(data_loader, model):
    model.eval()
    model.cuda()
    features = []
    labels = []
    correct = 0
    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            pred = output.argmax(dim=1)
            features.append(output.cpu().numpy())
            labels.extend(label.cpu().numpy())
            correct += (pred == label).sum()
    accuracy = correct * 1.0 / len(data_loader.dataset)
    features = np.vstack(features)
    return accuracy, features, np.array(labels)


def data_enhance(image_size):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.64, 1.0),
                                     ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])

    return transform_train, transform_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[48, 48])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_num', type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--milestones', type=int, nargs='+', default=[20, 25])

    config = parser.parse_args()
    main(config)
