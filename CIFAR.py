from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2) 
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2) 
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch, index=None):
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx == index:
            continue
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_acc = 100. * correct / (len(train_loader.dataset) - 64)
    return train_acc, get_params(model)

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def get_params(model):
    curr_parameters = []
    for i in model.parameters():
        curr_parameters.append(i.view(i.numel()))
    curr_parameters = torch.cat(curr_parameters, 0)
    curr_parameters = torch.div(curr_parameters, torch.norm(curr_parameters))
    return curr_parameters

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 13, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    index_1 = np.random.randint(0, len(train_loader))
    index_2 = np.random.randint(0, len(train_loader))
    while index_2 == index_1:
        index_2 = np.random.randint(0, len(train_loader))



    ## Start up first run.
    model1 = Net1().to(device)
    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum)
    torch.save(model1.state_dict(), 'init_params.pt')
    ## Start up second run.
    model2 = Net2().to(device);
    optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum)
    model2.load_state_dict(torch.load('init_params.pt'))

    params_dist = []
    gen_errors = []
    params_dist.append(torch.dist(get_params(model1), get_params(model2)).cpu())
    for epoch in range(1, args.epochs + 1):
        train_acc, params_1 = train(args, model1, device, train_loader, optimizer1, epoch, index=index_1)
        no_use, params_2 = train(args, model2, device, train_loader, optimizer2, epoch, index=index_2)

        test_acc = test(args, model1, device, test_loader)
        test(args, model2, device, test_loader)

        test_loss = 100 - test_acc
        train_loss = 100 - train_acc

        params_dist.append(torch.dist(params_1, params_2).cpu())
        gen_errors.append(np.absolute(test_loss - train_loss))

    x = [epoch for epoch in range(1, args.epochs + 1)]

    gen_errors = torch.Tensor(gen_errors)
    print (gen_errors)
    print (params_dist)
    torch.save(params_dist, 'CIFAR_final_result_dist_{}msgd_{}epochs.pt'.format(args.momentum, args.epochs))
    torch.save(gen_errors, 'CIFAR_final_result_error_{}msgd_{}epochs.pt'.format(args.momentum, args.epochs))
    # plt.plot(x, params_dist)
    # plt.show()
if __name__ == '__main__':
    main()
