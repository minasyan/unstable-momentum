import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms

# torch.manual_seed(1)    # reproducible

''' TODO:
* Make 8 layers (based on Alex Net)
* Plots '''

'''Constants'''
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

'''Loading MNIST'''
train_loader = torch.utils.data.DataLoader(
    dataset = dset.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE,
    shuffle=True,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    dataset = dset.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE,
    shuffle=True,
    **kwargs)


'''Fake dataset'''
# fake dataset
# x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
# y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))
# plot dataset
# plt.scatter(x.numpy(), y.numpy())
# plt.show()
# put dateset into torch dataset
# torch_dataset = Data.TensorDataset(x, y)
# loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)


# default network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(28*28, 256)   # hidden layer
        self.predict = torch.nn.Linear(256, 10)   # output layer
        self.l = torch.nn.Linear(256,10)

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

if __name__ == '__main__':
    # different nets
    net_SGD         = Net()
    # net_Momentum    = Net()
    # net_RMSprop     = Net()
    # net_Adam        = Net()
    nets = [net_SGD]

    # different optimizers
    opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    # opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    # opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    # opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD]

    loss_func = torch.nn.MSELoss()
    losses_his = [[]]   # record loss

    # training
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        for step, (x, target) in enumerate(train_loader):          # for each training step
            b_x = Variable(x)
            b_y = Variable(target)

            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)              # get output for every net
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()                # clear gradients for next train
                loss.backward()                # backpropagation, compute gradients
                opt.step()                     # apply gradients
                l_his.append(loss.data[0])     # loss recoder
                w = list(net.parameters())
                print(w)

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()
