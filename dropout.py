import torch
from torch import nn
from d2l import torch as d2l
from softmax_regression_scratch import train_ch3

dropout1, dropout2 = 0.2, 0.3
lr = 0.1
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout1),  # 对隐藏层进行dropout
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout2),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=256)
loss = nn.CrossEntropyLoss(reduction='none')
num_epochs = 10
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()
