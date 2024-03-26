import torch
from IPython import display
from d2l import torch as d2l
from d2l.torch import Accumulator

# 初始化模型参数
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784  # 因为每个样本是28*28的图像，每个像素当成一个feature，一共有28*28=784个feature
num_outputs = 10  # 输出是10个类别

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)  # 权重应该是784*10
b = torch.zeros(num_outputs, requires_grad=True)  # 偏移量应该和输出y一致，为10个


# 定义softmax操作
def softmax(X):
    X_exp = torch.exp(X)  # 对每一项求幂
    partition = X_exp.sum(1, keepdim=True)  # 对每一行相加，得到规范化常数 （因为一个样本就是一行，要保证每一行的概率相加为1）
    return X_exp / partition  # 将每一行除以其规范化常数


# 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)  # 假设有n个样本，则X为n*784，W为784*10，返回的y应该是n*10


# 定义损失函数
# 交叉熵损失函数：真实标签的预测概率的负对数似然
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


# 分类精度
# 正确预测数量与总预测数量之比
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 如果y_hat是矩阵，那么假定第二个维度存储每个类的预测分数
        y_hat = y_hat.argmax(axis=1)  # 使用argmax获得每行中最大元素的索引来获得预测类别
    # 将预测类别与真实y元素进行比较。
    # 由于等式运算符“==”对数据类型很敏感，因此我们将y_hat的数据类型转换为与y的数据类型一致。结果是一个包含0（错）和1（对）的张量
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# 评估在任意模型net的精度
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 训练
def train_epoch_ch3(net, train_iter, loss, updater):  # 定义一个通用的训练函数，来训练一个迭代周期，输入包含模型，dataset，loss和优化算法
    if isinstance(net, torch.nn.Module):  # 将模型设置为训练模式
        net.train()
    metric = Accumulator(3)  # 记录metric
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())  # 训练损失总和、训练准确度总和、样本数
    return metric[0] / metric[2], metric[1] / metric[2]  # 返回训练损失和训练精度


# 定义一个在动画中绘制数据的实用程序类
class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()


# 定义运行多个周期的训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


lr = 0.1


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


# 执行计划
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
d2l.plt.show()
