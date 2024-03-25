import random
import torch
from d2l import torch as d2l


# 从零实现线性回归
# -------------------------------------
# 生成数据集
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # 生成一个均值为0，标准差为1，shape为(num_examples, len(w))的张量
    y = torch.matmul(X, w) + b  # 定义y = Xw + b
    y += torch.normal(0, 0.01, y.shape)  # 给y生成一些随机噪声，其中噪声服从均值为0，标准差为0.01 所以 y = Xw + b + 噪声
    return X, y.reshape((-1, 1))  # 将X和y返回，其中y以列向量返回


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabels:', labels[0])  # features size: torch.Size([1000, 2])
print('features size:', features.size(), '\nlabels size:', labels.size())  # labels size: torch.Size([1000, 1])

d2l.set_figsize()  # 定义figure大小
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)


# scatter绘制散点图，横坐标为features的第二个feature，纵坐标为y，散点的直径为1
# d2l.plt.show()

# -------------------------------------
# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 1000
    indices = list(range(num_examples))  # 0 - 1000的list
    random.shuffle(indices)  # 将indices随机打乱
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])  # 每次随机生成一个步长为batch_size的切片，比如生成indices[0,10],indices[10,20]
        yield features[batch_indices], labels[batch_indices]
        # 读取features[indices[0,10]]和对应的labels片段，因为上面已经将indices随机打乱了，所以是随机读取的一个片段，比如读取的是features[5],features[20],...等


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# -------------------------------------
# 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# -------------------------------------
# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# -------------------------------------
# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# -------------------------------------
# 定义优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():  # 禁用梯度计算，因为我们在手动更新模型参数，而不是通过PyTorch的自动微分机制
        for param in params:
            param -= lr * param.grad / batch_size  # 根据梯度和学习率来更新参数，为了避免learning rate和batch_size挂钩，所以这里将lr / batch_size
            param.grad.zero_()  # 将参数的梯度清零


# -------------------------------------
# 训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):  # 在每个epoch中
    for X, y in data_iter(batch_size, features, labels):  # 小批量读取样本
        l = loss(net(X, w, b), y)  # 计算loss函数，l的形状为(batch_size,1)，所以下一步计算梯度需要sum
        l.sum().backward()  # 计算loss函数的梯度
        sgd([w, b], lr, batch_size)  # 使用优化算法来更新w和b的值
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)  # 计算所有样本的损失函数
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')  # 输出所有样本的损失函数的平均值

# 对比真实参数和训练得到的参数的误差
print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')
