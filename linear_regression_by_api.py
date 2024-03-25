import torch
from torch.utils import data
from d2l import torch as d2l
# 通过调用已有库的API来实现线性回归
# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)  # 直接调用d2l的包来生成数据


# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    # 创建了一个PyTorch数据集对象，使用TensorDataset类。
    # 这个类用于将输入数据和标签数据打包成一个数据集对象。
    # *data_arrays 表示将 data_arrays 中的每个元素解包传递给 TensorDataset 构造函数，通常 data_arrays 包含输入数据和对应的标签数据。
    dataset = data.TensorDataset(*data_arrays)
    # 将数据集分成小批次进行加载，创建了一个数据加载器
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))  # 通过next从迭代器中获取第一项

# 定义模型
from torch import nn  # nn是神经网络的缩写 neural network
# 定义神经网络的层，Sequential类将多个层串联在一起，这里我们只有一层，Linear层。
# Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推
# Linear输入的参数分别为：输入特征形状(X的形状）,输出特征形状(y的形状）
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)  # 查看Linear，里面包含两个模型参数，分别叫做weight和bias
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()  # 均方误差

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # 通过调用net(X)生成预测y_hat，并结合y计算损失
        trainer.zero_grad()  # 清空参数的梯度
        l.backward()  # 计算梯度
        trainer.step()  # 调用优化器来更新参数
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 比较真实参数和模型参数
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

