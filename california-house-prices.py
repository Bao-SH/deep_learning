# https://www.kaggle.com/c/california-house-prices/data
import os.path

from utils.download_from_kaggle import download_and_extract_data
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np

# 下载数据
competition_slug = "california-house-prices"
download_dir = ""
extract_dir = "data"
if os.path.exists(os.path.join(extract_dir, 'train.csv')) and os.path.exists(os.path.join(extract_dir, 'test.csv')):
    print('Files are already exist, skip downloading')
else:
    print('Start to download data...')
    download_and_extract_data(competition_slug, download_dir, extract_dir)

# 预处理
train_init_data = pd.read_csv('data/train.csv')
test_init_data = pd.read_csv('data/test.csv')

# 去除不合理的data，如State不是CA的record
train_init_data = train_init_data[train_init_data['State'] == 'CA']

# remove id 和 sold_price
train_raw_features = train_init_data.drop(columns=['Id', 'Sold Price'])
test_raw_features = test_init_data.drop(columns=['Id'])
print(f"train_raw_features.shape: {train_raw_features.shape}")
print(f"test_raw_features.shape: {test_raw_features.shape}")

all_raw_features = pd.concat((train_raw_features, test_raw_features))
print(f"all_raw_features.shape: {all_raw_features.shape}")

# 数值类型 进行标准化
numeric_features_idx = all_raw_features.dtypes[all_raw_features.dtypes != object].index
all_raw_features[numeric_features_idx] = all_raw_features[numeric_features_idx].apply(
    lambda x: ((x - x.mean()) / x.std())
)

# 处理缺失值
all_raw_features[numeric_features_idx] = all_raw_features[numeric_features_idx].fillna(0)


# 对文本信息，按照出现频率最多的20个词进行分类，然后其他的变成others


def get_sorted_object(df):
    print(df.columns)
    for column in df.columns:
        top_categories = df[column].value_counts().head(20).index.tolist()
        df.loc[:, column] = df[column].apply(
            lambda x: x if x in top_categories else 'others'
        )
    return df


object_features_idx = all_raw_features.dtypes[all_raw_features.dtypes == object].index
all_raw_features[object_features_idx] = get_sorted_object(all_raw_features[object_features_idx])
all_raw_features = pd.get_dummies(all_raw_features, dummy_na=True, dtype=int)
print(f"all_raw_features.shape: {all_raw_features.shape}")

# train_raw_features.to_csv('test.csv', index=False)
# bedroom 应该筛选bedroom关键词重新分类
# Address,Summary这种top10没有意义
train_features = torch.tensor(all_raw_features[:train_raw_features.shape[0]].values, dtype=torch.float32)
test_features = torch.tensor(all_raw_features[train_raw_features.shape[0]:].values, dtype=torch.float32)
train_labels = torch.tensor(train_init_data['Sold Price'].values.reshape(-1, 1), dtype=torch.float32)
print(f"train_features.shape: {train_features.shape}")
print(f"test_features.shape: {test_features.shape}")
print(f"train_labels.shape: {train_labels.shape}")


# 定义模型和损失函数
def get_net():
    return nn.Sequential(nn.Linear(in_features=train_features.shape[1], out_features=1))


loss = nn.MSELoss()


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


# 定义k折函数
def get_k_fold_data(k, i, X, y):  # 返回第i折的数据
    assert k > 1
    fold_size = X.shape[0] // k  # 确定每一折的size
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 找到第j折的index范围
        X_part, y_part = X[idx, :], y[idx]  # 第j折的数据
        if j == i:  # 如果输入的i == j
            X_valid, y_valid = X_part, y_part  # validation数据就是选中的这j折数据
        elif X_train is None:  # 如果是第一次进入循环
            X_train, y_train = X_part, y_part
        else:  # 其他j!=i的数据就是训练数据
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid  # 返回训练数据和验证数据


def train(net, train_features, train_labels, valid_features, valid_labels, num_epochs, lr, weight_decay, batch_size):
    train_ls, valid_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        train_ls.append(log_rmse(net, features=train_features, labels=train_labels))
        if valid_labels is not None:
            valid_ls.append(log_rmse(net, valid_features, valid_labels))
    return train_ls, valid_ls


# 返回训练误差和验证误差的平均值：
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


# 训练
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 256
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
d2l.plt.show()

print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')


#
# train_ls = train(num_epochs, batch_size, get_net(), loss, lr)
# d2l.plot(list(range(1, num_epochs + 1)), train_ls,
#          xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
#          legend=['train'], yscale='log')
# d2l.plt.show()

# 使用测试集
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_init_data,
               num_epochs, lr, weight_decay, batch_size)
