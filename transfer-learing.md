1. 迁移学习背景
随着越来越多的机器学习应用场景的出现，监督学习需要大量的标注数据，标注数据是一项枯燥无味且花费巨大的任务，所以迁移学习受到越来越多的关注。
实际项目中没有那么多样本。
在实际的解决问题中很少有人从头至尾搭建一个完全新的模型。一般从Github上找和自己问题相似的场景，直接使用已有的网络结构，在上面修改，然后应用于解决实际问题。
Pytorch和Tf有很多内置的经典模型，在Imagnet，coco等数据集上训练好的模型参数，所以我们可以不仅复用这些网络结构，还会复用参数。
固定绝大多数参数，对最后边的几层进行更新，从而解决实际问题。
已有的数据集，对于实际的项目也有用，可以扩充实际项目的数据集。
2. 什么是迁移学习？
Ability of a system to recognize and apply knowledge and skills learned in previous domains/tasks to novel domains/tasks.
迁移学习是指将之前的已经学习过的知识应用于新的领域或者任务。

迁移学习方法
冻结骨干网络每层参数，依据实际情况，改变网络的输出，训练时，只训练最后分类网络模型参数。
通过设置requires_grad为False，从而不需要更新网络参数。

for param in res_net.parameters():
    param.requires_grad = False
1
2
迁移学习举例
不迁移学习
不加载与训练模型，网络参数全部需要重新训练

在下例中res_net = models.resnet18(pretrained=False)的pretrained设置为False，表示不加载预训练模型。
在下面的例子中，res_net.fc = nn.Linear(num_ftrs, 10) # only update this part parameters，表示替换最后一层全连接网络。
在下面的例子中，param.requires_grad = False，表示不需要对参数求导更新，只要保留原来的预训练的参数，这段代码被注释掉，表示需要对整个网络进行训练。
# coding: utf-8
from torchvision import models
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
import time

cifar_10 = datasets.CIFAR10('cifar',train=True,transform=transforms.Compose([
        transforms.Resize((32,32)),transforms.ToTensor()
    ]),download = True)
train_loader = torch.utils.data.DataLoader(cifar_10,
                                          batch_size=512,
                                          shuffle=True)

cifar_test = datasets.CIFAR10('cifar', train=False, transform=transforms.Compose([
    transforms.Resize((32, 32)), transforms.ToTensor()]), download=True)
cifar_test = DataLoader(cifar_test, batch_size=512, shuffle=True)

device = torch.device('cuda')

#pretrained 设置为True表示，获取模型在Imgnet上训练好的模型参数。
res_net = models.resnet18(pretrained=False)


plt.imshow(cifar_10[10][0].permute(1, 2, 0))

# for param in res_net.parameters():
#     param.requires_grad = False

# ResNet: CNN(with residual)-> CNN(with residual)-CNN(with residual)-Fully Connected

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = res_net.fc.in_features

res_net.fc = nn.Linear(num_ftrs, 10) # only update this part parameters 

criterion = nn.CrossEntropyLoss().to(device)

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(res_net.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochslosses = []

epochs = 10

for epoch in range(epochs):
    loss_train = 0
    res_net = res_net.to(device)
    t0 = time.time()
    # 将模型设置为训练模式
    res_net.train()
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = res_net(imgs)
        
        loss = criterion(outputs, labels)
        
        optimizer_conv.zero_grad()
        
        loss.backward() # -> only update fully connected layer
        
        optimizer_conv.step()
        
        loss_train += loss.item()
        
        # if i > 0 and i % 10 == 0:
        #     print('Epoch: {}, batch: {} -- loss: {}'.format(epoch, i,loss_train / i))
    t1 = time.time()
    # 模型评估模式
    res_net.eval()
    with torch.no_grad():  # 禁止梯度计算
        # test
        total_correct = 0
        total_num = 0
        for x, label in cifar_test:
            # x.shape [b, 3, 32, 32]
            # label.shape [b]
            x, label = x.to(device), label.to(device)
            # [b, 10]
            logits = res_net(x)
            # [b]
            pred = logits.argmax(dim=1)
            # [b] vs [b] => scalar tensor
            total_correct += torch.eq(pred, label).float().sum()
            total_num += x.size(0)

        acc = total_correct / total_num
        print("time = {} epoch:{} loss:{} acc:{}".format(t1-t0,epoch, loss.item(), acc.item()))

195
程序运行结果：

time = 17.179419994354248 epoch:0 loss:2.151430130004883 acc:0.21639999747276306
time = 16.146508932113647 epoch:1 loss:2.0868115425109863 acc:0.2583000063896179
time = 16.203269004821777 epoch:2 loss:1.9852572679519653 acc:0.27799999713897705
time = 16.231138706207275 epoch:3 loss:2.0423152446746826 acc:0.29109999537467957
time = 16.30972409248352 epoch:4 loss:1.9748584032058716 acc:0.3019999861717224
time = 16.317437410354614 epoch:5 loss:1.8853967189788818 acc:0.31119999289512634
...
...

迁移学习
加载与训练模型，网络参数不需要全部需要重新训练，在已有的模型基础上做fine tuning

在下例中res_net = models.resnet18(pretrained=True)的pretrained设置为True，表示加载预训练模型。
在下面的例子中，res_net.fc = nn.Linear(num_ftrs, 10) # only update this part parameters，表示替换最后一层全连接网络。
在下面的例子中，param.requires_grad = False，表示不需要对参数求导更新，保留原来的预训练的参数，网络训练时只训练这一新添加的全连接网络。
# coding: utf-8
from torchvision import models
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
import time

cifar_10 = datasets.CIFAR10('cifar',train=True,transform=transforms.Compose([
        transforms.Resize((32,32)),transforms.ToTensor()
    ]),download = True)
train_loader = torch.utils.data.DataLoader(cifar_10,
                                          batch_size=512,
                                          shuffle=True)

cifar_test = datasets.CIFAR10('cifar', train=False, transform=transforms.Compose([
    transforms.Resize((32, 32)), transforms.ToTensor()]), download=True)
cifar_test = DataLoader(cifar_test, batch_size=512, shuffle=True)

device = torch.device('cuda')

#pretrained 设置为True表示，获取模型在Imgnet上训练好的模型参数。
res_net = models.resnet18(pretrained=True)


plt.imshow(cifar_10[10][0].permute(1, 2, 0))

for param in res_net.parameters():
    param.requires_grad = False

# ResNet: CNN(with residual)-> CNN(with residual)-CNN(with residual)-Fully Connected

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = res_net.fc.in_features

res_net.fc = nn.Linear(num_ftrs, 10) # only update this part parameters 

criterion = nn.CrossEntropyLoss().to(device)

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(res_net.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochslosses = []

epochs = 10

for epoch in range(epochs):
    loss_train = 0
    res_net = res_net.to(device)
    t0 = time.time()
    # 将模型设置为训练模式
    res_net.train()
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = res_net(imgs)
        
        loss = criterion(outputs, labels)
        
        optimizer_conv.zero_grad()
        
        loss.backward() # -> only update fully connected layer
        
        optimizer_conv.step()
        
        loss_train += loss.item()
        
        # if i > 0 and i % 10 == 0:
        #     print('Epoch: {}, batch: {} -- loss: {}'.format(epoch, i,loss_train / i))
    t1 = time.time()
    # 模型评估模式
    res_net.eval()
    with torch.no_grad():  # 禁止梯度计算
        # test
        total_correct = 0
        total_num = 0
        for x, label in cifar_test:
            # x.shape [b, 3, 32, 32]
            # label.shape [b]
            x, label = x.to(device), label.to(device)
            # [b, 10]
            logits = res_net(x)
            # [b]
            pred = logits.argmax(dim=1)
            # [b] vs [b] => scalar tensor
            total_correct += torch.eq(pred, label).float().sum()
            total_num += x.size(0)

        acc = total_correct / total_num
        print("time = {} epoch:{} loss:{} acc:{}".format(t1-t0,epoch, loss.item(), acc.item()))

运行结果：

time = 8.121769666671753 epoch:0 loss:1.8075354099273682 acc:0.3285999894142151
time = 6.940755605697632 epoch:1 loss:1.7919408082962036 acc:0.3836999833583832
time = 6.933975458145142 epoch:2 loss:1.727358341217041 acc:0.4059000015258789
time = 7.0217390060424805 epoch:3 loss:1.6633989810943604 acc:0.4244000017642975
time = 7.1327598094940186 epoch:4 loss:1.5895496606826782 acc:0.43369999527931213
time = 6.878564357757568 epoch:5 loss:1.6604712009429932 acc:0.43619999289512634

# 从不迁移学习和迁移学习的结果对比来看，迁移学习能快速收敛，而且只需要训练最后一层全连接层，所以训练速度快。

# 迁移学习模型层数保留原则
# 假设有一个模型，100层（包括CNN层和FC层）
# 我们在一个数据集A做的训练，取得了还不错的结果（98%）
# 如果使用该训练之后的模型做迁移学习:

# 新问题的数据和A类似，新问题的数据类别小于A的数据类别
# 新的问题数据和A类似，新问题的数据类别大于A的数据类别
# 新问题的数据和A不类似
# 如果我么要保留原来训练的模型的层，那么1和2那个需要保留的层数多一些？
# 答案是1保留的层数多一些，2保留的层数少一些。3保留的层数应该更加少。

# 原文链接：https://blog.csdn.net/zhuguiqin1/article/details/121313507
20220420-15:56今天做志愿者发物资