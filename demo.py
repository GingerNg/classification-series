import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
from data import train_dataloader, train_datasets, val_dataloader

trainset = train_datasets
trainloader = train_dataloader
testloader = val_dataloader

# 参数设置
num_epochs = 10
batch_size = 64
learning_rate = 0.001


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


# 参数设置
num_epochs = 15
batch_size = 64
learning_rate = 0.001
# 构建CNN模型

# in_channels (int): Number of channels in the input image
# out_channels (int): Number of channels produced by the convolution
# kernel_size (int or tuple): Size of the convolving kernel


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)  # 全连接层
        self.fc2 = nn.Linear(1024, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 图片显示


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# torchvision 数据集的输出是范围在[0,1]之间的 PILImage，我们将他们转换成归一化范围为[-1,1]之间的张量Tensors
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# # 获取CIFAR10训练集和测试集
# trainset = torchvision.datasets.CIFAR10(
#     root='data/', train=True, download=True, transform=transform)
# testset = torchvision.datasets.CIFAR10(
#     root='data/', train=False, download=True, transform=transform)
# # CIFAR10训练集和测试集装载
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=batch_size, shuffle=True, num_workers=0)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=batch_size, shuffle=False, num_workers=0)
# 图片类别
classes = ('plane', 'car', 'bird')
# 图片显示
images, labels = next(iter(trainloader))
imshow(torchvision.utils.make_grid(images))

# 定义损失函数和优化器
cnn_model = CNNNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9)

# train-训练模型
for epoch in range(num_epochs):
    running_loss = 0.00
    running_correct = 0.0
    print("Epoch  {}/{}".format(epoch, num_epochs))
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        running_correct += torch.sum(pred == labels.data)
    print("Loss is :{:.4f},Train Accuracy is:{:.4f}%".format(
        running_loss / len(trainset), 100 * running_correct / len(trainset)))

model_path = '/home/ginger/Projects/githubProjs/pytorch_classification/data/flower_3_32/cnn_model.pt'
# 保存训练好的模型
torch.save(cnn_model, model_path)

# test
# 加载训练好的模型
cnn_model = torch.load(model_path)
cnn_model.eval()
# 使用测试集对模型进行评估
correct = 0.0
total = 0.0
with torch.no_grad():   # 为了使下面的计算图不占用内存
    for data in testloader:
        images, labels = data
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("Test Average accuracy is:{:.4f}%".format(100 * correct / total))

# 求出每个类别的准确率
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        try:
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        except IndexError:
            continue
for i in range(len(classes)):
    print('Accuracy of %5s : %4f %%' %
          (classes[i], 100 * class_correct[i] / class_total[i]))
