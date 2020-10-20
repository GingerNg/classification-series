import torch
import os
from data import train_dataloader, train_datasets, val_dataloader
import cfg
from utils import adjust_learning_rate_cosine, adjust_learning_rate_step
from evaluation_index import f1, acc
torch.multiprocessing.set_sharing_strategy('file_system')


testloader = val_dataloader
# testloader = train_dataloader
batch_size = cfg.BATCH_SIZE

cur_path = os.path.dirname(os.path.abspath(__file__))
# 加载训练好的模型
# model_path = cur_path + '/data/flower_50datasets/%s.pt' % cfg.model_name
# model = torch.load(model_path)
# model.eval()  # 不启用 BatchNormalization 和 Dropout

# model_path = cur_path + "/data/flower_50datasets/weights/resnet50/epoch_35.pth"
model_path = cur_path + "/data/flower_50datasets/weights/moblienetv2/epoch_20.pth"
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model
model = load_checkpoint(model_path)

# 模型放置在gpu上进行计算
if torch.cuda.is_available():
    model.cuda()

# 使用测试集对模型进行评估
correct = 0.0
total = 0.0
with torch.no_grad():   # 为了使下面的计算图不占用内存
    for data in testloader:
        images, labels = data
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("Test Average accuracy is:{:.4f}%".format(100 * correct / total))

# 求出每个类别的准确率
traindata_path = cfg.BASE + 'train'
classes = tuple(os.listdir(traindata_path))

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
Predicts = []
Labels = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
        outputs = model(images)
        if torch.cuda.is_available():
            outputs = outputs.cpu()
        _, predicted = torch.max(outputs, 1)
        # c = (predicted == labels).squeeze()  # 当不给定dim时，将输入张量形状中的1 去除并返回。
        # print(predicted)
        Predicts += predicted
        Labels += labels
        print(type(Predicts[0]))
print(len(Labels))
print(f1.micro_f1_index(y_true=Labels, y_pred=Predicts))
print(f1.macro_f1_index(y_true=Labels, y_pred=Predicts))
print(acc.acc_index(y_true=Labels, y_pred=Predicts))
        # try:
        #     for i in range(batch_size):
        #         label = labels[i]
        #         class_correct[label] += c[i].item()
        #         class_total[label] += 1
        # except IndexError:
        #     continue
# for i in range(len(classes)):
#     print('Accuracy of %5s : %4f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = cnn_model(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         try:
#             for i in range(batch_size):
#                 label = labels[i]
#                 class_correct[label] += c[i].item()
#                 class_total[label] += 1
#         except IndexError:
#             continue
# for i in range(len(classes)):
#     print('Accuracy of %5s : %4f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))