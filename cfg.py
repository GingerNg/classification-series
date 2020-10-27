# -*- coding:utf-8 -*-
# @time :2019.09.07
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
from models import Resnet50, Resnet101, Resnext101_32x8d, Resnext101_32x16d, Densenet121, Densenet169, Mobilenetv2, \
    Efficientnet, Resnext101_32x32d, Resnext101_32x48d, Inception3, LeNet, AlexNet

import os
# home = os.path.expanduser('~')
home = os.path.dirname(os.path.abspath(__file__))
# 数据集的类别
NUM_CLASSES = 50

# 训练时batch的大小
BATCH_SIZE = 32

# 网络默认输入图像的大小
INPUT_SIZE = 128
# 训练最多的epoch
MAX_EPOCH = 50
# 使用gpu的数目
GPUS = 2
# 从第几个epoch开始resume训练，如果为0，从头开始
RESUME_EPOCH = 0

WEIGHT_DECAY = 5e-4  # L2正则化的目的就是为了让权重衰减到更小的值，在一定程度上减少模型过拟合的问题，所以权重衰减也叫L2正则化。
MOMENTUM = 0.9
# 初始学习率
LR = 1e-3


# 采用的模型名称
# model_name = 'resnext101_32x32d'

# model_name = "efficientnet-b7"

# model_name = 'lenet5'
# model_name = 'resnet50'
model_name = "alexnet"
# model_name = "moblienetv2"

MODEL_NAMES = {
    'resnext101_32x8d': Resnext101_32x8d,
    'resnext101_32x16d': Resnext101_32x16d,
    'resnext101_32x48d': Resnext101_32x48d,
    'resnext101_32x32d': Resnext101_32x32d,
    'resnet50': Resnet50,
    'resnet101': Resnet101,
    'densenet121': Densenet121,
    'densenet169': Densenet169,
    'moblienetv2': Mobilenetv2,
    'efficientnet-b7': Efficientnet,
    'efficientnet-b8': Efficientnet,
    "lenet5": LeNet,
    "alexnet": AlexNet

    # 'inception_3': Inception3
}


# BASE = home + '/data/flower_3/'
BASE = home + "/data/flower_50datasets/"


# 训练好模型的保存位置
SAVE_FOLDER = BASE + 'weights/'

# 数据集的存放位置
TRAIN_LABEL_DIR = BASE + 'train.txt'
VAL_LABEL_DIR = BASE + 'val.txt'
TEST_LABEL_DIR = BASE + 'test.txt'


# 训练完成，权重文件的保存路径,默认保存在trained_model下
TRAINED_MODEL = BASE + 'weights/resnext101_32x32d/epoch_40.pth'
