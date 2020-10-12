import sys
sys.path.append("..")
import random
import cfg
import os
import glob





if __name__ == '__main__':
    traindata_path = cfg.BASE + 'train'
    labels = os.listdir(traindata_path)
    # print(labels)
    valdata_path = cfg.BASE + 'val'
    # 写train.txt文件
    txtpath = cfg.BASE
    # print(labels)
    for index, label in enumerate(labels):

        imglist = glob.glob(os.path.join(traindata_path, label, '*.png')) + \
            glob.glob(os.path.join(traindata_path, label, '*.jpg'))
        # print(imglist)
        with open(txtpath + 'train.txt', 'a')as f:
            for img in imglist:
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')

        imglist = glob.glob(os.path.join(valdata_path, label, '*.jpg')) + \
            glob.glob(os.path.join(valdata_path, label, '*.png'))
        with open(txtpath + 'val.txt', 'a')as f:
            for img in imglist:
                f.write(img + ' ' + str(index))
                f.write('\n')