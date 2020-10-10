import sys
sys.path.append("..")
import random
import cfg
import os
import glob
from PIL import Image



def resize_img(filein, fileout, width, height, type="png"):
    img = Image.open(filein)
    # resize image with high-quality
    out = img.resize((width, height), Image.ANTIALIAS)
    out.save(fileout, type)


if __name__ == '__main__':
    traindata_path = cfg.BASE + 'train'
    labels = os.listdir(traindata_path)
    valdata_path = cfg.BASE + 'test'
    # 写train.txt文件
    txtpath = cfg.BASE
    # print(labels)
    for index, label in enumerate(labels):
        imglist = glob.glob(os.path.join(traindata_path, label, '*.png')) + \
            glob.glob(os.path.join(traindata_path, label, '*.jpg')) + glob.glob(os.path.join(valdata_path, '*.jpg'))
        # print(imglist)
        for img_path in imglist:
            filein = img_path
            fileout = img_path.replace("flower_3", "flower_3_32").replace("jpg", "png")
            file_dir = os.path.dirname(fileout)
            isExists = os.path.exists(file_dir)
            if not isExists:
                os.makedirs(file_dir)
            resize_img(img_path, fileout, width=32, height=32)
    #     random.shuffle(imglist)
    #     print(len(imglist))
    #     trainlist = imglist[:int(0.8*len(imglist))]
    #     vallist = imglist[(int(0.8*len(imglist))+1):]
    #     with open(txtpath + 'train.txt', 'a')as f:
    #         for img in trainlist:
    #             # print(img + ' ' + str(index))
    #             f.write(img + ' ' + str(index))
    #             f.write('\n')

    #     with open(txtpath + 'val.txt', 'a')as f:
    #         for img in vallist:
    #             # print(img + ' ' + str(index))
    #             f.write(img + ' ' + str(index))
    #             f.write('\n')

    # imglist = glob.glob(os.path.join(valdata_path, '*.jpg'))
    # with open(txtpath + 'test.txt', 'a')as f:
    #     for img in imglist:
    #         f.write(img)
    #         f.write('\n')
