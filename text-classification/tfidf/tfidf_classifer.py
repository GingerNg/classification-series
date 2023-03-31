import os
import pickle

import jieba
from tfidf import file_utils

from sklearn.datasets.base import Bunch

from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类

from sklearn.naive_bayes import MultinomialNB  # 多项式贝叶斯算法


class TFIDFClassifier(object):
    def __init__(self):
        pass

    def segText(self,inputPath, resultPath):
        fatherLists = os.listdir(inputPath)  # 主目录
        for eachDir in fatherLists:  # 遍历主目录中各个文件夹
            eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
            each_resultPath = resultPath + eachDir + "/"  # 分词结果文件存入的目录
            if not os.path.exists(each_resultPath):
                os.makedirs(each_resultPath)
            childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
            for eachFile in childLists:  # 遍历每个文件夹中的子文件
                eachPathFile = eachPath + eachFile  # 获得每个文件路径
                #  print(eachFile)
                content = file_utils.readFile(eachPathFile)  # 调用上面函数读取内容
                # content = str(content)
                result = (str(content)).replace("\r\n", "").strip()  # 删除多余空行与空格
                # result = content.replace("\r\n","").strip()

                cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
                words = " ".join(cutResult)
                file_utils.saveFile(each_resultPath + eachFile, words)  # 调用上面函数保存文件


    def bunch_merge_data_label(self, inputFile, outputFile):
        """
        文本和标签组成字典后保存为文件
        :param inputFile:
        :param outputFile:
        :return:
        """
        catelist = os.listdir(inputFile)
        bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
        bunch.target_name.extend(catelist)  # 将类别保存到Bunch对象中
        for eachDir in catelist:
            eachPath = inputFile + eachDir + "/"
            fileList = os.listdir(eachPath)
            for eachFile in fileList:  # 二级目录中的每个子文件
                fullName = eachPath + eachFile  # 二级目录子文件全路径
                bunch.label.append(eachDir)  # 当前分类标签
                bunch.filenames.append(fullName)  # 保存当前文件的路径
                bunch.contents.append(file_utils.readFile2(fullName).strip())  # 保存文件词向量
        with open(outputFile, 'wb') as file_obj:  # 持久化必须用二进制访问模式打开
            pickle.dump(bunch, file_obj)
            # pickle.dump(obj, file, [,protocol])函数的功能：将obj对象序列化存入已经打开的file中。
            # obj：想要序列化的obj对象。
            # file:文件名称。
            # protocol：序列化使用的协议。如果该项省略，则默认为0。如果为负值或HIGHEST_PROTOCOL，则使用最高的协议版本

    def getTFIDFMat(self, inputPath, stopWordList, outputPath):  # 求得TF-IDF向量
        """
        文档向量化
        :param inputPath:
        :param stopWordList:
        :param outputPath:
        :return:
        """
        bunch = file_utils.readBunch(inputPath)
        tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                           vocabulary={})
        # 初始化向量空间
        vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5)
        transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
        # 文本转化为词频矩阵，单独保存字典文件
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_  # 获取词汇
        file_utils.writeBunch(outputPath, tfidfspace)

    def getTestSpace(self,testSetPath, trainSpacePath, stopWordList, testSpacePath):
        bunch = file_utils.readBunch(testSetPath)
        # 构建测试集TF-IDF向量空间
        testSpace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                          vocabulary={})
        # 导入训练集的词袋
        trainbunch = file_utils.readBunch(trainSpacePath)
        # 使用TfidfVectorizer初始化 向量空间模型  使用训练集词袋向量
        vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        transformer = TfidfTransformer()
        testSpace.tdm = vectorizer.fit_transform(bunch.contents)
        testSpace.vocabulary = trainbunch.vocabulary
        # 持久化
        file_utils.writeBunch(testSpacePath, testSpace)

    def feature_select(self):
        pass

    def bayesAlgorithm(self, trainPath, testPath):
        trainSet = file_utils.readBunch(trainPath)
        testSet = file_utils.readBunch(testPath)
        clf = MultinomialNB(alpha=0.001).fit(trainSet.tdm, trainSet.label)
        # alpha:0.001 alpha 越小，迭代次数越多，精度越高
        # print(shape(trainSet.tdm))  #输出单词矩阵的类型
        # print(shape(testSet.tdm))
        predicted = clf.predict(testSet.tdm)
        total = len(predicted)
        rate = 0
        for flabel, fileName, expct_cate in zip(testSet.label, testSet.filenames, predicted):
            if flabel != expct_cate:
                rate += 1
                print(fileName, ":实际类别：", flabel, "-->预测类别：", expct_cate)
        print("error_num:%s and total_num:%s" %(rate,total),)
        print("error rate:", float(rate) * 100 / float(total), "%")


    def train(self,trainPath):
        pass

    def test(self,):
        pass


