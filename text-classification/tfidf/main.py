
from tfidf import file_utils
from tfidf.tfidf_classifer import TFIDFClassifier

tfidf_classifier = TFIDFClassifier()

# train
# 分词，第一个是分词输入，第二个参数是结果保存的路径
tfidf_classifier.segText("./data/data/", "./data/segResult/")
tfidf_classifier.bunch_merge_data_label("./data/segResult/", "./data/train_set.dat")  
stopWordList = file_utils.getStopWord("./data/stop/stopword.txt")  # 获取停用词
tfidf_classifier.getTFIDFMat("./data/train_set.dat", stopWordList, "./data/tfidfspace.dat")  # 输入词向量，输出特征空间


# test
tfidf_classifier.segText("./data/test1/", "./data/test_segResult/")  # 分词
tfidf_classifier.bunch_merge_data_label("./data/test_segResult/", "./data/test_set.dat")
tfidf_classifier.getTestSpace("./data/test_set.dat", "./data/tfidfspace.dat", stopWordList, "./data/testspace.dat")
tfidf_classifier.bayesAlgorithm("./data/tfidfspace.dat", "./data/testspace.dat")