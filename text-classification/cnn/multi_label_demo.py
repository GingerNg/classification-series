
"""
Mulan: A Java Library for Multi-Label Learning + DataSets
http://mulan.sourceforge.net/datasets-mlc.html

https://blog.csdn.net/qq_38906523/article/details/80210527
"""
from cnn.models import MultiLabelModel

if __name__ == '__main__':
    import scipy
    from scipy.io import arff
    import pandas as pd

    # data, meta = scipy.io.arff.loadarff('/Users/shubhamjain/Documents/yeast/yeast-train.arff')
    # df = pd.DataFrame(data)

    from sklearn.datasets import make_multilabel_classification

    # this will generate a random multi-label dataset
    """
    sparse（稀疏）:如果是True，返回一个稀疏矩阵，稀疏矩阵表示一个有大量零元素的矩阵。
    n_labels:每个实例的标签的平均数量。
    return_indicator:“sparse”在稀疏的二进制指示器格式中返回Y。
    allow_unlabeled:如果是True，有些实例可能不属于任何类。
    """
    X, y = make_multilabel_classification(n_samples=1100,
                                          n_features=20,
                                          n_classes=10,
                                          sparse=False,
                                          n_labels=20,
                                          # return_indicator='sparse',
                                          allow_unlabeled=False)

    n_train = 1000
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    print(trainX.shape, testX.shape)

    model = MultiLabelModel(feature_dim=20,label_dim=10)

    history = model.fit(trainX, trainy, epochs=50, verbose=0)
    print(history.history.keys())
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc', 'loss'], loc='upper left')
    fig.savefig('performance.png')
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='lower left')

    # model.predict()
    score = model.evaluate(testX, testy, verbose=0)
    # model.save('mnist.h5')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
