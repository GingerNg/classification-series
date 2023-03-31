from cnn.models import MultiClassModel

if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_blobs
    from keras.utils import to_categorical
    # generate 2d classification dataset
    n_samples = 1100
    X, y = make_blobs(n_samples=1100, centers=3, n_features=20, cluster_std=2, random_state=2)
    multi_label_y = []
    # one hot encode output variable
    y = to_categorical(y)
    # split into train and test
    n_train = 100
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    print(trainX.shape, testX.shape)

    model = MultiClassModel(feature_dim=20,label_dim=3)

    model.fit(trainX, trainy, epochs=500, verbose=0)

    score = model.evaluate(testX, testy, verbose=0)
    # model.save('mnist.h5')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
