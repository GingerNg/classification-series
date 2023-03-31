

"""
训练数据的label请用0和1的向量来表示
binary_crossentropy

categorical_crossentropy
"""

from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from keras.utils import plot_model


def MultiClassModel(feature_dim,label_dim):

    model = Sequential()
    print("create model. feature_dim = %s, label_dim = %s" % (feature_dim,label_dim))
    model.add(Dense(500, activation='relu', input_dim=feature_dim))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(label_dim, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def baseline_model(y,max_features,embedding_dims,filters): # cnn
    kernel_size = 3

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims))        # 使用Embedding层将每个词编码转换为词向量
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # 池化
    model.add(GlobalMaxPooling1D())

    model.add(Dense(y.shape[1], activation='softmax')) #第一个参数units: 全连接层输出的维度，即下一层神经元的个数。
    model.add(Dropout(0.2))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    return model


def MultiLabelModel(feature_dim,label_dim):

    model = Sequential()
    print("create model. feature_dim = %s, label_dim = %s" % (feature_dim,label_dim))
    model.add(Dense(500, activation='relu', input_dim=feature_dim))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(label_dim, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # plot_model(model, to_file='multi_label_model.png')
    return model


