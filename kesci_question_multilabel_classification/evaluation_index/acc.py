from sklearn.metrics import accuracy_score
import numpy as np


class AccMetric(object):

    def get_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def __str__(self):
        return super().__str__()

    def name(self):
        return "ACC"


if __name__ == "__main__":
    preds = np.array([1, 2, 1, 1, 3])
    labels = np.array([1, 2, 3, 1, 3])
    r = labels == preds
    print(sum(r))