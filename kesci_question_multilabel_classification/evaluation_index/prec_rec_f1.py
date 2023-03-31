from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from evaluation_index.utils import reformat


class PrecRecF1Metric(object):
    def __init__(self):
        super().__init__()

    def get_score(self, y_true, y_pred):
        """[summary]

        Args:
            y_true ([type]): [description]
            y_pred ([type]): [description]

        Returns:
            [type]: [%]
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        f1 = f1_score(y_true, y_pred, average='macro') * 100
        p = precision_score(y_true, y_pred, average='macro') * 100
        r = recall_score(y_true, y_pred, average='macro') * 100

        return str((reformat(p, 2), reformat(r, 2), reformat(f1, 2))), reformat(f1, 2)

    def name(self):
        return "prec_rec_f1"

    def __str__(self):
        return super().__str__()


