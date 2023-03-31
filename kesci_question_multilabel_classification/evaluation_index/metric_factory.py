from evaluation_index  import acc, prec_rec_f1, f1


def get_metric(name):
    if name == 'acc':
        return acc.AccMetric()
    elif name == 'prec_rec_f1':
        return prec_rec_f1.PrecRecF1Metric()
    elif name == 'f1':
        return f1.F1Metric()



