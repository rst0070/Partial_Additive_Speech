from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_EER(scores, labels):
    if len(scores) != len(labels):
        raise Exception('length between scores and labels is different')
    elif len(scores) == 0:
        raise Exception("There's no elements in scores")
        
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return EER