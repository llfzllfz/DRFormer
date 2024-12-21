from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import matthews_corrcoef

def update_metric(x, y):
    for key in x.keys():
        y[key] = x[key]
        # if key not in y.keys():
        #     y[key] = x[key]
        # else:
        #     y[key] = max(x[key], y[key])

def Init_metric():
    result = {
        'ACC': 0,
        'AUROC': 0,
        'PRECISION': 0,
        'RECALL': 0,
        'AUPRC': 0,
        'F1': 0,
        'MCC': 0,
        'LOSS': 1e5
    }
    return result

def pd_to_metric(data, loss):
    result = {'LOSS': loss}
    for column in data.columns.tolist():
        result[column] = data[column].mean()
    return result

def cal_metric(pred, label, loss):
    # print(pred[:100])
    # print(label[:100])
    acc = Accuracy(label, pred).item()
    auroc = Roc(label, pred).item()
    auprc = Prc(label, pred).item()
    precision = Precision(label, pred).item()
    recall = Recall(label, pred).item()
    f1 = F1(label, pred).item()
    mcc_ = mcc(label, pred).item()
    result = {
        'ACC': acc,
        'AUROC': auroc,
        'PRECISION': precision,
        'RECALL': recall,
        'AUPRC': auprc,
        'F1': f1,
        'MCC': mcc_,
        'LOSS': loss
    }
    return result

def list_to_metric(x):
    result = {
        'ACC': [],
        'AUROC': [],
        'PRECISION': [],
        'RECALL': [],
        'AUPRC': [],
        'F1': [],
        'MCC': []
    }
    for tmp in x:
        for key in tmp.keys():
            result[key].append(tmp[key])
    return result


def Accuracy(label, prediction):
    metric = np.array(accuracy_score(label, np.round(prediction)))
    return metric

def Roc(label, prediction):
    ndim = np.ndim(label)
    fpr, tpr, thresholds = roc_curve(label, prediction)
    score = auc(fpr, tpr)
    metric = np.array(score)
    return metric

def Prc(label, prediction):
    precision, recall, threshold = precision_recall_curve(label, prediction)
    auprc = auc(recall, precision)
    return np.array(auprc)

def Precision(label, prediction):
    return np.array(precision_score(label, np.round(prediction), average = 'macro'))

def Recall(label, prediction):
    return np.array(recall_score(label, np.round(prediction), average = 'macro'))

def F1(label, prediction):
    return np.array(f1_score(label, np.round(prediction), average = 'macro'))

def mcc(label, prediction):
    metric = np.array(matthews_corrcoef(label, np.round(prediction)))
    return metric