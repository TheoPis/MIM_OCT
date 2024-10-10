# adapted from javier/yannis
from sklearn import metrics
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score # Attention: Imbalanced data
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

supported_datasets = ['OctBiom', 'DR', 'OCTID', 'OCTDL', 'OLIVES']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def calculate_metrics(y_true, y_pred, dataset: str):
    if dataset == 'OctBiom':
        dict_metrics = metrics_OctBiom(y_true, y_pred)
    elif dataset == 'OLIVES':
        dict_metrics = metrics_OLIVES(y_true, y_pred)
    elif dataset == 'DR':
        dict_metrics = metrics_DR(y_true, y_pred)
    elif dataset in 'OCTID':
        dict_metrics = metrics_OCTID(y_true, y_pred)
    elif dataset in 'OCTDL':
        dict_metrics = metrics_OCTDL(y_true, y_pred)
    elif dataset in 'STAGE':
        # todo STAGE this should be dataset + task dependent for STAGE
        raise NotImplementedError('STAGE dataset not implemented yet')
    else:
        raise ValueError(f'Dataset {dataset} not supported {supported_datasets}')
    return dict_metrics


def metrics_OctBiom(y_true, y_pred, thresh=0.5):
    labels = ['Healthy', 'SRF', 'IRF', 'HF', 'Drusen', 'RPD', 'ERM', 'GA', 'ORA', 'FPED']
    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)

    roc_auc = metrics.roc_auc_score(y_true, y_pred, multi_class='ovr', average='samples')
    mAP = metrics.average_precision_score(y_true, y_pred, average='micro')
    MAP = metrics.average_precision_score(y_true, y_pred, average='macro')

    y_pred_sigmoid = sigmoid(y_pred)
    prediction_int = np.zeros_like(y_pred_sigmoid)
    prediction_int[y_pred_sigmoid > thresh] = 1

    mf1 = metrics.f1_score(y_true, prediction_int, average='micro')
    Mf1 = metrics.f1_score(y_true, prediction_int, average='macro')
    F1s = metrics.f1_score(y_true, prediction_int, average=None)

    precision_per_class = metrics.precision_score(y_true, prediction_int, average=None)
    recall_per_class = metrics.recall_score(y_true, prediction_int, average=None)

    dic_precision_per_class = dict_metrics_per_class([f'Prec {a}' for a in labels], precision_per_class)
    dic_recall_per_class = dict_metrics_per_class([f'Recall {a}' for a in labels], recall_per_class)
    dic_F1_per_class = dict_metrics_per_class([f'F1 {a}' for a in labels], F1s)

    dict_metrics = {'ROC AUC': roc_auc, 'mAP': mAP, 'MAP': MAP, 'mF1': mf1, 'MF1': Mf1, 'thresh': thresh}
    dict_metrics.update(dic_precision_per_class)
    dict_metrics.update(dic_recall_per_class)
    dict_metrics.update(dic_F1_per_class)
    return dict_metrics

def metrics_OLIVES(y_true, y_pred, thresh=0.5):
    labels = ['Fluid IRF', 'DRT-ME', 'IR HRF', "Atrophy-thinning of retinal layers",
              'Fully attached vitreous face', 'Partially attached vitreous face']
    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)

    roc_auc = metrics.roc_auc_score(y_true, y_pred, multi_class='ovr', average='samples')
    mAP = metrics.average_precision_score(y_true, y_pred, average='micro')
    MAP = metrics.average_precision_score(y_true, y_pred, average='macro')

    y_pred_sigmoid = sigmoid(y_pred)
    prediction_int = np.zeros_like(y_pred_sigmoid)
    prediction_int[y_pred_sigmoid > thresh] = 1

    mf1 = metrics.f1_score(y_true, prediction_int, average='micro')
    Mf1 = metrics.f1_score(y_true, prediction_int, average='macro')
    F1s = metrics.f1_score(y_true, prediction_int, average=None)
    precision_per_class = metrics.precision_score(y_true, prediction_int, average=None)
    recall_per_class = metrics.recall_score(y_true, prediction_int, average=None)

    dic_precision_per_class = dict_metrics_per_class([f'Prec {a}' for a in labels], precision_per_class)
    dic_recall_per_class = dict_metrics_per_class([f'Recall {a}' for a in labels], recall_per_class)
    dic_F1_per_class = dict_metrics_per_class([f'F1 {a}' for a in labels], F1s)
    # recall_per_class = metrics.recall_score(y_true, prediction_int, average=None)
    # dic_recall_per_class = dic_metrics_per_class(labels, precision_per_class)

    dict_metrics = {'ROC AUC': roc_auc, 'mAP': mAP, 'MAP': MAP, 'mF1': mf1, 'MF1': Mf1, 'thresh': thresh}
    dict_metrics.update(dic_precision_per_class)
    dict_metrics.update(dic_recall_per_class)
    dict_metrics.update(dic_F1_per_class)
    # dict_metrics.update(dic_recall_per_class)
    return dict_metrics


def metrics_DR(y_true, y_pred, thresh=0.5):
    label_names = ['NPDR', 'PDR']
    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)

    y_pred_hard = np.argmax(softmax(y_pred), axis=1)
    y_pred_soft_PDR = softmax(y_pred)[:, 1]

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_soft_PDR)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_soft_PDR)
    acc = accuracy_score(y_true, y_pred_hard)
    auc = roc_auc_score(y_true, y_pred_soft_PDR)

    ap = average_precision_score(y_true, y_pred_soft_PDR)
    confusion = confusion_matrix(y_true, y_pred_hard, labels=[0, 1])
    dict_metrics = {'accuracy': acc,
                    # 'tpr': tpr,
                    # 'fpr': fpr,
                    # 'threshold_roc': thresholds_roc,
                    'ROC AUC': auc,
                    # 'precision': precision,
                    # 'recall': recall,
                    # 'threshold_pr': thresholds_pr,
                    'AP': ap}
    # return {'accuracy': accuracy_score(y_true, y_pred),
    #         'tpr': tpr,
    #         'fpr': fpr,
    #         'threshold_roc': thresholds_roc,
    #         'auc': roc_auc_score(y_true, y_soft),
    #         'precision': precision,
    #         'recall': recall,
    #         'threshold_pr': thresholds_pr,
    #         'ap': average_precision_score(y_true, y_soft),
    #         'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0,1])}


    # dict_metrics = {'ROC AUC': roc_auc, 'mAP': mAP, 'MAP': MAP, 'mF1': mf1, 'MF1': Mf1, 'thresh': thresh}
    # dict_metrics.update(dic_precision_per_class)
    # dict_metrics.update(dic_recall_per_class)
    return dict_metrics


def metrics_OCTDL(y_true, y_pred, thresh=0.5):
    label_names = ['NO', 'AMD', 'DME', 'ERM', 'RAO', 'RVO', 'VID']

    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
        # assert y_true.shape[1] == len(label_names)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)
        assert y_pred.shape[1] == len(label_names)

    y_pred_soft = softmax(y_pred)
    y_pred_hard = np.argmax(y_pred_soft, axis=1)

    # fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_soft_PDR)
    # precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_soft)
    acc = accuracy_score(y_true, y_pred_hard)
    auc = roc_auc_score(y_true, y_pred_soft, multi_class='ovr', average='weighted')
    mP = metrics.precision_score(y_true, y_pred_hard, average='micro')
    MP = metrics.precision_score(y_true, y_pred_hard, average='macro')

    mf1 = metrics.f1_score(y_true, y_pred_hard, average='micro')
    Mf1 = metrics.f1_score(y_true, y_pred_hard, average='macro')

    f1_per_class = metrics.f1_score(y_true, y_pred_hard, average=None)
    f1_per_class = dict_metrics_per_class(label_names, f1_per_class)

    # confusion = confusion_matrix(y_true, y_pred_hard, labels=[0, 1, 2, 3, 4])
    dict_metrics = {'accuracy': acc,
                    'ROC AUC': auc,
                    'mP': mP,
                    'MP': MP,
                    'mF1': mf1,
                    'MF1': Mf1
                    }
    dict_metrics.update(f1_per_class)
    return dict_metrics


def metrics_OCTID(y_true, y_pred, thresh=0.5):
    label_names = ['NORMAL', 'AMD', 'DR', 'MH', 'CSC']

    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
        # assert y_true.shape[1] == len(label_names)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)
        assert y_pred.shape[1] == len(label_names)

    y_pred_soft = softmax(y_pred)
    y_pred_hard = np.argmax(y_pred_soft, axis=1)

    # fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_soft_PDR)
    # precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_soft)
    acc = accuracy_score(y_true, y_pred_hard)
    auc = roc_auc_score(y_true, y_pred_soft, multi_class='ovr', average='weighted')
    mP = metrics.precision_score(y_true, y_pred_hard, average='micro')
    MP = metrics.precision_score(y_true, y_pred_hard, average='macro')

    mf1 = metrics.f1_score(y_true, y_pred_hard, average='micro')
    Mf1 = metrics.f1_score(y_true, y_pred_hard, average='macro')

    f1_per_class = metrics.f1_score(y_true, y_pred_hard, average=None)
    f1_per_class = dict_metrics_per_class(label_names, f1_per_class)

    # confusion = confusion_matrix(y_true, y_pred_hard, labels=[0, 1, 2, 3, 4])
    dict_metrics = {'accuracy': acc,
                    'ROC AUC': auc,
                    'mP': mP,
                    'MP': MP,
                    'mF1': mf1,
                    'MF1': Mf1
                    }
    dict_metrics.update(f1_per_class)
    return dict_metrics


def dict_metrics_per_class(labels, results):
    dic_results = {}
    for label, result in zip(labels, results):
        dic_results[label] = result
    return dic_results
