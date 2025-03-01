import os.path
from datetime import datetime
import openpyxl
import torch
from sklearn import metrics
import numpy as np
import random, json
import scipy.io as scio

def print_args(args):
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

def get_batch_from_list(data_list, indices):
    selected_elements = list(map(lambda i: data_list[i], indices))
    return selected_elements

def evaluate(y_score, y_true):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)
    return auc, aupr

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def segment_sequence(lst, chunk_size, pad_id):
    seg_result = np.array_split(lst, len(lst) // chunk_size + (len(lst) % chunk_size > 0))

    for i, item in enumerate(seg_result):
        if len(item) < chunk_size:
          item = np.pad(item, (0, chunk_size - len(item)), 'constant', constant_values=pad_id)
          seg_result[i] = item

    return seg_result

def evaluate_test(results, args, de_nova=False):
    drug_id = torch.cat([x[0] for x in results]).cpu().tolist()
    disease_id = torch.cat([x[1] for x in results]).cpu().tolist()
    prediction = torch.cat([x[2] for x in results]).cpu().tolist()
    label = torch.cat([x[3] for x in results]).cpu().tolist()
    drug_disease_relation = list(zip(drug_id, disease_id))
    set_drug_disease = set(drug_disease_relation)

    results = []
    labels = []
    i = 0
    for item in set_drug_disease:
        i += 1
        indicts = [i for i, x in enumerate(drug_disease_relation) if x == item]
        scores = [prediction[i] for i in indicts]
        scores_pos = [score for score in scores if score > args.T_score]
        scores_neg = [score for score in scores if score < args.T_score]
        if len(scores_pos) > len(scores_neg):
            results.append(np.mean(scores_pos))
        elif len(scores_pos) < len(scores_neg):
            results.append(np.mean(scores_neg))
        else:
            results.append(np.mean(scores))
        labels.append(label[indicts[0]])

    auc, aupr = evaluate(results, labels)

    if de_nova:
        return results, labels, auc, aupr
    else:
        return auc, aupr

def save_experiment_results(args, auroc, aupr):
    filename = datetime.now().strftime("%Y%m%d%H%M%S")
    results = {
        "hyperparameters": vars(args),
        "auroc": auroc,
        "aupr": aupr
    }
    with open(os.path.join('log', filename + '.json'), "w") as f:
        json.dump(results, f, indent=4)

