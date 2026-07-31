import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred, target_names=None):
    """Compute Accuracy, Precision, Recall, Macro F1, and Per-Class F1."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    cm = confusion_matrix(y_true, y_pred)

    class_metrics = {}
    if target_names:
        for idx, name in enumerate(target_names):
            class_metrics[name] = {
                'precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'f1': float(f1[idx]),
                'support': int(support[idx])
            }

    return {
        'accuracy': float(acc),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': class_metrics,
        'confusion_matrix': cm.tolist()
    }
