"""Evaluation helpers."""
from __future__ import annotations
import numpy as np
try:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def evaluate(y_true, y_pred, scores=None):
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp = int(((y_true==1)&(y_pred==1)).sum()); fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum()); tn = int(((y_true==0)&(y_pred==0)).sum())
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    metrics = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
               "n_true_anomalies": int(y_true.sum()), "n_predicted_anomalies": int(y_pred.sum()),
               "n_total": len(y_true), "roc_auc": None}
    if HAS_SKLEARN and scores is not None and len(np.unique(y_true)) > 1:
        try: metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        except ValueError: pass
    return metrics

def print_report(metrics, method=""):
    print("\n" + "="*50)
    print(f" Evaluation Report {'— '+method if method else ''} ".center(50, "="))
    print("="*50)
    print(f"  Precision: {metrics.get('precision',0):.4f}  Recall: {metrics.get('recall',0):.4f}  F1: {metrics.get('f1',0):.4f}")
    if "tp" in metrics: print(f"  TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    print("="*50 + "\n")
