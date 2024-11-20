import os 
import io
import json
import logging
import colorlog
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    f1_score, 
    roc_curve
)



def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(
            obj, 
            f, 
            ensure_ascii=False,
            indent=indent, 
            default=default
            )
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


## Logging
def setup_logging():
    # Set up file handler
    file_handler = logging.FileHandler("app.log")  
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
            )
        )

    # Set up color console handler using colorlog
    color_handler = colorlog.StreamHandler()
    color_handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s: %(message)s",
        log_colors={
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    ))

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the lowest log level to capture all messages
    logger.addHandler(file_handler)
    logger.addHandler(color_handler)
    

## Evalution
def compute_binary_metrics_from_results(results):
    """
    Compute evaluation metrics for binary classification using results from a JSON object.

    Args:
        results (list / json object): A list of dictionaries, where each dictionary contains:
            - 'id': ID of the record.
            - 'pred_prob': List of two floats, [negative_prob, positive_prob].
            - 'label': Integer (0 or 1), the true label.
            - 'query': (Ignored in this implementation).

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # Step 1: Extract labels and predictions
    labels = [item['label'] for item in results]
    pos_probs = [item['pred_prob'][1] for item in results]  # Extract positive class probabilities
    predictions = [1 if prob > 0.5 else 0 for prob in pos_probs]  # Threshold at 0.5 for binary prediction

    # Step 2: Compute metrics
    accuracy = sum(1 for pred, label in zip(predictions, labels) if pred == label) / len(labels)
    roc_auc = roc_auc_score(labels, pos_probs)
    precision, recall, _ = precision_recall_curve(labels, pos_probs)
    pr_auc = auc(recall, precision)
    f1 = f1_score(labels, predictions)
    
    # KS score calculation
    fpr, tpr, _ = roc_curve(labels, pos_probs)
    ks_score = max(abs(tpr - fpr))

    # Step 3: Combine metrics into a dictionary
    metrics = {
        'accuracy': accuracy,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'F1_score': f1,
        'KS_score': ks_score,  # Add KS score to metrics
        'num': len(labels)  # Number of evaluated instances
    }

    return metrics