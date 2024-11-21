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
    roc_curve,
    accuracy_score
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
    

def compute_binary_metrics_from_results(results):
    """
    Compute evaluation metrics for binary classification using results from a JSON object.

    Args:
        results (list / json object): A list of dictionaries, where each dictionary contains:
            - 'id': ID of the record.
            - 'pred_prob': List of two floats, [good_prob, bad_prob].
            - 'pred_label': Integer (0, 1) or "miss", the predicted label based on text generation.
            - 'label': Integer (0 or 1), the true label.
            - 'query': (Ignored in this implementation).

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # Extract relevant data from results
    labels = [item['label'] for item in results]
    pos_probs = [item['pred_prob'][1] for item in results]  # Extract probabilities for 'bad' (positive class)
    pred_labels = [item['pred_label'] for item in results]

    # Filter out "miss" entries for text prediction metrics
    valid_indices = [i for i, label in enumerate(pred_labels) if label != "miss"]
    filtered_pred_labels = [pred_labels[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]

    # Compute metrics using probabilities (pred_probs)
    roc_auc = roc_auc_score(labels, pos_probs)
    precision, recall, _ = precision_recall_curve(labels, pos_probs)
    pr_auc = auc(recall, precision)
    # KS score calculation
    fpr, tpr, _ = roc_curve(labels, pos_probs)
    ks_score = max(abs(tpr - fpr))

    # Compute metrics using binary predictions (text_prediction_label)
    accuracy = accuracy_score(filtered_labels, filtered_pred_labels)
    f1 = f1_score(filtered_labels, filtered_pred_labels)

    # Compute "miss" percentage
    miss_percentage = pred_labels.count("miss") / len(pred_labels)

    # Combine metrics into a dictionary
    metrics = {
        'accuracy': accuracy,
        'F1_score': f1,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'KS_score': ks_score,
        'miss_percentage': miss_percentage,
        'num_valid': len(filtered_labels),  # Number of valid predictions
        'num_total': len(labels)  # Total number of instances
    }

    return metrics