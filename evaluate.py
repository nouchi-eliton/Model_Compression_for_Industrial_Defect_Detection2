import tensorflow as tf
from metrics import BalancedAccuracy
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def eval_model_logits(y_true, y_pred_logits):
    """
    Evaluate using logits. Diplays the confusion matrix and calculates precision, recall, f1-score, and balanced accuracy

    Args:
        y_true: true labels.
        y_pred_logits: model predictions logits

    Returns:
        List containing the metrics.
    """
    y_pred = tf.argmax(y_pred_logits, axis=-1)
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Defective'], output_dict=True)
    balanced_acc = BalancedAccuracy()(y_true, y_pred).numpy().item()
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues', display_labels=['Normal', 'Defective']);
    plt.show()

    precision = report['Defective']['precision']
    recall = report['Defective']['recall']
    f1 = report['Defective']['f1-score']
    return [precision, recall, f1, balanced_acc]


def eval_tflite(tflite_model, x, y_true, batch_size):
    """
    Evaluate tflite model.

    Args:
        tflite_model: tflite model to be evaluated.
        x: images to be used in the evaluation.
        y_true: labels of the images.
        batch_size: int. Number of images in each batch.

    Returns:
        List containing the metrics.
    """
    interpreter = tf.lite.Interpreter(model_content= tflite_model)
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    n_images = x.shape[0]
    if tuple(input_shape[1:]) != x.shape[1:]:
        temp = []
        for i in range(0, n_images, batch_size):
            batch_resized = tf.cast(tf.image.resize_with_pad(x[i: i + batch_size], input_shape[1], input_shape[2]), tf.float32)
            temp.append(batch_resized)
        x = tf.concat(temp, axis=0)
    else:
        x = tf.cast(x, tf.float32)

    interpreter.resize_tensor_input(input_index, (batch_size,) + x.shape[1:])
    interpreter.allocate_tensors()
    y_pred_logits = []
    for i in range(0, n_images, batch_size):
        if (i + batch_size) < n_images:
            batch = x[i : i + batch_size]
        else:
            batch = x[i:]
            pad = np.zeros((batch_size - batch.shape[0], *batch.shape[1:]), dtype=np.float32)
            batch = np.vstack([batch, pad])

        interpreter.set_tensor(input_index, batch)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        y_pred_logits.append(output)

    y_pred_logits = np.vstack(y_pred_logits)[:n_images]
    report_test = eval_model_logits(y_true, y_pred_logits)

    return report_test

def plot_metrics(object_metrics, obj):
    """
    Plot a scatterplot comparing the size and f1-score in each stage of compression.
    Plot a lineplot with the evolution of the precision, recall, f1-score and balanced accuracy in each stage of compression.

    Args:
        object_metrics: dict containing all the metrics of the object.
        obj: str. Name of the object.
    """
    index_metrics = ['precision', 'recall', 'f1-score', 'balanced_acc', 'size_mb']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(obj)

    df_metrics = pd.DataFrame(object_metrics, index= index_metrics).T
    axes[0].set_title('Size MB x F1_score')
    scatter = sns.scatterplot(ax= axes[0], data= df_metrics, x= 'size_mb', y= 'f1-score', hue=df_metrics.index)
    scatter.legend_.set_title('Models')
    axes[1].set_title('Metrics evolution')
    sns.lineplot(ax= axes[1], data= df_metrics.iloc[:, :-1]) # remove the size_mb from the plot
    plt.tight_layout()
    plt.show()

