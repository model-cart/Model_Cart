import logging
import numpy
import numpy as np
from sklearn.metrics import confusion_matrix


def calc_performance(y_test, y_pred):
    """
    Calculate model performance metrics for both binary and multi-class classification.
    Returns overall accuracy and macro-averaged Recall, Precision, and F1 Score
    :param y_test: True labels
    :param y_pred: Predicted labels
    :return: Overall accuracy, Macro-averaged Recall, Precision, and F1 Score
    """
    try:
        logging.info("Calculating model performance metrics ======>")

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logging.info(f"Confusion Matrix ============================>{cm}")

        # Overall accuracy
        try:
            accuracy = np.trace(cm) / np.sum(cm)
            logging.info(f"Manually calculated Accuracy Score: {accuracy}")
            if accuracy <= 0 or accuracy is numpy.NaN:
                raise Exception("encountered Nan in calculation")
        except Exception as e:
            logging.error(f"An error occurred while calculating accuracy: {e}")
            return None, None, None, None

        try:
                # Precision, Recall, and F1 for each class
            precision = np.diag(cm) / np.sum(cm, axis=0)
            recall = np.diag(cm) / np.sum(cm, axis=1)
            f1_scores = 2 * (precision * recall) / (precision + recall)

            logging.info(f"Manually calculated Precision: {precision}")
            logging.info(f"Manually calculated Recall: {recall}")
            logging.info(f"Manually calculated F1 Score: {f1_scores}")

            # Macro-averaged metrics
            macro_precision = np.nanmean(precision)
            macro_recall = np.nanmean(recall)
            macro_f1 = np.nanmean(f1_scores)

            logging.info(f"Manually calculated Macro-averaged Precision: {macro_precision}")
            logging.info(f"Manually calculated Macro-averaged Recall: {macro_recall}")
            logging.info(f"Manually calculated Macro-averaged F1 Score: {macro_f1}")
            if macro_recall <= 0 or macro_recall is numpy.NaN or \
                    macro_precision <= 0 or macro_precision is numpy.NaN or \
                    macro_f1 <= 0 or macro_f1 is numpy.NaN:
                raise Exception("Encountered nan value while calculating metrics")
        except Exception as e:
            logging.error(f"An error occurred while calculating Metrics: {e}")
            return None, None, None, None

        return accuracy, macro_recall, macro_precision, macro_f1
    except Exception as e:
        logging.error(f"An error occurred while calculating performance metrics: {e}")
        return None, None, None, None  # Return None for each metric if there's an error


def eval_model_performance(lst_eval_metrics, algorithm, X_test, y_test, run):
    """
    Calculate the performance of the model for the user provided evaluation metrics
    :param run:
    :param y_test:
    :param X_test:
    :param algorithm:
    :param lst_eval_metrics:
    :return: performance metrics for the model
    """
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    # Predict the classes of the test set
    y_pred = algorithm.predict(X_test)
    user_eval_metrics_dict = lst_eval_metrics[run]
    calc_accuracy_score = None
    calc_recall_score = None
    calc_precision_score = None
    calc_f1_score = None

    model_performance_metrics_dict = {}
    for metrics_name, metrics_object in user_eval_metrics_dict.items():
        if metrics_name == 'precision_score' or metrics_name == 'recall_score' or metrics_name == 'f1_score':
            model_performance_metrics_dict[metrics_name] = metrics_object(y_test, y_pred, average='weighted')
        else:
            model_performance_metrics_dict[metrics_name] = metrics_object(y_test, y_pred)
            if metrics_name == 'confusion_matrix':
                calc_accuracy_score, calc_recall_score, calc_precision_score, calc_f1_score = \
                    calc_performance(y_test, y_pred)

    if calc_accuracy_score is not None and calc_recall_score is not None and calc_precision_score is not None \
            and calc_f1_score is not None:
        for metrics_name, metrics_value in model_performance_metrics_dict.items():
            if metrics_name == 'accuracy_score':
                if calc_accuracy_score != metrics_value:
                    logging.info(
                        f"Using Calculated Accuracy Score {calc_accuracy_score} as the function generated doesn't match {metrics_value}")
                    model_performance_metrics_dict[metrics_name] = calc_accuracy_score
            elif metrics_name == 'precision_score':
                if calc_precision_score != metrics_value:
                    logging.info(
                        f"Using Calculated Precision Score {calc_precision_score} as the function generated doesn't match {metrics_value}")
                    model_performance_metrics_dict[metrics_name] = calc_precision_score
            elif metrics_name == 'recall_score':
                if calc_recall_score != metrics_value:
                    logging.info(
                        f"Using Calculated Recall Score {calc_recall_score} as the function generated doesn't match {metrics_value}")
                    model_performance_metrics_dict[metrics_name] = calc_recall_score
            elif metrics_name == 'f1_score':
                if calc_f1_score != metrics_value:
                    logging.info(
                        f"Using Calculated f1 Score {calc_f1_score} as the function generated doesn't match {metrics_value}")
                    model_performance_metrics_dict[metrics_name] = calc_f1_score

    return model_performance_metrics_dict


