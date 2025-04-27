import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import logging


def shap_feature_attribution(shap_values):
    """
    Calculate the mean absolute SHAP values for each feature across all samples, and optionally,
    across all classes for multi-class classification problems.

    This function processes the SHAP values generated for a model's predictions, averaging the absolute
    values of these SHAP values across all data points to quantify the overall impact of each feature
    on the model's predictions. For multi-class classification, it averages the impact across all classes
    to provide a single importance score per feature.

    Parameters:
    - shap_values (shap.Explanation): An object containing SHAP values for the model's predictions. This can
      be a 2D array for binary classification/regression models, or a 3D array for multi-class classification
      models, with dimensions [samples, features, classes].

    Returns:
    - pd.DataFrame: A DataFrame with a single column 'mean_abs_shap' listing the mean absolute SHAP values for
      each feature, indexed by feature names. The DataFrame is sorted in descending order of mean absolute
      SHAP values, highlighting the most influential features at the top.
    """
    # Determine if SHAP values are 3D (multi-class classification)
    if shap_values.values.ndim == 3:
        # Compute the mean absolute SHAP value across all classes for each feature
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0).mean(axis=1)
    else:
        # For 2D SHAP values (binary classification), compute mean absolute value directly across samples
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

    # Creating DataFrame with features and their corresponding mean absolute SHAP values
    feat_att = pd.DataFrame(mean_abs_shap, index=shap_values.feature_names, columns=['mean_abs_shap']).sort_values(
        by='mean_abs_shap', ascending=False)

    return feat_att


# Create an instance of the SHAP Explainer class, passing in the trained model and the training set as arguments,
# and returns explainer instance
def gen_explainer_instance(model, X_train):
    '''Create an instance of the SHAP Explainer class, passing in the trained model and the training set as arguments,
and returns explainer instance'''
    explainer = shap.Explainer(model, X_train)
    logging.info(f"Explainer Used for model: {model} ==========================> {explainer}")
    return explainer


# Generate SHAP values for trained data set and return the SHAP values and explainer
def gen_shap_value(explainer, X_train, rfc_flag, linear_model_flag):
    '''Generate SHAP values for trained data set and return the SHAP values and explainer'''
    if linear_model_flag == 'Y':
        logging.info("Linear Model Only logic running")
        shap_values = explainer(X_train)
        logging.info(f"Shap values shape for the linear model {shap_values.values.shape}")
        logging.info(f"Shap_values=================================>{shap_values}")
    else:
        logging.info("Non Linear Model logic running")
        shap_values = explainer(X_train, check_additivity=False)
        logging.info(f"Shap values shape for the non Linear model {shap_values.values.shape}")
        logging.info(f"Shap_values=================================>{shap_values}")

    feat_att = shap_feature_attribution(shap_values)

    logging.info(f"Feat Att===================================>{feat_att}")

    mean_shap_values = np.abs(shap_values.values).mean(axis=0)

    return shap_values, explainer, mean_shap_values, feat_att


# visualization of the shap values
def plot_model_explanation(shap_value, X_train):
    return shap.summary_plot(shap_value, X_train)


# Visualize the SHAP values using the function from the SHAP library.
def gen_visualization(shap_values, X_train, explainer, exp_data_df, mean_shap_values,  feat_att, rfc_flag, linear_model_flag):
    if linear_model_flag == 'Y':
        shap_value_obj = explainer.shap_values(X_train)
        logging.info(f"shap_value_obj for Linear Model =========================================> {shap_value_obj} ")
    else:
        shap_value_obj = explainer.shap_values(X_train, check_additivity=False)
        logging.info(f"shap_value_obj for Non Linear Model =========================================> {shap_value_obj} ")

    plot_model_explanation(shap_value_obj, X_train)

    plt.tight_layout()
    plt.savefig("Experiment_Visualization_Output/explanation_bar")
    plt.close()
    shap_result_dict = feat_att['mean_abs_shap'].to_dict()
    return shap_result_dict


# Create SHAP (SHapley Additive exPlanations) explain the output of machine learning model
def create_shapley(model, X_train, rfc_flag, linear_model_flag, exp_data_df, target):
    """
    :param model:
    :param X_train:
    :param rfc_flag:
    :param linear_model_flag:
    :param exp_data_df:
    :param target:
    :return:
    """

    logging.info(f"Creating shap values: model received: {model}")
    trained_params = model.get_params()
    logging.info(f"trained with param: {trained_params} ")
    exp_data_df = exp_data_df.drop(target, axis=1)
    shap_values, explainer, mean_shap_values,  feat_att = gen_shap_value(gen_explainer_instance(model, X_train), X_train, rfc_flag, linear_model_flag)
    shap_result_dict = gen_visualization(shap_values, X_train,explainer, exp_data_df, mean_shap_values,  feat_att, rfc_flag, linear_model_flag)

    return shap_result_dict





