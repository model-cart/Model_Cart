import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import importlib
import logging
from pandas_profiling import ProfileReport
from Model_Training import ModelTraining


def Data_prep_tts(dataset, data_module, target, data_param_dict, csv_delimiter):
    ''' Load the Experiment dataset &
Prepare the data for training'''

    logging.info(f"Dataset : {dataset}\n Data Module: {data_module}\n target: {target}")

    # Load the dataset from library, but we don't need this in real scenario
    if dataset != None:
        load_dataset = importlib.import_module(dataset)
        load_datamodule = getattr(load_dataset, data_module)
        exp_data = load_datamodule()
        # Convert to a pandas dataframe
        exp_data_df = pd.DataFrame(data=exp_data.data, columns=exp_data.feature_names)
        # Add the target column
        exp_data_df[target] = exp_data.target
        print(f"exp_data_df:\n {exp_data_df}")

        # Split into X and y : split the dataset into input features (X) and target values (y)
        X = exp_data_df.drop(target, axis=1)
        y = exp_data.target

    else:
        exp_data_df = pd.read_csv(data_module, delimiter=csv_delimiter)

        X = exp_data_df.drop(target, axis=1)
        y = exp_data_df.loc[:, target]

    # Create the Pandas-Profiling report
    profile = ProfileReport(exp_data_df, title="Dataset Profiling Report", explorative=True)

    # Save the report as an HTML file
    profile.to_file("Experiment_reports/dataset_report.html")

    if 'stratify' in data_param_dict:
        data_param_dict['stratify'] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, **data_param_dict)
    return X_train, X_test, y_train, y_test, X, y, exp_data_df, target



def Data_prep_model_train_kfold(dataset, data_module, target, split_method, split_fold, experiment_name, lst_models, lst_algo_param, user_eval_metrics_dict, lst_model_description,data_parm_dict, csv_delimiter,exp_username, yaml_path, exp_thread_id):
    '''Load the Experiment dataset &
Prepare the data for training, and
Call the Model Training module to train model for each iteration of folds as requested by the user
'''

    logging.info(f"Dataset : {dataset}\n Data Module: {data_module}\n target: {target}")

    # Load the dataset from library, but we dont need this in reality
    if dataset != None:
        load_dataset = importlib.import_module(dataset)
        load_datamodule = getattr(load_dataset, data_module)
        exp_data = load_datamodule()
        # Convert to a pandas dataframe
        exp_data_df = pd.DataFrame(data=exp_data.data, columns=exp_data.feature_names)
        # Add the target column
        exp_data_df[target] = exp_data.target
        # Split into X and y : split the dataset into input features (X) and target values (y)
        X = exp_data_df.drop(target, axis=1)
        y = exp_data.target
    else:
        exp_data_df = pd.read_csv(data_module, delimiter=csv_delimiter)

        X = exp_data_df.drop(target, axis=1)
        y = exp_data_df.loc[:, target]

    # Create the Pandas-Profiling report
    profile = ProfileReport(exp_data_df, title="Dataset Profiling Report", explorative=True)

    # Save the report as an HTML file
    profile.to_file("Experiment_reports/dataset_report.html")

    kf = KFold(n_splits=split_fold)
    logging.info("Kfold iteration Started-------------------------------->")
    # Loop over each fold
    for train_index, test_index in kf.split(X):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        ModelTraining.ML_Model(experiment_name, X_train, X_test, y_train, y_test, X, y, exp_data_df, lst_models,
                               lst_algo_param, target, user_eval_metrics_dict, lst_model_description,split_method, split_fold, data_parm_dict, exp_username, yaml_path, exp_thread_id)
    logging.info("Kfold iteration Ended-------------------------------->")







