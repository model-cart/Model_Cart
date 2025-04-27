from Model_Training import ModelTraining
from Data_Processing import DataLoading
from YAML_Processing import ProcessYAML
import logging
import argparse

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Execute ML experiment based on YAML configuration.')
    parser.add_argument('--yaml', required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--username', required=True, help='experiment username')
    parser.add_argument('--thread', required=True, help='thread allocated to the experiment')
    args = parser.parse_args()
    yaml_path = args.yaml
    exp_username = args.username
    exp_thread_id = args.thread

    config = ProcessYAML.load_config(yaml_path)
    experiment_name = ProcessYAML.getexperiment(config)
    lst_models, lst_algo_param, lst_eval_metrics, lst_model_description = ProcessYAML.load_module(config)
    dataset, data_module, target, split_method, split_fold, data_param_dict, csv_delimiter = ProcessYAML.load_data(config)

    # call modules for different phases here.
    # evaluate performance on the split, but the final model artifact should be trained on entire dataset,
    # also the explanation should be performed on the entire dataset.
    if split_method == 'train_test_split':
        X_train, X_test, y_train, y_test, X, y, exp_data_df, target = DataLoading.Data_prep_tts(dataset, data_module, target, data_param_dict, csv_delimiter)
        logging.info("Data prep Completed-------------------------------->")
        ModelTraining.ML_Model(experiment_name, X_train, X_test, y_train, y_test, X, y, exp_data_df, lst_models, lst_algo_param, target, lst_eval_metrics, lst_model_description,split_method, split_fold, data_param_dict, exp_username, yaml_path, exp_thread_id)
    elif split_method == 'KFold':
        experiment_name = f"{experiment_name}_{split_method}_with nsplit = {split_fold}"
        DataLoading.Data_prep_model_train_kfold(dataset, data_module, target, split_method, split_fold, experiment_name, lst_models, lst_algo_param, lst_eval_metrics, lst_model_description,data_param_dict, csv_delimiter,exp_username, yaml_path, exp_thread_id)