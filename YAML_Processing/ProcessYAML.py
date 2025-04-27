import yaml
import logging
import inspect
import importlib


def get_params_for_algo(installed_algo):
    '''
    prepare the list of parameters that are acceptable for the model
    :param installed_algo: user provided model for which the list of parameter is to be prepared
    :return param_dtc:  list of acceptable parameter for the model
    '''
    # Get the signature of the machine learning model
    sig = inspect.signature(installed_algo)

    # Get the parameters from the signature
    p = sig.parameters

    param_dtc = []
    # prepare the list of parameters accepted for the model
    for name, default in p.items():
        param_dtc.append(name)

    return param_dtc


def validate_params(installed_algo, algo_param):
    '''
    validate the user provided list of parameter for the model with the list of acceptable parameter for that model
    :param installed_algo:  user provided model for which the list of parameter is to be prepared
    :param algo_param: user provided list of parameters for the model
    :return:
    Stop's further processing if validation fails
    '''
    user_param = [*algo_param]
    list_param = []

    # param_dtc = get_params_for_algo(installed_algo)
    valid_params = installed_algo().get_params().keys()
    for param in user_param:
        if param not in valid_params:
            list_param.append(param)
            logging.warning(f"parameter '{param}' is not a valid input for the {installed_algo}")
    if list_param != []:
        exit(f"The list of parameter {list_param} not accepted by {installed_algo}, Correct and try again")



def load_module(config):
    ''' fetch the required modules for the experiment from YAML file and dynamically import the required models for the experiment'''
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    module_name = config['models']
    algo_param = {} #dictionary to load all parameters
    module_loaded = {}
    user_eval_metrics_dict = {}
    lst_models = []
    lst_param = []
    lst_eval_metrics = []
    lst_model_description = {}

    for models in module_name:
        m_ = module_name[models]["module"]
        n_ = module_name[models]["name"]
        print(f"module:{m_}")
        print(f"name:{n_}")
        install_module = importlib.import_module(module_name[models]["module"])
        print(f"loading module: {install_module}")
        installed_algo = getattr(install_module, module_name[models]["name"])
        print(f"loading algo: {installed_algo}")
        user_eval_metrics_dict = load_eval_metrics(module_name[models]["evaluation"])
        module_loaded[module_name[models]["name"]] = installed_algo
        lst_models.append(module_loaded)
        lst_model_description[module_name[models]["name"]] = module_name[models]["description"]

        module_loaded = {}

        if module_name[models]["params"]:
            logging.info("Using User provided parameters")
            validate_params(installed_algo, module_name[models]["params"])
            algo_param[module_name[models]["name"]] = module_name[models]["params"]
            lst_param.append(algo_param)
            algo_param = {}
        else:
            # setting up default parameters for model if user didn't provide
            logging.info("Using Default Parameters as User didn't provided model parameters")
            algo_param[module_name[models]["name"]] = installed_algo().get_params()
            lst_param.append(algo_param)
            algo_param = {}
        # validate_hyperparams(module_name[models]["name"], module_name[models]["hyperparameters"])
        lst_eval_metrics.append(user_eval_metrics_dict)
        user_eval_metrics_dict = {}
    return lst_models, lst_param, lst_eval_metrics, lst_model_description


def load_config(yaml_path):
    ''' returns the configuration of the experiment set by the end user as config variable
    '''
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_data(config):
    ''' returns the user provided dataset and the data_module to load the dataset
    config:
    '''
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    data_param_dict = {}
    data_config = config["data"]
    dataset = data_config["data_set"]
    data_module = data_config["data_file"]
    target = data_config["target"]
    csv_delimiter = data_config["delimiter"]
    split_method = data_config["split-method"]
    split_fold = data_config["n_splits"]
    additional_parameters = data_config["additional_parameters"]
    for data_param, data_parm_value in additional_parameters.items():
        if data_parm_value != None:
            data_param_dict[data_param] = data_parm_value
    # print(data_param_dict)

    return dataset, data_module, target, split_method, split_fold, data_param_dict, csv_delimiter


def load_eval_metrics(eval_metrics):
    """
    fetch the required evaluation metrics for the experiment from YAML file and dynamically import the modules for the experiment
    :param eval_metrics:
    :return: user_eval_metrics_dict: a dictonary with evaluation metrics name as key and evaluation metrics object as value
    """
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    user_eval_metrics_dict = {}

    for metrics_name, metrics_lib in eval_metrics.items():
        load_metrics_lib = importlib.import_module(metrics_lib)
        load_metrics = getattr(load_metrics_lib, metrics_name)
        user_eval_metrics_dict[metrics_name] = load_metrics
    return user_eval_metrics_dict


def getexperiment(config):
    ''' returns the experiment name from the loaded yaml file '''
    exp_name = config["experiment_name"]
    return exp_name

