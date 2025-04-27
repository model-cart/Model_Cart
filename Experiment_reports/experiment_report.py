import mlflow
import logging


def load_experiments():
    '''
    To provide user a list of Experiment's present in the MLflow to the User
    :return: A list of Experiments present in the mlflow
    '''
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    exps = mlflow.search_experiments()
    exp_cnt = 1
    exp_dict = {}
    print("Select Experiment to load:\n")
    for exp_id in exps:
        print(f" {exp_cnt}: Exp. ID - {exp_id.experiment_id} / Exp. Name - {exp_id.name}")
        exp_dict[exp_cnt] = exp_id.experiment_id
        exp_cnt += 1
    user_selected_exp_number = int(input(f"Enter your choice (1 - {exp_cnt - 1}): "))
    logging.info(
        f"You have selected Experiment ID: {exp_dict[user_selected_exp_number]} to load for selecting models to compare")
    user_selected_exp_id = str(exp_dict[user_selected_exp_number])
    run_id = mlflow.search_runs(experiment_ids=[user_selected_exp_id])

    return run_id

def select_runs(run_id):
    '''
    Provides a list of run id's to the user to select all the required run id to compare their performance and explanation of model output
    :return: a list of all the user selected run id's from the selected experiment to compare their performance and explanation for model output
    '''
    print(run_id)
    pass

def select_performance_metrics_to_compare():
    '''
    provide a list of available performance metrics to select a final list of performance metrics, which will be used to compare the model's
    :return: a list of performance metrics to compare the selected run id's(model's)
    '''
    pass

def compare_models_performance():
    '''
    compare all the selected run id's(model) on the basis of the selected performance metrics
    :return:
    '''
    pass

def visualize_performance_comparison():
    '''
    prepare the visualization for the model's performance comparison
    :return: return visualization for the model's performance comparison
    '''
    pass

def prepare_experiment_report():
    '''
    A final report for the experiment containing all the detailed comparison for the selected model's from the experiment
    :return: A report for the experiment containing all the model performance and explanation comparison
    '''
    pass

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(f"Initiating Experiment Report Process")
    select_runs(load_experiments())
