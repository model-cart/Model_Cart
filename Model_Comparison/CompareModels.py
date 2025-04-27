'''
compare trained models in MLflow and visualize the evaluation metrics.

Create MLflow experiments: Create a separate experiment in MLflow for each model to track
the metrics, parameters, and artifacts associated with each run.

Train multiple models: Train multiple models and
log the evaluation metrics using MLflow's logging API.


Log evaluation metrics: Use MLflow's logging API to log the evaluation metrics for each
run, including accuracy, precision, recall, and F1 score.

Retrieve metrics with the MLflow API: Use the MLflow API to retrieve the metrics for each
run and store them in a Pandas DataFrame.

Visualize evaluation metrics comparison for different models with Python libraries


retrieve the evaluation metrics for each model run in an experiment using the MLflow API:

import mlflow
import pandas as pd

experiment_id = ... # enter the experiment ID for the desired experiment
runs = mlflow.search_runs(experiment_ids=experiment_id)

# retrieve the metrics for each run
metrics = []
for run_id in runs["run_id"]:
    run = mlflow.get_run(run_id)
    metrics.append(run.data.metrics)

# store the metrics in a Pandas DataFrame
df = pd.DataFrame(metrics)

Create custom visualizations to compare the evaluation metrics for the selected models

allowing user to compare the performance of the different models.

Use this script to creat a mlflow plugin that can be used with any project
'''


import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import plotly.express as px


# load all the experiments
def load_experiments():
    exps = mlflow.search_experiments()
    exp_cnt = 1
    exp_dict = {}
    print("Select Experiment to load:\n")
    for exp_id in exps:
        print(f" {exp_cnt}: Exp. ID - {exp_id.experiment_id} / Exp. Name - {exp_id.name}")
        exp_dict[exp_cnt] = exp_id.experiment_id
        exp_cnt += 1
    user_selected_exp_number = int(input(f"Enter your choice (1 - {exp_cnt - 1}): "))
    print(
        f"You have selected Experiment ID: {exp_dict[user_selected_exp_number]} to load for selecting models to compare")
    user_selected_exp_id = str(exp_dict[user_selected_exp_number])
    runs = mlflow.search_runs(experiment_ids=[user_selected_exp_id])
    # print(f"runs:{runs} ")
    return runs



# take user input to select the evaluation metrics to compare
def select_metrics():
    eval_metrics_dict = {
        1: "accuracy",
        2: "Kappa Value",
        3: "training_accuracy_score",
        4: "training_f1_score",
        5: "training_log_loss",
        6: "training_precision_score",
        7: "training_recall_score",
        8: "training_roc_auc",
        9: "training_score",
        10: "Load all metrics"
    }
    print(f"Select the metrics to compare with other models: \n")
    for key, value in eval_metrics_dict.items():
        print(f"{key}: '{value}'")
    user_metrics_input = []
    flg = 'y'
    while flg == 'y':
        user_input = int(input("\nEnter your choice (1-10): \n"))
        if user_input != 10:
            user_metrics_input.append(eval_metrics_dict[user_input])
            flg = input("\nSelect 'y' to add more, otherwise enter 'n' to end model selection ").lower()
        elif user_input == 10:
            cnt = user_input
            while cnt > 1:
                cnt -= 1
                user_metrics_input.append(eval_metrics_dict[cnt])
            flg = 'n'

    print(f"You have selected: '{user_metrics_input}' metrics to compare models\n")
    return user_metrics_input

# take user input to select models to compare and prepare dataframe for all the selected model
def select_model(runs):
    model_dict = {
        1: 'cbd9cd967e464778be9e5e859a197546',
        2: 'a73b016b3fc3429a807e5fc257ac0a47',
        3: '9c328a35c38e43fe9bd8b2b268e3758e',
        4: 'e0348f48a1e042f4957fb11dbd3ed3e4',
        5: '1358973a53044cbc91bcaa7eccb83a68'
    }

    check_run_id = {'cbd9cd967e464778be9e5e859a197546': 'KNN',
                    'a73b016b3fc3429a807e5fc257ac0a47': 'DT',
                    '9c328a35c38e43fe9bd8b2b268e3758e': 'SVM',
                    'e0348f48a1e042f4957fb11dbd3ed3e4': 'LR',
                    '1358973a53044cbc91bcaa7eccb83a68': 'RFC'}

    user_selected_model = {}

    print("Select the model's to Compare:")
    print("1. K-Nearest Neighbors")
    print("2. Decision Tree")
    print("3. Support Vector Machine")
    print("4. LogisticRegression")
    print("5. RandomForestClassifier")
    print("6. All models mentioned above")
    flg = 'y'
    while flg == 'y':
        user_model_input = int(input("Enter your choice (1-6): "))
        if user_model_input != 6:
            user_selected_model[model_dict[user_model_input]] = check_run_id[model_dict[user_model_input]]
            flg = input("\nSelect 'y' to add more, otherwise enter 'n' to end model selection ").lower()
        elif user_model_input == 6:
            flg = 'n'
            model_int = user_model_input
            while model_int > 1:
                model_int -= 1
                user_selected_model[model_dict[model_int]] = check_run_id[model_dict[model_int]]

    # print(f"User selected Model: {user_selected_model}")
    metrics = []
    # eval_metrics = {}
    for run_id in runs["run_id"]:
        if run_id in user_selected_model:
            run = mlflow.get_run(run_id)
            eval_metrics = run.data.metrics
            eval_metrics['run_id'] = check_run_id[run_id]
            metrics.append(eval_metrics)

    # store the metrics in a Pandas DataFrame
    df = pd.DataFrame(metrics)
    print(f"DataFrame with selected model and metrics: \n{df}")
    return df

# plot the visualisation for the evaluation metrics for each selected
def plot_eval_metrics(df,user_metrics_input):
    flg_cnt = 1
    for metrics in user_metrics_input:
        fig = px.histogram(df, title=f"Compare Evaluation Metrics {flg_cnt}", x="run_id", y=metrics)
        fig.show()
        flg_cnt += 1

# fig = px.histogram(df, title="Compare Evaluation Metrics", x="run_id", y="accuracy")
# fig.show()



# enter the experiment ID for the desired experiment
# experiment_id = ["953448091605061543"]
# runs = mlflow.search_runs(experiment_ids=experiment_id)
#
# print(f"runs:{runs}")

# models_eval = {}
# cnt_flag = 1
# metrics = []
# for run_id in runs["run_id"]:
#     if cnt_flag < 3:
#         run = mlflow.get_run(run_id)
#         model_name = f"{run.to_dictionary()['info']['run_name']}_{cnt_flag}"
#         print(f"\nmodel_name: {model_name}")
#         # metrics.append(run.data.metrics)
#         metrics = run.data.metrics
#         models_eval[model_name] = [metrics]
#         # print(f"\nmetrics: {metrics}")
#         print(f"\nmodels_eval: {models_eval}")
#         cnt_flag += 1
#
#
# df_svc = pd.DataFrame(models_eval["SVC(probability=True)_1"])
# df_lr = pd.DataFrame(models_eval["LogisticRegression()_2"])
#
# print(f"\n DF_LR: {df_lr}, DF_SVC: {df_svc}")
# plt.scatter(df_lr.values.flatten(), df_svc.values.flatten())
# plt.xlabel('df_lr')
# plt.ylabel('df_svc')
#
# diff = df_lr - df_svc
# sns.heatmap(diff, cmap="coolwarm")
# plt.title("Comparison of Model Accuracy")
# plt.show()


# df = pd.DataFrame(metrics)
# print(f"\naccuracy df:\n {df['accuracy']}")
#
# print(f"\nData frame created: \n{df}")
#
# plt.plot(df["accuracy"], label="Model 1")
# plt.plot(df_svc["accuracy"], label="svc")
# plt.plot(df_lr["accuracy"], label="lr")
# plt.plot(models_eval["KNeighborsClassifier()"]["accuracy"], label="KNN")
# plt.plot(df["accuracy"], label="Model 2")
# plt.plot(df["accuracy"], label="Model 3")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Comparison of Model Accuracy")
# plt.legend()
# plt.show()
# model_dict = {
#         1: 'cbd9cd967e464778be9e5e859a197546',
#         2: 'a73b016b3fc3429a807e5fc257ac0a47',
#         3: '9c328a35c38e43fe9bd8b2b268e3758e',
#         4: 'e0348f48a1e042f4957fb11dbd3ed3e4',
#         5: '1358973a53044cbc91bcaa7eccb83a68'
#     }
#
# check_run_id = {'cbd9cd967e464778be9e5e859a197546': 'KNN',
#         'a73b016b3fc3429a807e5fc257ac0a47': 'DT',
#         '9c328a35c38e43fe9bd8b2b268e3758e': 'SVM',
#        'e0348f48a1e042f4957fb11dbd3ed3e4': 'LR',
#         '1358973a53044cbc91bcaa7eccb83a68': 'RFC'}
#
#
# print("Select the model's to Compare:")
# print("1. K-Nearest Neighbors")
# print("2. Decision Tree")
# print("3. Support Vector Machine")
# print("4. LogisticRegression")
# print("5. RandomForestClassifier")
# print("6. All models mentioned above")
#
# user_model_input = int(input("Enter your choice (1-6): "))
# metrics = []
# eval_metrics = {}
# for run_id in runs["run_id"]:
#     if run_id in check_run_id:
#         run = mlflow.get_run(run_id)
#         eval_metrics = run.data.metrics
#         eval_metrics['run_id'] = check_run_id[run_id]
#         metrics.append(eval_metrics)

# print(eval_metrics)

# store the metrics in a Pandas DataFrame
# df = pd.DataFrame(metrics)
# print(df)
#
# fig = px.histogram(df, title="Compare Evaluation Metrics", x="run_id", y="accuracy")
# fig.show()
#
# sns.set_palette("husl")
# sns.catplot(x='run_id', y="accuracy", hue="accuracy", kind="point", data=df)

# Set plot labels
# plt.xlabel("Models")
# plt.ylabel("Evaluation Metrics: Accuracy")
# plt.title("Comparison of Evaluation Metrics for Different Models")
# plt.legend(title="Metrics")

# plt.show()

# plt.plot(df["accuracy"], label="accuracy")
# plt.plot(df["training_score"], label="training_score")
# plt.plot(df["Kappa Value"], label="Kappa Value")


# plt.plot(df["accuracy"], label=df["run_id"])
# plt.xlabel("Models")
# plt.ylabel("Accuracy")
# plt.title("Comparison of Model Accuracy")
# plt.legend()
# plt.show()

# if __name__ == '__main__':
#     print("Checking working")
#     runs = load_experiments()
#     user_metrics_input = select_metrics()
#     df = select_model(runs)
#     plot_eval_metrics(df, user_metrics_input)

