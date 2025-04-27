import logging
import mlflow
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import tempfile
import os
from Model_Explanation.ModelExplanation import create_shapley
from jinja2 import Environment, FileSystemLoader
from sklearn.model_selection import cross_val_score
from Model_Evaluation.custom_evaluation_methods import eval_model_performance as evmp
from main import db,  Experiment, app, Thread
from sklearn.preprocessing import LabelEncoder


# Recursive function to handle non-serializable items
def make_serializable(item):
    if isinstance(item, pd.Series):
        return item.tolist()
    elif isinstance(item, dict):
        return {k: make_serializable(v) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        return [make_serializable(i) for i in item]
    # Handle any other non-serializable types here
    else:
        return item



def ML_Model(experiment_name, X_train, X_test, y_train, y_test, X, y, exp_data_df, lst_models, lst_algo_param, target,
            lst_eval_metrics, lst_model_description,split_method, split_fold, data_parm_dict, exp_username, yaml_path, exp_thread_id):
    '''
    for the provided ML algorithm to use for classification, trains the algorithm on the training dataset
    and uses MLflow to log the model and all the artifacts for the run.
    '''
    rfc_flag = 'N'
    linear_model_flag = 'N'
    model_report = {}
    experiment_name = f"{experiment_name}_{exp_username}"
    # # set experiment ID
    try:
        # creating a new experiment
        exp_id = mlflow.create_experiment(name=experiment_name)
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    run = 0

    for loaded_model in lst_models:
        # start training models
        model_name = [*loaded_model]
        model_object = [*loaded_model.values()]
        if model_name[0] == 'RandomForestClassifier':
            rfc_flag = 'Y'
        elif model_name[0] == 'LogisticRegression' or model_name[0] == 'LinearRegression':
            linear_model_flag = 'Y'
        print(f"rfc_flg : {rfc_flag} & linear_model_flag : {linear_model_flag}")
        with mlflow.start_run(experiment_id=exp_id, run_name=model_name[0], description=lst_model_description[model_name[0]] ) as exp_run:
            mlflow.set_tag("mlflow.user", exp_username)
            model_report["Experiment Name"] = experiment_name
            model_report["Experiment ID"] = exp_id
            model_report["Thread ID"] = exp_thread_id
            model_report["Run ID"] = exp_run.info.run_id
            model = model_object[0]
            model_report["Model Name"] = str(model_name[0])
            model_report["Model Description"] = lst_model_description[model_name[0]]
            mlflow.log_text(lst_model_description[model_name[0]], "Model_Description.txt")

            with app.app_context():
                try:
                    # Assuming exp_thread_id holds the thread_id of the Thread you want to update
                    thread = Thread.query.filter_by(thread_id=exp_thread_id).first()
                    if thread is not None:
                        thread.experiment_name = experiment_name
                        thread.mlflow_experiment_id = exp_id
                        thread.mlflow_run_id = exp_run.info.run_id
                        thread.model_name = str(model_name[0])
                        db.session.commit()  # Commit the changes to the database
                    else:
                        # Handle the case where the Thread does not exist if necessary
                        pass
                except Exception as e:
                    app.logger.error(f'Error updating Thread: {e}')

            # log the parameters
            mlflow.autolog()

            # log experiment data

            mlflow.log_artifact(yaml_path)
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_file = os.path.join(tmp_dir, 'dataframe.csv')
                exp_data_df.to_csv(csv_file, index=False)
                mlflow.log_artifact(csv_file, 'experiment_data_csv')
            print(lst_algo_param)
            # if lst_algo_param:
            model_report["Parameters"] = lst_algo_param[run][model_name[0]]
            print(f"model_paramters: {lst_algo_param[run][model_name[0]]} ")
            algorithm = model(**lst_algo_param[run][model_name[0]])




            model_report["Data-Parameters"] = data_parm_dict

            # Train the selected algorithm on the training data:
            logging.info("Training Model on Training  DataSet")
            if model_name[0] == 'XGBClassifier':
                encoder = LabelEncoder()
                y_train_encoded = encoder.fit_transform(y_train)
                y_test_encoded = encoder.transform(y_test)
                algorithm.fit(X_train, y_train_encoded)
            else:
                algorithm.fit(X_train, y_train)
            trained_params = algorithm.get_params()
            print(f"model {algorithm} trained with param: {trained_params}")

            if split_method == 'KFold':
                cross_val_scores = cross_val_score(algorithm, X, y, cv=split_fold)
                cross_val_scores_json = json.dumps(cross_val_scores.tolist())
                mlflow.log_text(cross_val_scores_json, "cross_val_score.json")
                msd = "%0.2f accuracy with a standard deviation of %0.2f" % (cross_val_scores.mean(), cross_val_scores.std())
                print(msd)
                mlflow.log_text(msd, "Score Mean & SD.txt")

            mlflow.log_artifact("Experiment_reports/dataset_report.html", "experiment_data_csv")

            model_performance_metrics_dict = evmp(lst_eval_metrics, algorithm, X_test, y_test, run)

            evaluation_dict = {}

            #     Log accuracy of the model
            for key, value in model_performance_metrics_dict.items():

                if key == 'confusion_matrix':
                    # Log confusion matrix
                    confusion_matrix_list = value.tolist()

                    #create a formatted list
                    confusion_matrix_string = '\n'.join(
                        ['\t'.join([str(elem) for elem in row]) for row in confusion_matrix_list])
                    mlflow.log_text(confusion_matrix_string, "confusion matrix.txt")

                elif key == 'classification_report':
                    #Log classification report of the model
                    mlflow.log_text(value, "classification_report.txt")
                else:
                    mlflow.log_metric(key, value)
                    evaluation_dict[key] = value

            # print(f"evaluation_dict: {evaluation_dict}")
            model_report["Performance"] = evaluation_dict

            #     # create the shap explaination
            shap_result_dict = create_shapley(algorithm, X_train, rfc_flag, linear_model_flag, exp_data_df, target)

            shap_result_json = json.dumps(shap_result_dict)
            model_report["SHAP values"] = shap_result_dict
            mlflow.log_text(shap_result_json, "shap_values.json")  # {exp id:shp}

            # log the shap explanation
            mlflow.log_artifact("Experiment_Visualization_Output/explanation_bar.png")

            model_report_serializable = make_serializable(model_report)
            # Serialize the modified dictionary
            model_report_json = json.dumps(model_report_serializable)

            sns.pairplot(data=exp_data_df)
            plt.savefig("Experiment_Visualization_Output/pairplot.png")
            mlflow.log_artifact("Experiment_Visualization_Output/pairplot.png")
            plt.close()

            mlflow.log_text(model_report_json, 'model_report.json')

            # Create a jinja2 environment with the current directory as the template search path
            env = Environment(loader=FileSystemLoader(''))
            # Load the template from a file
            template = env.get_template('Report_Template/report_template.html')

            # Render the template with the model report dictionary as the variables
            html_report = template.render(report=model_report)

            # Write the result to a file
            with open('Experiment_reports/model_report.html', 'w') as f:
                f.write(html_report)

            mlflow.log_artifact("Experiment_reports/model_report.html")

            # Train the selected algorithm on the entire data:
            logging.info("Training Model on Entire DataSet")
            if model_name[0] == 'XGBClassifier':
                y_encoded = encoder.fit_transform(y)
                algorithm.fit(X, y_encoded)
            else:
                algorithm.fit(X, y)

            logging.info("Model Trained on Entire DataSet, ready to be logged as an artifact")
            # Log the model as an artifact
            mlflow.sklearn.log_model(algorithm, "model")
            logging.info(f"Model {model_name[0]} logged as an artifact")

            # Then, create a new record in the database for the Experiment
            with app.app_context():
                new_experiment = Experiment(name=experiment_name, mlflow_experiment_id=exp_id, mlflow_run_id= exp_run.info.run_id, user_id=exp_username)
                db.session.add(new_experiment)
                thread = Thread.query.filter_by(thread_id=exp_thread_id).first()
                thread.status = 'Completed'
                thread.result = 'Experiment completed successfully'
                db.session.commit()
            rfc_flag = 'N'
            linear_model_flag = 'N'
            run += 1
            model_report = {}
