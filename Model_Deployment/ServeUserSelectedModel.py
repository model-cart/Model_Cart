import requests


# User Load's Test Dataset
def load_data():
    test_data_list = []
    cont_testing = 'y'
    while cont_testing == 'y':
        print("Enter Test Data to request prediction from the model")
        sepal_length = float(input("Enter sepal_length: "))
        sepal_width = float(input("Enter sepal_width: "))
        petal_length = float(input("Enter petal_length: "))
        petal_width = float(input("Enter petal_width: "))
        data_list = [sepal_length, sepal_width, petal_length, petal_width]
        test_data_list.append(data_list)
        cont_testing = input("Press 'y' to add more test data. Else press any key: ")
    return test_data_list


# User Selects the model to run prediction for Loaded DataSet
def select_model():
    model_dict = {
        1: 'cbd9cd967e464778be9e5e859a197546',
        2: 'a73b016b3fc3429a807e5fc257ac0a47',
        3: '9c328a35c38e43fe9bd8b2b268e3758e',
        4: 'e0348f48a1e042f4957fb11dbd3ed3e4',
        5: '1358973a53044cbc91bcaa7eccb83a68'
    }
    print("Select the model to use for prediction:")
    print("1. K-Nearest Neighbors")
    print("2. Decision Tree")
    print("3. Support Vector Machine")
    print("4. LogisticRegression")
    print("5. RandomForestClassifier")
    user_selected_model = model_dict[int(input("Enter your choice (1-5): "))]

    model_uri = f"mlruns/953448091605061543/{user_selected_model}/artifacts/model"
    print(f"Now serving model: {model_uri} @ 127.0.0.1:8000")
    return model_uri


# request call to model api endpoint with user selected model and dataset for generating prediction
def request_prediction(test_data_list, model_uri):

    # Preparing the dictionary with Test dataset and user selected model for prediction
    test_data_dict = {
        "data": test_data_list,
        "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "model_uri": model_uri
    }

    # make request call to the mlflow model api serving on localhost at port 8000
    response = requests.post("http://127.0.0.1:8000/predict", json=test_data_dict)

    # print response code
    print(response)

    # print response data using json representation
    print(response.json())


if __name__ == '__main__':
    request_prediction(load_data(), select_model())
