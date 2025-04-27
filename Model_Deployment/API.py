from flask import Flask, request, jsonify
import mlflow.pyfunc

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    # Get the input data from the request
    data = request.json['data']
    model_uri = request.json['model_uri']
    columns = request.json['columns']

    # Load the user selected model from MLflow
    model = mlflow.pyfunc.load_model(model_uri, suppress_warnings=False)

    # Make predictions on the input data
    predictions = model.predict(data)

    # Return the predictions as JSON
    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(port=8000, debug=True)
