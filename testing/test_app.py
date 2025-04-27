import pytest
from flask_login import login_user
from io import BytesIO
import sys
import os


# Adjusting the system path so the main directory is recognized
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from main import app, db  # Importing the app instance directly
from main import Thread, User


# the path to test database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DB_PATH = os.path.join(BASE_DIR,'Project_DB', 'test_app.db')


# Setup for testing using the existing app instance
@pytest.fixture
def test_client():
    # Configure the app to use the test database
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{TEST_DB_PATH}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Create the test database and tables
    with app.app_context():
        db.create_all()
        # Create a test user if it doesn't exist
        if not User.query.filter_by(username='testuser').first():
            test_user = User(username='testuser', password='testpassword')
            db.session.add(test_user)
            db.session.commit()

        yield app.test_client()

        # Clean up: drop all tables and remove session
        # db.drop_all()
        # db.session.remove()


@pytest.fixture
def logged_in_client(test_client):
    # Using test_request_context for login to simulate request context
    with app.test_request_context():
        login_user(User.query.filter_by(username='testuser').first())
        yield test_client


def test_new_experiment(logged_in_client):

    data = {
        'yaml_file': (BytesIO(b'yaml content'), 'experiment_wine_xgboost.yaml'),
        'csv_file': (BytesIO(b'csv content'), 'winequality-red.csv')
    }
    response = logged_in_client.post('/new_experiment', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert Thread.query.count() == 9

def test_check_thread(logged_in_client):
    # GET request to the check_thread endpoint
    response = logged_in_client.get('/check_thread')
    assert response.status_code == 200


def test_thread_status(logged_in_client):
    # checking first test case: create a thread with a known thread_id to check the testing
    thread = Thread(
        thread_id='test_thread_id',
        user_id=User.query.filter_by(username='testuser').first().username,
        status='Running',
        experiment_name='Dummy Experiment',
        mlflow_experiment_id='Dummy Experiment ID',
        mlflow_run_id='Dummy Run ID',
        model_name='Dummy Model'
    )
    db.session.add(thread)
    db.session.commit()

    # GET request to the thread_status endpoint for the created thread
    response = logged_in_client.get('/thread_status/test_thread_id')
    assert response.status_code == 200


# Testing /experiment Endpoint
def test_experiment_list_fetching(logged_in_client):
    response = logged_in_client.get('/experiment')
    assert response.status_code == 200


# Testing runs Endpoint
def test_fetching_runs_for_experiment(logged_in_client):
    response = logged_in_client.get(f'/None')
    assert response.status_code == 200


def test_invalid_experiment_id(logged_in_client):
    response = logged_in_client.get('/999999975956917229')
    assert response.status_code == 404


# Testing /run_details Endpoint
def test_run_details_invalid_run_id(logged_in_client):
    run_id = '3f69d3240e29411ea8e56028317a2XYZ'
    response = logged_in_client.get(f'/run_details?run_id={run_id}')
    assert response.status_code == 404


def test_run_details_valid_run_id(logged_in_client):
    run_id = 'Dummy Run ID'
    response = logged_in_client.get(f'/run_details?run_id={run_id}')
    assert response.status_code == 200


def test_run_details_submit_notes(logged_in_client):
    notes = {'notes': 'Test note content'}
    response = logged_in_client.post('/run_details?run_id=Dummy Run ID', data=notes)
    assert response.status_code == 200
    # Verify that notes are saved appropriately

# Additional test cases

