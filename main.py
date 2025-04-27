import logging

from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, send_from_directory, current_app
import mlflow, os, json, zipfile
from urllib.parse import unquote
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import subprocess
import pandas as pd
from scipy.stats import wasserstein_distance
import datetime
from flask_migrate import Migrate
import threading
import uuid
from Serve_Data_server import run_server


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# For the database
DB_PATH = os.path.join(BASE_DIR, 'Project_DB', 'app.db')
app = Flask(__name__, template_folder='template')
app.secret_key = 'key1'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DB_PATH
Session(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
# setup login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# upload path for the user files
UPLOAD_FOLDER = os.path.join(f"{BASE_DIR}/", 'Experiment_repository')
YAML_SAMPLE_FOLDER = os.path.join(f"{BASE_DIR}/", 'Sample_YAML_Template')

ALLOWED_EXTENSIONS = {'yaml', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAMPLE_FOLDER'] = YAML_SAMPLE_FOLDER


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username


class Experiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    mlflow_experiment_id = db.Column(db.String(120), nullable=False)
    mlflow_run_id = db.Column(db.String(120), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    mlflow_run_id_notes = db.Column(db.String(5000), nullable=True)

    user = db.relationship('User', backref=db.backref('experiments', lazy=True))

    def __repr__(self):
        return '<Experiment %r>' % self.name


class Comparison(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    comparison_notes = db.Column(db.String(1000), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow())
    selected_run_ids = db.Column(db.Text, nullable=True)

    user = db.relationship('User', backref=db.backref('comparisons', lazy=True))

    def __repr__(self):
        return f'<Comparison {self.id}>'


class Thread(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.String(128), unique=True, nullable=False)
    status = db.Column(db.String(64), nullable=False, default='Running')
    result = db.Column(db.Text, nullable=True)
    error = db.Column(db.Text, nullable=True)
    experiment_name = db.Column(db.String(120), nullable=False)
    mlflow_experiment_id = db.Column(db.String(120), nullable=False)
    mlflow_run_id = db.Column(db.String(120), nullable=False)
    model_name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow())
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow(),
                           onupdate=datetime.datetime.utcnow())

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('threads', lazy=True))

    def __repr__(self):
        return f'<Thread {self.thread_id}>'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/sample_yaml')
def download_sample_yaml():
    return send_from_directory(directory=YAML_SAMPLE_FOLDER, path="sample.yaml", as_attachment=True)


def run_experiment(yaml_filepath, username, thread_id, thread_app):
    with thread_app.app_context():
        thread = Thread.query.filter_by(thread_id=thread_id).first()
        try:
            subprocess_details = subprocess.run(['python', 'driver.py', '--yaml', yaml_filepath, '--username', username, '--thread', thread_id])
            if subprocess_details.returncode == 0:
                thread.status = 'Completed'
                thread.result = 'Experiment completed successfully'
            else:
                thread.status = 'Failed'
                thread.result = f'Experiment failed with return code: {subprocess_details.returncode}'
        except Exception as e:
            thread.status = 'Error'
            thread.result = str(e)
        db.session.commit()  # Commit the changes to the database


@app.route('/new_experiment', methods=['GET', 'POST'])
@login_required
def upload_files():
    if request.method == 'POST':
        # Check if the post request has the files part
        if 'yaml_file' not in request.files or 'csv_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            # return 'No file part'

        yaml_file = request.files['yaml_file']
        csv_file = request.files['csv_file']

        if yaml_file.filename == '' or csv_file.filename == '':
            flash('No selected file')
            # return 'No selected file'
            return redirect(request.url)

        if yaml_file and allowed_file(yaml_file.filename) and csv_file and allowed_file(csv_file.filename):
            yaml_filename = secure_filename(yaml_file.filename)
            csv_filename = secure_filename(csv_file.filename)

            yaml_filepath = os.path.join(app.config['UPLOAD_FOLDER'], yaml_filename)
            csv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)

            yaml_file.save(yaml_filepath)
            csv_file.save(csv_filepath)

            # Create a unique thread_id using uuid
            unique_thread_id = str(uuid.uuid4())

            # Create a new Thread record
            new_thread = Thread(thread_id=unique_thread_id, status='Running', experiment_name='None',
                                mlflow_experiment_id='None', mlflow_run_id='None', model_name='None', user_id=current_user.username)

            db.session.add(new_thread)
            db.session.commit()

            thread_app = current_app._get_current_object()

            # Launch the driver.py script in a new thread
            thread = threading.Thread(target=run_experiment,
                                      args=(yaml_filepath, current_user.username, new_thread.thread_id, thread_app))
            thread.start()

            flash("Experiment started", 'success')
            return jsonify({'thread_id': new_thread.thread_id})
    return render_template('upload.html')


@app.route('/check_thread')
def check_thread():
    threads = Thread.query.filter_by(user_id=current_user.username).all()
    return render_template('check_thread.html', threads=threads)


@app.route('/thread_status/<thread_id>', methods=['GET'])
def thread_status(thread_id):
    thread = Thread.query.filter_by(thread_id=thread_id, user_id=current_user.username).first()
    if thread:
        return render_template('thread_status.html', thread=thread)
    else:
        return render_template('thread_status.html', status="404", result="Thread not found", error="Thread not found")


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm-password')

            # Check if passwords match
            if password != confirm_password:
                flash('Passwords do not match. Please try again.', 'danger')
                return redirect(url_for('register'))
            # Check if the user already exists
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash('Username already exists. Choose a different one.', 'danger')
                return redirect(url_for('register'))

            user = User(username=username, password=generate_password_hash(password, method='sha256'))
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            # Roll back the session in case of errors
            db.session.rollback()
            flash(f'An error occurred during registration: {e}', 'danger')
            return redirect(url_for('register'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password, password):
            return 'Please check your login details and try again.'
        login_user(user)
        session['logged_in'] = True
        session['username'] = username
        return redirect(url_for('home'))
    return render_template('login.html')


@app.route("/profile")
@login_required
def profile():
    return render_template('profile.html')


@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    logout_user()
    return redirect(url_for('home'))


@app.route("/add_to_cart", methods=["POST"])
@login_required
def add_to_cart():
    selected_runs = request.form.getlist("selected_runs[]")

    # Extract the priorities from the form data
    priorities = request.form.to_dict(flat=False)
    priorities = {k.replace('priorities[', '').replace(']', ''): v[0] for k, v in priorities.items() if 'priorities[' in k}

    # Update the user's cart in the session
    cart_key = f"user_cart_{current_user.id}"
    if cart_key not in session:
        session[cart_key] = []

    for selected_run in selected_runs:
        experiment_id, run_id = selected_run.split('-')

        # Check if run_id is already in the cart
        if any(item["run_id"] == run_id for item in session[cart_key]):
            flash(f"Run ID {run_id} is already in the cart.", "warning")
            return jsonify({"error": f"Run ID {run_id} is already in the cart."}), 400

        # Add the run details along with its priority to the session
        session[cart_key].append({
            "run_id": run_id,
            "experiment_id": experiment_id,
            "priority": priorities.get(run_id, 0)  # Default priority is set to 0 if not found
        })

    print(f"session: {session}")
    flash("Selected runs added to the cart.", "success")
    # return redirect(url_for("view_cart"))
    return jsonify({"success": True})


def get_mlflow_run_notes(run_id):
    """Retrieve the user notes for a given run_id from the Experiment table."""
    notes = ""
    run_id_entry = Experiment.query.filter_by(mlflow_run_id=run_id).first()
    if run_id_entry:
        notes = run_id_entry.mlflow_run_id_notes
    else:
        notes = ""
    return notes


def get_comparison_notes(run_id):
    """Retrieve the comparison notes for a given run_id from the Comparison table."""
    # Convert run_id to string since it will be stored as text in selected_run_ids column
    run_id_str = str(run_id)
    comparisons_with_run_id = Comparison.query.filter(Comparison.selected_run_ids.contains(run_id_str)).all()

    # Extract comparison notes from the returned records
    notes_list = [comparison.comparison_notes for comparison in comparisons_with_run_id]

    return notes_list


@app.route("/cart")
@login_required
def view_cart():
    cart_key = f"user_cart_{current_user.id}"
    raw_cart = session.get(cart_key, [])

    # Convert cart list to a dictionary grouped by experiment_id
    grouped_cart = {}
    grouped_cart_notes = {}
    for item in raw_cart:
        run_id = item['run_id']
        run_notes = get_mlflow_run_notes(run_id)
        comparison_notes_list = get_comparison_notes(run_id)

        run_info = {
            'run_id': run_id,
            'run_priority': item.get('priority', 0),
            'notes': run_notes,

        }

        if item['experiment_id'] not in grouped_cart:
            grouped_cart[item['experiment_id']] = []
            grouped_cart_notes[item['experiment_id']] = []

        grouped_cart[item['experiment_id']].append(run_info)
        grouped_cart_notes[item['experiment_id']].append(comparison_notes_list)

    print(f"cart session in view: {session}")
    return render_template("cart.html", cart=grouped_cart, grouped_cart_notes=grouped_cart_notes)


@app.route("/remove_from_cart", methods=["POST"])
@login_required
def remove_from_cart():
    run_id = request.form.get('run_id')
    cart_key = f"user_cart_{current_user.id}"
    cart = session.get(cart_key, [])
    # Create a new list with only the dictionaries that don't have the specified run_id
    new_cart = [item for item in cart if item.get('run_id') != run_id]
    session[cart_key] = new_cart

    return redirect(url_for("view_cart"))


@app.route("/clear_cart")
@login_required
def clear_cart():
    # Clear the 'cart' list in session for the specific user
    cart_key = f"user_cart_{current_user.id}"
    session.pop(cart_key, None)

    return redirect(url_for("view_cart"))


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/add_to_comparison", methods=["POST"])
@login_required
def add_to_comparison():
    run_id = request.form.get("run_id")
    run_name = request.form.get("run_name")
    experiment_id = request.form.get("experiment_id")

    # Update the user's comparison list in the session
    comparison_key = f"user_comparison_{current_user.id}"  # Using the user's id to identify the comparison list

    if comparison_key not in session:
        session[comparison_key] = []

    # Check if run_id is already in the comparison list
    for item in session[comparison_key]:
        if item["run_id"] == run_id:
            flash("Run Id is already present in the Comparison list", "warning")
            return redirect(url_for("run_details", run_id=run_id))

    # If not present, add the run_id to the comparison list
    session[comparison_key].append({"run_id": run_id, "run_name": run_name, "experiment_id": experiment_id})

    print(f"session: {session}")

    flash("Run Id added to the Comparison list", "success")
    return redirect(url_for("run_details", run_id=run_id))


@app.route("/saved_comparison")
@login_required
def view_saved_comparison():
    # Fetch the saved comparisons for the current user from the database
    saved_comparisons = Comparison.query.filter_by(user_id=current_user.id).all()

    comparison_saved_dict = {}
    for comp in saved_comparisons:
        run_ids = comp.selected_run_ids.split(',')
        comparison_info = {
            'notes': comp.comparison_notes,
            'runs': run_ids,
            'timestamp': comp.timestamp,
            'id': comp.id
        }

        comparison_time = comp.timestamp
        if comparison_time in comparison_saved_dict:
            comparison_saved_dict[comparison_time].append(comparison_info)
        else:
            comparison_saved_dict[comparison_time] = [comparison_info]

    return render_template("saved_comparison.html", comparison_saved=comparison_saved_dict)


@app.route("/comparison")
@login_required
def view_comparison():
    # Fetch the comparison list contents from the session for the specific user
    comparison_key = f"user_comparison_{current_user.id}"
    comparison_list = session.get(comparison_key, [])

    comparison_dict = {}
    for item in comparison_list:
        experiment_id = item.get('experiment_id')
        if experiment_id in comparison_dict:
            comparison_dict[experiment_id].append(item)
        else:
            comparison_dict[experiment_id] = [item]

    # Fetch the saved comparisons for the current user from the database
    saved_comparisons = Comparison.query.filter_by(user_id=current_user.id).all()

    comparison_saved_dict = {}
    for comp in saved_comparisons:
        run_ids = comp.selected_run_ids.split(',')
        comparison_info = {
            'notes': comp.comparison_notes,
            'runs': run_ids,
            'timestamp': comp.timestamp,
            'id': comp.id
        }

        comparison_time = comp.timestamp
        if comparison_time in comparison_saved_dict:
            comparison_saved_dict[comparison_time].append(comparison_info)
        else:
            comparison_saved_dict[comparison_time] = [comparison_info]

    return render_template("comparison.html", comparison=comparison_dict, comparison_saved=comparison_saved_dict)


def scipy_wasserstein_distance(p, q):
    """
    given feature vvectors p and q,
    return their wasserstein_distance as a measure of the distance between them
    """
    return wasserstein_distance(range(len(p)), range(len(q)), u_weights=p, v_weights=q)


@app.route("/compare_runs", methods=["GET", "POST"])
@login_required
def compare_runs():
    try:
        if request.method == "POST":
            selected_runs = request.form.getlist('selected_runs[]')
            print(f"selected_run_ids from comparison page {selected_runs}")
            selected_run_ids = [run.split('-')[1] for run in selected_runs]
            comp = ""
        else:
            selected_run_ids = request.args.getlist('selected_run_ids')
            comparison_id = request.args.getlist('comparison_id')
            comp = Comparison.query.get(comparison_id)

        # selected_runs = request.form.getlist('selected_runs[]')

        all_runs_data = {}
        shap_values_list = []
        metrics_comparison = {}
        run_mapping = {}
        model_explanations = []
        run_notes_dict = {}
        run_duration_dict = {}

        for run_id in selected_run_ids:
            run = mlflow.get_run(run_id)
            run_starttime_str, run_endtime_str, run_duration_in_seconds = calc_run_duration(run.info.start_time,
                                                                                            run.info.end_time)
            run_experiment_name = Experiment.query.filter_by(mlflow_run_id=run_id).first().name
            run_experiment_id = run.info.experiment_id

            run_info = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run_starttime_str,
                "end_time": run_endtime_str,
                "duration": run_duration_in_seconds
            }
            run_mapping[run.info.run_id] = run.info.run_name
            run_params = run.data.params
            run_metrics = run.data.metrics
            run_duration_dict[f"{run.info.run_name}-{run.info.run_id}"] = run_duration_in_seconds

            run_notes_dict[run_id] = Experiment.query.filter_by(mlflow_run_id=run_id).first().mlflow_run_id_notes

            # Store metrics for visualization
            for metric_name, metric_value in run_metrics.items():
                if metric_name not in metrics_comparison:
                    metrics_comparison[metric_name] = []

                metrics_comparison[metric_name].append({
                    'value': round(metric_value, 3),
                    'run_id': run_id
                })

            artifact_uri = run.info.artifact_uri
            print(f"artifact_uri: {run.info.artifact_uri}")

            # Verify the existence of the directory
            local_path = unquote(artifact_uri).replace(f"file://{BASE_DIR}/", "")
            if os.path.exists(local_path):
                for root, dirs, files in os.walk(local_path):
                    for file in files:
                        if file == "shap_values.json":  # for all model's SHAP values are stored in this file
                            with open(os.path.join(root, file), "r") as f:
                                shap_values = json.load(f)
                                shap_values_list.append({
                                    'values': shap_values,
                                    'run_id': run_id
                                })
                                # Calculate feature attributions,
                                feat_att = pd.DataFrame(
                                    [v for k, v in shap_values.items()],
                                    index=list(shap_values.keys())
                                )
                                feat_att.columns = [run_id]
                                model_explanations.append(feat_att)

            all_runs_data[run_id] = {
                "info": run_info,
                "params": run_params,
                "metrics": run_metrics
            }

        print(f"Preparing the model_explanations list for Wasserstein Distance {model_explanations}")
        # Combine the feature attributions into a dataframe
        model_explanations_df = pd.concat(model_explanations, axis=1)
        print(f"Preparing the model_explanations DataFrame for Wasserstein Distance {model_explanations_df}")

        model_explanations_df = model_explanations_df.div(model_explanations_df.max(0), axis='columns')

        # Initialize the explanation matrix
        model_explanations_matrix = pd.crosstab(selected_run_ids, selected_run_ids).applymap(lambda x: None)
        print(f"Preparing the model_explanations_matrix matrix for Wasserstein Distance {model_explanations_matrix}")

        # Compute the Wasserstein distance between feature attributions and store in the matrix
        for i in selected_run_ids:
            for j in selected_run_ids:
                # model_explanations_matrix.loc[i, j] = wasserstein_distance(
                #     model_explanations_df[i], model_explanations_df[j]
                model_explanations_matrix.loc[i, j] = scipy_wasserstein_distance(
                    model_explanations_df[i], model_explanations_df[j]
                )

        # Convert the DataFrame to a dictionary for creating JSON serializable object
        model_explanations_dict = model_explanations_df.to_dict(orient='index')
        model_explanations_matrix_dict = model_explanations_matrix.to_dict(orient='index')

        print(f"selected_run_ids: {selected_run_ids}")
        print(f"run_mapping: {all_runs_data}")

        # First, find the union of all keys used across all models
        all_keys = set()
        for run in shap_values_list:
            all_keys.update(run['values'].keys())

        # Sort the keys to have a consistent order
        sorted_keys = sorted(all_keys)

        # Now, rearrange each model's values to match the sorted order of keys
        standardized_shap_values_list = []
        for run in shap_values_list:
            values = run['values']
            standardized_values = {key: values.get(key, 0) for key in sorted_keys}
            standardized_shap_values_list.append({
                'values': standardized_values,
                'run_id': run['run_id']
            })

    except Exception as e:

        print(f"An error occurred: {e}")


        flash("An unexpected error occurred. Please try again.", "danger")


        return redirect(url_for("view_comparison"))

    return render_template("compare_page.html", all_runs_data=all_runs_data,
                           metrics_comparison=metrics_comparison, run_shap_values_list=standardized_shap_values_list,
                           model_explanations_df=model_explanations_dict,
                           model_explanations_matrix=model_explanations_matrix_dict, run_mapping=run_mapping,
                           selected_run_ids=selected_run_ids, comp=comp, run_notes_dict=run_notes_dict,
                           run_duration_dict=run_duration_dict, run_experiment_name=run_experiment_name,
                           run_experiment_id=run_experiment_id)


@app.route("/remove_from_comparison", methods=["POST"])
@login_required
def remove_from_comparison():
    # Capture the selected_runs data
    selected_runs = request.form.getlist("selected_runs[]")
    comparison_key = f"user_comparison_{current_user.id}"
    comparison = session.get(comparison_key, [])

    # Extract only the run_ids from the selected_runs data
    selected_run_ids = [run.split('-')[1] for run in selected_runs]

    # Create a new list excluding the selected run_ids
    new_comparison = [item for item in comparison if item.get('run_id') not in selected_run_ids]

    session[comparison_key] = new_comparison
    flash('Experiment removed from Comparison List successfully!', 'danger')

    return redirect(url_for("view_comparison"))


@app.route("/clear_comparison", methods=['POST'])
@login_required
def clear_comparison():
    comparison_key = f"user_comparison_{current_user.id}"
    session.pop(comparison_key, None)
    flash('Comparison List cleared successfully!', 'danger')
    return redirect(url_for("view_comparison"))


@app.route("/save_comparison", methods=["POST"])
@login_required
def save_comparison():
    notes = request.form.get('notes')
    selected_run_ids = request.form.getlist('selected_run_ids[]')

    # Convert list of run IDs to a comma-separated string
    run_ids_str = ','.join(selected_run_ids)

    # Save the comparison to the database
    new_comparison = Comparison(
        user_id=current_user.id,
        comparison_notes=notes,
        selected_run_ids=run_ids_str
    )
    db.session.add(new_comparison)
    db.session.commit()
    comparison_id = new_comparison.id

    flash('Comparison saved successfully!', 'success')
    return redirect(url_for("compare_runs", selected_run_ids=selected_run_ids, comparison_id=comparison_id ))


@app.route("/delete_comparison/<int:comp_id>", methods=["POST"])
@login_required
def delete_comparison(comp_id):
    comparison = Comparison.query.get_or_404(comp_id)
    if comparison.user_id == current_user.id:
        db.session.delete(comparison)
        db.session.commit()
        flash('Comparison deleted successfully!', 'danger')
    else:
        flash('Unauthorized action.', 'danger')
    return redirect(url_for('view_saved_comparison'))


@app.route("/experiment")
@login_required
def experiment():
    # Fetch only the experiments associated with the logged-in user
    experiments = Experiment.query.filter_by(user_id=current_user.username).all()
    user_experiments = {}
    for experiment in experiments:

        user_experiments[experiment.mlflow_experiment_id] = experiment.name

    return render_template('experiments.html', experiments=user_experiments)


@app.route("/<experiment_id>")
def runs(experiment_id):
    try:
        # Get all run IDs for the selected experiment
        run_id = mlflow.search_runs(experiment_ids=experiment_id)
        # Check if the result is empty or not found
        if run_id.empty:
            return render_template("error.html", error_message=f"No Experiment found for this experiment ID: {experiment_id}"), 404
        else:
            return render_template("runs.html", runs=run_id)
    except Exception as e:
        # Handle exceptions (like experiment_id not found)
        return render_template("error.html", error_message=f"An error occurred: {str(e)}"), 500


def calc_run_duration(start_time,end_time):
    run_endtime = datetime.datetime.utcfromtimestamp(end_time/1000)
    run_starttime = datetime.datetime.utcfromtimestamp(start_time/1000)
    run_duration = run_endtime - run_starttime
    run_duration_in_seconds = run_duration.total_seconds()
    run_starttime_str = run_starttime.strftime('%Y-%m-%d %H:%M:%S')
    run_endtime_str = run_endtime.strftime('%Y-%m-%d %H:%M:%S')
    return run_starttime_str, run_endtime_str, run_duration_in_seconds


@app.route("/run_details", methods=['GET', 'POST'])
def run_details():
    run_id = request.args.get('run_id')
    if not run_id:
        return render_template("error.html", error_message=f"Run ID: {run_id} not in system"), 404

    try:
        run = mlflow.get_run(run_id)
    except Exception as e:
        # Handle the case where run_id is invalid or the run ID does not exist
        return render_template("error.html", error_message=f"Run details not found for Run ID: {run_id}"), 404

    run_description = run.data.tags['mlflow.note.content']
    experiment_entry = Experiment.query.filter_by(mlflow_run_id=run_id).first()
    if request.method == 'POST':
        notes = request.form.get('notes')
        if experiment_entry:
            experiment_entry.mlflow_run_id_notes = notes
            db.session.commit()
        else:
            new_entry = Experiment(mlflow_run_id=run_id, mlflow_run_id_notes=notes)
            db.session.add(new_entry)
            db.session.commit()
        flash("Notes Saved Successfully", "success")

    run_info = run.info
    run_starttime_str, run_endtime_str, run_duration_in_seconds = calc_run_duration(run.info.start_time, run.info.end_time)

    run_notes = experiment_entry.mlflow_run_id_notes if experiment_entry else None
    print(f"run_notes: {run_notes}")
    run_params = run.data.params
    run_metrics = run.data.metrics
    run_metrics = {key: value for key, value in run_metrics.items() if not key.startswith("training")}
    artifact_uri = run.info.artifact_uri
    artifacts = {}

    # Verify the existence of the directory
    if not os.path.exists(artifact_uri):
        # print(f"Directory does not exist: {artifact_uri}")
        # Convert artifact_uri to local file path
        local_path = unquote(artifact_uri).replace(f"file://{BASE_DIR}/", "")

    for root, dirs, files in os.walk(local_path):
        print(f"root: {root} \n dirs: {dirs} \n files: {files}")
        for file in files:
            artifacts[file] = f" http://localhost:8080/{root}/{file}"

    cart = session.get("cart", [])

    # Fetch all run_ids for the experiment associated with the current run
    runs_df = mlflow.search_runs(experiment_ids=run.info.experiment_id)
    all_run_ids = runs_df["run_id"].tolist() if not runs_df.empty else []

    try:
        current_index = all_run_ids.index(run_id)
    except ValueError:
        current_index = -1

    next_run_id = all_run_ids[current_index + 1] if current_index < len(all_run_ids) - 1 else None
    prev_run_id = all_run_ids[current_index - 1] if current_index > 0 else None

    return render_template("run_details.html", run_info=run_info, run_params=run_params,
                           run_metrics=run_metrics, artifacts=artifacts, cart=cart, run_description=run_description,
                           run_notes=run_notes, next_run_id=next_run_id, prev_run_id=prev_run_id,
                           run_endtime_str=run_endtime_str, run_starttime_str=run_starttime_str,
                           run_duration_in_seconds=run_duration_in_seconds)


@app.route("/deploy/<run_id>")
@login_required
def deploy(run_id):
    run = mlflow.get_run(run_id)
    artifact_uri = run.info.artifact_uri

    # Verify the existence of the directory
    if not os.path.exists(artifact_uri):
        print(f"Directory does not exist: {artifact_uri}")
        # Convert artifact_uri to local file path
        local_path = unquote(artifact_uri).replace(f"file://{BASE_DIR}/", "")
    else:
        local_path = artifact_uri

    # Create a zip file from the files in local_path
    zip_filename = f"{run.data.tags['mlflow.runName']}_{run_id}_{run.data.tags['mlflow.user']}_{datetime.datetime.now()}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.basename(file_path))

    # Send the compressed file for download
    return send_from_directory(directory=os.getcwd(), path=zip_filename , as_attachment=True)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(f"Initiating Model-Cart UI & Data Server Process")
    with app.app_context():
        db.create_all()
    # app.run(ssl_context=('cert.pem', 'key.pem'),debug=True, port=5000)
    try:
        thread_data_server = threading.Thread(target=run_server)
        thread_data_server.daemon = True  # the thread will close when the main program exits
        thread_data_server.start()
        app.run(ssl_context=('localhost.pem', 'localhost-key.pem'), debug=True, port=5000)
    except Exception as e:
        logging.error("Model-Cart UI & Data Server Didn't started as expected")

