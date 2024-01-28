#hyperparameter tuning example

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#connect to workspace
import azureml.core
from azureml.core import Workspace

ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#create dataset for tuning
from azureml.core import Dataset

#get datastore from workspace
default_ds = ws.get_default_datastore()

#upload file
if 'titanic dataset' not in ws.datasets:
    default_ds.upload_files(files=['./data/titanic.csv'],
                        target_path='titanic-data/', 
                        overwrite=True, 
                        show_progress=True)

    tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'titanic-data/titanic.csv'))

#from file, register new dataset
    try:
        tab_data_set = tab_data_set.register(workspace=ws, 
                                name='titanic dataset',
                                description='titanic data',
                                tags = {'format':'CSV'},
                                create_new_version=True)
        print('Dataset registered.')
    except Exception as ex:
        print(ex)
else:
    print('Dataset already registered.')


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#create folder in compute
import os

experiment_folder = 'titanic-hyperdrive'
os.makedirs(experiment_folder, exist_ok=True)

print('Folder ready.')


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#write a new script
%%writefile $experiment_folder/titanic_training.py
#script trains model, calculates accuracy, calculate area under curve (AUC)
# Import libraries
import argparse, joblib, os
from azureml.core import Run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

run = Run.get_context()

parser = argparse.ArgumentParser()
#add hyperparameters to parser
parser.add_argument("--input-data", type=str, dest='input_data', help='training dataset')

# Hyperparameters
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.1, help='learning rate')
parser.add_argument('--n_estimators', type=int, dest='n_estimators', default=100, help='number of estimators')

args = parser.parse_args()

# Log Hyperparameter values
run.log('learning_rate',  np.float(args.learning_rate))
run.log('n_estimators',  np.int(args.n_estimators))

#prepare data
print("Loading Data...")
titanic = run.input_datasets['training_data'].to_pandas_dataframe()

X, y = titanic[['PassengerId','Pclass','Sex','Age','SibSp','Parch', 'Fare', 'Embarked']].values, titanic['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#model with hyperparameters
print('Training a classification model')
model = GradientBoostingClassifier(learning_rate=args.learning_rate,
                                   n_estimators=args.n_estimators).fit(X_train, y_train)

y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/titanic_model.pkl')

run.complete()


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#use cluster so we don't have to use compute yet
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "titanic-cluster"

try:
    # Check for existing compute target
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


%%writefile $experiment_folder/hyperdrive_env.yml
name: batch_environment
dependencies:
- python=3.6.2
- scikit-learn
- pandas
- numpy
- pip
- pip:
  - azureml-defaults


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#run hyperparameter tuning with HyperDriveConfig
from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice
from azureml.widgets import RunDetails

hyper_env = Environment.from_conda_specification("experiment_env", experiment_folder + "/hyperdrive_env.yml")

titanic_ds = ws.datasets.get("titanic dataset")

script_config = ScriptRunConfig(source_directory=experiment_folder,
                                script='titanic_training.py',
                                arguments = ['--input-data', titanic_ds.as_named_input('training_data')],
                                environment=hyper_env,
                                compute_target = training_cluster)

params = GridParameterSampling(
    {
        '--learning_rate': choice(0.01, 0.1, 1.0),
        '--n_estimators' : choice(10, 100)
    }
)

hyperdrive = HyperDriveConfig(run_config=script_config, 
                          hyperparameter_sampling=params, 
                          policy=None,
                          primary_metric_name='AUC',
                          primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
                          max_total_runs=6,
                          max_concurrent_runs=2)

experiment = Experiment(workspace=ws, name='titanic-hyperdrive')
run = experiment.submit(config=hyperdrive)

RunDetails(run).show()
run.wait_for_completion()


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#determine the best run
best_run = run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
script_arguments = best_run.get_details() ['runDefinition']['arguments']
print('Best Run Id: ', best_run.id)
print(' -AUC:', best_run_metrics['AUC'])
print(' -Accuracy:', best_run_metrics['Accuracy'])
print(' -Arguments:',script_arguments)