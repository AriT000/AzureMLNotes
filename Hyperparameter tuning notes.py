#Improving your model with hyperparameter tuning


"""
Discrete hyperparameters:
  - these take distinct, separate values (i.e. number of clusters)
  - (qnormal, quniform, qlognormal, qloguniform) <-- this syntax refers to distributions; each one is used to define search spaces

Continuous hyperparameters:
  - these take a range of values (i.e. threshold values)
  - (normal, uniform, lognormal, loguniform)

 * 'q' prefix means quantized, so the data sample values are restricted to certain values

"""


#define search space hyperparameters (batch, learn) and the function (choice, normal)
param_space = {
    '--batch_size': choice(16, 32, 64),
    '--learning_rate': normal(10, 3)
    }


"""
to go thru search space, define sampling
  - Grid Sampling (goes thru all values, only for discrete)
  - Random Sampling (randomly pick values, for both discrete and continuous)
  - Bayesian Sampling (picks based on performance of previous sleection, only for choice, uniform, and quniform)

"""

#define sampling method for the hyperparameters
from azureml.train.hyperdrive import RandomParameterSampling

param_sampling = RandomParameterSampling(param_space)


"""
Early Termination Policy
  - Bandit Policy (stop if underperforming best run)
  - Median Stopping (stop if worse than median of runs)
  - Truncation Selection (stop if worst below certain % of runs)
"""

#terminate the compute early
from azureml.train.hyperdrive import TruncationSelectionPolicy

stop_policy = TruncationSelectionPolicy(evaluation_interval = 1,
                                        truncation_percentage = 20,
                                        delay_evaluation = 5)

#ex)
#Script to execute training, hyperparameter used is '--reg' and there is also the log performance metric
parser.add_argument('--reg', type = float, dest = 'reg_rate')
...
run.log('Accuracy', model_accuracy)

#hyperparameter tuning with ScripRunConfig for training script (script_config), params added to script arguments, and names of logged metrics
hyperdrive = HyperDriveConfig(run_config = script_config,
                              hyperparamter_sampling = param_sampling,
                              policy = stop_policy,
                              primary_metric_name = 'Accuracy',
                              primary_metric_goal = PrimaryMetricGoal.MAXIMIZE,
                              max_total_runs = 6,
                              max_concurrent_runs = 4)

hyperdrive_run = experiment.submit(config = hyperdrive)