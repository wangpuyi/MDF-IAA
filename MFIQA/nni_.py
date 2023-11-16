#在2卡上跑
from nni.experiment import Experiment
experiment = Experiment('local')

search_space = {
    'learning_rate': {'_type': 'loguniform', '_value': [0.00001, 0.1]},
    'batch_size': {'_type': 'choice', '_value': [4, 8, 16]},
    'decay_epoch': {'_type': 'choice', '_value': [10, 20, 30, 40, 50]},
    'num_epochs': {'_type': 'choice', '_value': [50, 100, 120, 150, 200, 250, 300]},
}

experiment.config.trial_command = 'python trainnni.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 100
experiment.config.trial_concurrency = 8
experiment.run(8081)
input()