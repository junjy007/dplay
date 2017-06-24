"""
Helper function to generate json configuration for experiments.
Usage:
  json_gen.py experiment <xid> [from <pid>]

"""
import json
import dplay_utils.xdeploy as xd
project_path = xd.get_project_path()
conf = {
    'experiment_id': '',
    'parent_experiment_id': '',
    'experience_opts': {
        'capacity': 15000,
        'discount': 0.99
    },

    'encoder_opts': {
        'input_channels': 1,
    }

    'decoder_opts': {
        'input_num': None,
        'fc1_hidden_unit_num': 256,
        'output_num': 4
    }

    'trainer_opts': {'Optimiser': torch.optim.Adagrad, 'learning_rate': 1e-4}

    'path_opts': {
        'BASE_PATH': project_path,
        'RUN_PATH': 'RUNS',
    }
}

running_dir = os.path.join(conf['path_opts']['BASE_PATH'],
                           conf['path_opts']['RUN_PATH'],
                           conf['path_opts']['experiment_id'])
save_dir = os.path.join(running_dir, 'checkpoints')

conf['keeper_opts'] = {
    'train_every_n_episodes': 300,
    'save_every_n_training_steps': 100,
    'draw_every_n_training_steps': -1,
    'max_training_steps': 2000000,
    'save_path': save_dir,
    'report': {'save_checkpoint': True,
               'every_n_steps': 10000,
               'every_n_training': 1,
               'every_n_episodes': 1,
               'every_n_time_records': 100}
}
