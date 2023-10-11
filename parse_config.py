import os
import logging
from pathlib import Path
# from functools import reduce, partial
# from operator import getitem
from datetime import datetime
# from logger import setup_logging
from utils.util import read_json, write_json
from typing import Dict



class ConfigParser:
    def __init__(self, config: Dict, resume=None, modification=None, run_id: str=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        """
        # load config file and apply modification
        self._config = config


        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        experiment_name = self.config['name']
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        
        self._save_dir = save_dir / 'models' / experiment_name / run_id
        self._log_dir = save_dir / 'log' / experiment_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # # configure logging module
        # setup_logging(self.log_dir)
        # self.log_levels = {
        #     0: logging.WARNING,
        #     1: logging.INFO,
        #     2: logging.DEBUG
        # }

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)


    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]
        
    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir




# def _get_opt_name(flags):
#     for flg in flags:
#         if flg.startswith('--'):
#             return flg.replace('--', '')
#     return flags[0].replace('--', '')

# def _set_by_path(tree, keys, value):
#     """Set a value in a nested object in tree by sequence of keys."""
#     keys = keys.split(';')
#     _get_by_path(tree, keys[:-1])[keys[-1]] = value

# def _get_by_path(tree, keys):
#     """Access a nested object in tree by sequence of keys."""
#     return reduce(getitem, keys, tree)






