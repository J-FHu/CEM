import sys
import argparse
import os
import torch
import pickle


class BaseConfigures():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--base_dir', type=str, default='./nEXPs', help='base experiment dir')
        parser.add_argument('--name', type=str, default='SR', help='name of the experiment. It also decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./nEXPs/CARN', help='models are saved here')

        # build model
        parser.add_argument('--model_name', type=str, default='CARN', help='base experiment dir')
        parser.add_argument('--sr_factor', type=int, default=4, help='SR factor')
        parser.add_argument('--lr_channels', type=int, default=3, help='The channels of the input low-resolution images')
        parser.add_argument('--load_from_opt_file', type=int, default=0, help='enable training with an image encoder.')


        self.initialized = True
        self.isTrain = False
        return parser

    def gather_configurations(self):
        # initialize parser with basic configurations
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        # get the basic configurations
        config, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default configurations will be overwritten
        if config.load_from_opt_file:  # TODO: config_file
            parser = self.update_configurations_from_file(parser, config)

        config = parser.parse_args()
        self.parser = parser
        return config

    def print_configurations(self, config):
        message = ''
        message += '----------------- Configurations ---------------\n'
        for k, v in sorted(vars(config).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '------------------------ End -------------------'
        print(message)

    def configuration_file_path(self, config, makedir=False):
        expr_dir = os.path.join(config.checkpoints_dir, config.name)
        if makedir:
            os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'config')
        return file_name

    def save_configurations(self, config):
        file_name = self.configuration_file_path(config, makedir=True)
        with open(file_name + '.txt', 'wt') as config_file:
            for k, v in sorted(vars(config).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                config_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as config_file:
            pickle.dump(config, config_file)

    def update_configurations_from_file(self, parser, config):
        new_config = self.load_configurations(config)
        for k, v in sorted(vars(config).items()):
            if hasattr(new_config, k) and v != getattr(new_config, k):
                new_val = getattr(new_config, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_configurations(self, config):
        file_name = self.configuration_file_path(config, makedir=False)
        new_config = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_config

    def parse(self):
        config = self.gather_configurations()
        config.isTrain = self.isTrain   # train or test

        # Print for debugging
        self.print_configurations(config)
        if config.isTrain:
            self.save_configurations(config)

        # set gpu ids
        str_ids = config.gpu_ids.split(',')
        config.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                config.gpu_ids.append(id)
        if len(config.gpu_ids) > 0:
            torch.cuda.set_device(config.gpu_ids[0])

        assert len(config.gpu_ids) == 0 or config.batch_size % (len(config.gpu_ids) * 2) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (config.batch_size, len(config.gpu_ids))

        self.config = config
        return self.config