import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

import importlib
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str)
parser.add_argument('--config', type=str, default='configs.pyr')
parser.add_argument('--config_skirt', type=str, default='configs.pyr_skirt')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--iteration', type=str, default='')
parser.add_argument('-g', '--gpu', type=str)
args = vars(parser.parse_args())

config_module = importlib.import_module(args['config'])
config_skirt_module = importlib.import_module(args['config_skirt'])
config = config_module.config
config_skirt = config_skirt_module.config

import argparse
import numpy as np
from utils.utils import Frame


def get_error():

    errors = dict()
    errors_mean = dict()

    with open(f'{config["out_dir"]}/{config["name"]}.csv', 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.split(',')
        category = line[2]
        if category not in errors.keys():
            errors[category] = []
        errors[category].append(float(line[3]))


    with open(f'{config_skirt["out_dir"]}/{config_skirt["name"]}.csv', 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.split(',')
        category = line[2]
        if category not in errors.keys():
            errors[category] = []
        errors[category].append(float(line[3]))

    for category in errors:
        errors_mean[category] = round(np.mean(errors[category]), 5)
    print(errors_mean) 

    mean = round(np.mean(list(errors_mean.values())), 5)
    print(f'Final error: {mean}')

if __name__ == '__main__':
    get_error()