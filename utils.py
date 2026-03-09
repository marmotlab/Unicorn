import os
import math
import json
import torch
import pickle
import pandas as pd
import scipy.signal as signal

from parameters import *


def save_as_json(file, data):
    """
    Save data to the json file
    @ params: file: the json file that will be saved to
    @ params: data: data that need to be saved
    """
    data = json.dumps(data)
    with open(file, 'w+') as file:
        file.write(data)
    file.close()


def load_json(file):
    """
    load saved data from json file
    @ params: file: the file that will be loaded
    """
    with open(file, 'r+') as file:
        content = file.read()

    return json.loads(content)


def make_gif(env, f_name):
    command = 'ffmpeg -framerate 5 -i "{tempGifFolder}/step%03d.png" {outputFile}'.format(tempGifFolder=env.gif_folder,
                                                                                          outputFile=f_name)

    os.system(command)

    deleteTempImages = "rm {tempGifFolder}/*".format(tempGifFolder=env.gif_folder)
    os.system(deleteTempImages)
    print("wrote gif")


def check_dir(dirs):
    """
    Check if the dir exists, otherwise create it
    @param dirs:
    @return:
    """
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def create_dirs(params):
    """
    Create all experiment directories
    @param params: class
    @return:
    """
    for k, v in params.__dict__.items():
        if 'PATH' in k.split('_') and len(k.split('_')) == 2:
            check_dir(dirs=v)


def create_config_dict():
    """
    Create config dict to save all params
    @return: dictionary
    """
    params_list = [EXPERIMENT_PARAMS, TRAIN_PARAMS, NETWORK_PARAMS, SUMO_PARAMS]

    config_dict = {}
    for params in params_list:
        params_dict = vars(params)
        for k, v in params_dict.items():
            if not k.startswith('__'):
                config_dict[k] = v

    return config_dict


def create_config_json(path, params):
    """
    create a txt file to record the config dictionary
    @param path: string
    @param params: dictionary, key/param name, value/param value
    """
    file = open(path, 'w')
    for k, v in params.items():
        file.write('{}: {}\n'.format(k, v))
    file.close()


def save_as_csv(file, data):
    """
    save the input data to a csv file
    @param file: file path
    @param data: input data
    @return:
    """
    data = pd.DataFrame(data)
    data.to_csv(file)


def save_as_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2

    return math.sqrt(sum_grad)


def convert_to_item(tensor):
    return tensor.cpu().detach().numpy().item()


def convert_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def convert_to_tensor(data, data_type, device):
    if device is None:
        return torch.as_tensor(data=data,
                               dtype=data_type)
    else:
        return torch.as_tensor(data=data,
                               dtype=data_type,
                               device=device)


def calculate_discount_return(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def set_env(server_number, test=False):
    from env.matsc import MATSC_Env
    return MATSC_Env(server_number, test)
