'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-03-10 10:58:15
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-04-01 17:26:57
FilePath: /mru/Few-Shot-Learning/helper/utils.py
Description: 

'''
import os
import yaml
import torch
import shutil
import random
import numpy as np
from datetime import datetime
from termcolor import colored, cprint
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim


# --------------------- set seed ---------------------
def set_seed(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- set device ---------------------
def set_device(cuda, _device):
    if cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = torch.device(_device)
        return device, [_device, torch.cuda.get_device_name()]
    else:
        device = 'cpu'
        return device, [_device]


# --------------------- set log file ---------------------
def log_file(log_root: str, data: dict, fname=None):

    # named file name as datetime
    if fname:
        log_dir = os.path.join(log_root, fname)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_root, timestamp)
    
    # make directory
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    if not os.path.exists(log_root):
        os.mkdir(log_root)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    elif fname in ["testing", "test"] and os.path.isdir(log_dir):
        print("Removing the original test / testing file...")
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)
    else:
        print("Directory already exist, change a file name...")
        os._exit(1)
    
    # save parameters
    with open(os.path.join(log_dir, "args.yaml"), 'w') as f:
        yaml.dump(data, f, Dumper=yaml.CDumper)

    return log_dir


# --------------------- set logger ---------------------
def logger(text, title=False, log_dir=None, mode="train"):

    # stdout
    if title:
        cprint(f"\n=========================== {text} ===========================", "green")
    else:
        cprint(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] \t {text}", "blue")

    # file
    if title:
        with open('{}/{}_record.txt'.format(log_dir, mode), 'a') as train_record:
            train_record.write(f'\n=========================== {text} ===========================\n')
    else:
        with open('{}/{}_record.txt'.format(log_dir, mode), 'a') as train_record:
            train_record.write(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] \t {text}\n")


# --------------------- set optimizer ---------------------
def set_optimizer(opt, model, lr, weight_decay, momentum):

    if opt.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt.lower() == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif opt.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    return optimizer
