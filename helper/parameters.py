'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-04-01 17:37:01
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-04-09 14:01:00
FilePath: /mru/Few-Shot-Learning/helper/parameters.py
Description: 

'''
import argparse

# --------------------- parameters ---------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('--batch_size', default=100, type=int, help='batch size only for classical training')
    parser.add_argument('--criterion', default='CrossEntropyLoss', choices=['CrossEntropyLoss'], help='criterion')
    parser.add_argument('--epoch_size', default=300, type=int, help='epoch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--mode', default="train", choices=['train', 'test'])
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--num_classes', default=64, type=int, help='num_classes while training')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading threads')
    parser.add_argument('--optimizer', default="adamw", choices=['adam', 'adamw', 'adadelta', 'adagrad', 'rmsprop', 'sgd'])
    parser.add_argument('--schedular', default="ReduceLROnPlateau", choices=['ReduceLROnPlateau'])
    parser.add_argument('--seed', default=777, type=int, help='manual seed')
    parser.add_argument('--val_freq', default=10, type=int, help='validation frequency')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

    # few-shot learning
    parser.add_argument('--backbone', default="resnet18", help='Prototypical network backbone')
    parser.add_argument('--n_query', default=5, type=int)
    parser.add_argument('--n_shot', default=5, type=int)
    parser.add_argument('--n_task_train', default=1536, type=int)
    parser.add_argument('--n_task_val', default=384, type=int)
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--train_mode', default="classical", choices=['episodic', 'classical'], help='episodic: train under sup/qry')
    
    # root
    parser.add_argument('--data_root', default='./release', help='root directory for data')
    parser.add_argument('--fname', default=None, help='log directory name')
    parser.add_argument('--log_root', default='./logs', help='root directory for log')
    parser.add_argument('--log_dir', default='./logs/test', help='root for well-trained model')

    # utils
    parser.add_argument('--cuda', default=True, action='store_false')
    parser.add_argument('--device', default="cuda:0", help='GPU number')
    parser.add_argument('--save_model', default=True, action='store_false')
    
    args = parser.parse_args()
    return args