'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-03-24 16:34:50
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-04-09 13:43:10
FilePath: /mru/Few-Shot-Learning/main.py
Description: 

TODO:
    add show()
    try different backbone
    prototypical distance
    set_schedular
    set accumulate loss backword
    
'''
# package
import os
import random
import argparse
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# other .py
from helper.utils import log_file, logger, set_seed, set_device, set_optimizer
from helper.parameters import parse_args
from train.train import train_episodic, train_classical, evaluate, predict
from train.models import resnet18, resnet34, resnet50, PrototypicalNetworks, set_resnet
from train.dataloader import Fewshot, Fewshot_batch_sampler, set_transform, set_episodic_data_loader, set_classical_data_loader


def main():

    # --------------------- Parameter ---------------------
    args = parse_args()


    # --------------------- log ---------------------
    if args.mode == "train":
        log_dir = log_file(args.log_root, args.__dict__, args.fname)
    elif args.mode == "test":
        assert args.log_dir != None, "You must give the log_dir if you want to implement prediction"
        assert os.path.isdir(args.log_dir), " No such file or directory: {}".format(args.log_dir)
        log_dir = args.log_dir


    # --------------------- Device ---------------------
    device, info = set_device(args.cuda, args.device)
    logger(f"Device: {', '.join(info)}", title=True, log_dir=log_dir, mode=args.mode)
    

    # --------------------- Random seed ---------------------
    set_seed(args.seed)
    logger(f"Random Seed: {args.seed}", title=True, log_dir=log_dir, mode=args.mode)


    # --------------------- load train dataset & model & optimizers & criterion & tensorboard --------------------- 
    if args.mode == "train":
        
        # dataset
        transform = set_transform()

        if args.train_mode == "episodic":
            train_dataset, _, train_loader = set_episodic_data_loader(args.data_root, transform, "train", args.n_way, args.n_shot, args.n_query, args.n_task_train, args.num_workers)
        elif args.train_mode == "classical":
            train_dataset, train_loader = set_classical_data_loader(args.data_root, transform, args.batch_size, args.num_workers)
        val_dataset, _, val_loader = set_episodic_data_loader(args.data_root, transform, "validation", args.n_way, args.n_shot, args.n_query, args.n_task_val, args.num_workers)

        # initialize PrototypicalNetworks model
        assert len(set(train_dataset.get_labels())) == args.num_classes, "remember to modify args.num_classes, this value must be the same as the training set."
        backbone = set_resnet(args.backbone, len(set(train_dataset.get_labels())), device)
        
        backbone.set_use_fc(True) if args.train_mode == "classical" else backbone.set_use_fc(False)
        model = PrototypicalNetworks(backbone).to(device)

        # optimizers (note that while implementing classical training, optimizing backbone is the same as optimizing model)
        optimizer = set_optimizer(args.optimizer, model, args.lr, args.weight_decay, args.momentum)

        # lr scheduler
        if args.schedular.lower() == 'ReduceLROnPlateau'.lower():
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)
    
        # criterion
        criterion = nn.CrossEntropyLoss()
    
        # tensorboard
        writer = SummaryWriter(os.path.join(log_dir, "tb_record"))
    

        # --------------------- training loop ---------------------

        # logger
        logger("Training", title=True, log_dir=log_dir, mode=args.mode)
        
        max_accuracy = 0

        for epoch in range(1, args.epoch_size+1):

            # train
            if args.train_mode == "episodic":
                epoch_loss, accuracy = train_episodic(model, epoch, args.epoch_size, train_loader, optimizer, criterion, device)
            elif args.train_mode == "classical":
                epoch_loss, accuracy = train_classical(backbone, epoch, args.epoch_size, train_loader, optimizer, criterion, device)

            # lr scheduler
            scheduler.step(epoch_loss)
            
            # logger
            logger('epoch : {:03d}/{} \t Epoch loss:{:5.6f} \t Accuracy: {:.4f}'\
                    .format(epoch, args.epoch_size, epoch_loss, accuracy),\
                    title=False, log_dir=log_dir, mode=args.mode)

            # tb record
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Train/loss', epoch_loss, epoch)
            writer.add_scalar('Train/accuracy', accuracy, epoch)


            # --------------------- validation ---------------------
            if epoch % args.val_freq == 0 or epoch == 1:
                
                # evaluate
                model.backbone.set_use_fc(False)
                accuracy = evaluate(model, val_loader, device)
                model.backbone.set_use_fc(True) if args.train_mode == "classical" else model.backbone.set_use_fc(False)
                
                # logger
                if accuracy > max_accuracy:
                    logger(f"##### Accuracy on validation set: {accuracy:.4f} #####", title=False, log_dir=log_dir, mode=args.mode)
                    logger(f"##### Best accuracy so far: {max_accuracy:.4f}  =>  saving model... #####", title=False, log_dir=log_dir, mode=args.mode)
                else:
                    logger(f"##### Accuracy on validation set: {accuracy:.4f} #####", title=False, log_dir=log_dir, mode=args.mode)
                    logger(f"##### Best accuracy so far: {max_accuracy:.4f} #####", title=False, log_dir=log_dir, mode=args.mode)

                # tb record
                writer.add_scalar('Train/val_accuracy', accuracy, epoch)

                # save model
                if args.save_model and accuracy > max_accuracy:
                    torch.save(model.state_dict(), os.path.join(log_dir, 'model.pt'))
                    max_accuracy = accuracy
        
        # close tb recorder
        writer.close()

    # --------------------- testing loop ---------------------
    elif args.mode == "test":
        
        # dataset
        test_dataset = Fewshot(root=args.data_root, transform=None, mode="test")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # define model
        backbone = set_resnet(args.backbone, args.num_classes, device)
        model = PrototypicalNetworks(backbone).to(device)
        
        # load model
        assert os.path.isfile(os.path.join(log_dir, 'model.pt')), "model.pt is not in {}".format(args.log_dir)
        logger(f"Loading model from {os.path.join(log_dir, 'model.pt')}", title=True, log_dir=log_dir, mode=args.mode)
        model.load_state_dict(torch.load(os.path.join(log_dir, 'model.pt')))

        # predict
        model.backbone.set_use_fc(False)
        df_pred = predict(model, test_loader, device)

        # store
        df_pred.to_csv(os.path.join(log_dir, 'pred.csv'), index_label='Id')


    # -------------- code end --------------
    open(os.path.join(log_dir, 'done'), 'a').close()
    logger("Finished running", title=True, log_dir=log_dir, mode=args.mode)


if __name__ == '__main__':
    main()