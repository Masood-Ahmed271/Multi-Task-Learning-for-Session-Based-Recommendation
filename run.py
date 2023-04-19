import torch
import random
import time
import argparse
from utils import *
from trainer import *
from neg_sampler import *
from load_model import *
from splitter import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from main import *

#----------------GRU4Rec---------------#
print('=============sequence=============')
train_loader, val_loader, test_loader = get_data(args)
model = get_model(args)
SeqTrain(args.epochs, model, train_loader, val_loader, writer, args)
if args.eval:
    best_weight = torch.load(os.path.join(args.save_path,
                                                  '{}_{}_seed{}_is_pretrain_{}_best_model_lr{}_wd{}_block{}_hd{}_emb{}.pth'.format(args.task_name, args.model_name, args.seed, args.is_pretrain,
                                                                                                                              args.lr, args.weight_decay, args.block_num, args.hidden_size, args.embedding_size)))
    model.load_state_dict(best_weight)
    model = model.to(args.device)



#-----------------MMOE----------------#
train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict = get_data(args)
model = MMOE(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, device=args.device, num_task=num_task)
mtlTrain(model, train_dataloader, val_dataloader, test_dataloader, args, train=False)