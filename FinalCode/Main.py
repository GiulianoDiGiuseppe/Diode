from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import imageio.v2 as io 
import json
import yaml
from skimage.transform import resize
from tqdm.auto import tqdm
import argparse
import os

from Lib.LoadDIODE import *
from Lib.VisualizationDIODE import *
from Lib.ModelsDIODE import *

parser = argparse.ArgumentParser(description='Lettura file YAML')
parser.add_argument('--config', type=str, help='Percorso del file YAML di configurazione')
args = parser.parse_args()
with open(str(args.config), 'r') as file: #../hyp/Config.yaml
    config = yaml.safe_load(file)
print("Read yaml file" )

train_scenes = config["train_scenes"]
val_scenes = config["val_scenes"]
test_scenes = config["test_scenes"]
enable_aug_train = config["enable_aug_train"]
enable_aug_val = config["enable_aug_val"]
enable_aug_test = config["enable_aug_test"]
clamp = config["clamp"]
percentuale = config["percentuale"]
batch_size = config["batch_size"]
shuffle = config["shuffle"]
show_clip = config["show_clip"]
en_clip = config["en_clip"]
enable_BatchNorm2d_alllayer = config["enable_BatchNorm2d_alllayer"]
enable_Dropout_alllayer = config["enable_Dropout_alllayer"]
decrease_dropout = config["decrease_dropout"]
value_dropout = config["value_dropout"]
criterion = nn.L1Loss() 
lr = config["lr"]
weight_decay = config["weight_decay"]
patience_sched = config["patience_sched"]
factor_sched = config["factor_sched"]
verbose_sched = config["verbose_sched"]
num_epochs = 2#config["num_epochs"]
patience_earlt = config["patience_earlt"]
size = tuple(config["size"])
path_dataset=config["path_dataset"]
path_dst = config["path_dst"]
csv_path=config["csv_path"]
csv_path_aug=config["csv_path_aug"]
csv_path_not_aug=config["csv_path_not_aug"]
kaggle=config["kaggle"]
model_type=config["model_type"]

config_dict = {
    "train_scenes": train_scenes,
    "val_scenes": val_scenes,
    "test_scenes": test_scenes,
    "enable_aug_train": enable_aug_train,
    "enable_aug_val": enable_aug_val,
    "enable_aug_test": enable_aug_test,
    "clamp": clamp,
    "percentuale": percentuale,
    "batch_size": batch_size,
    "shuffle": shuffle,
    "show_clip": show_clip,
    "en_clip": en_clip,
    "enable_BatchNorm2d_alllayer": enable_BatchNorm2d_alllayer,
    "enable_Dropout_alllayer": enable_Dropout_alllayer,
    "decrease_dropout": decrease_dropout,
    "value_dropout": value_dropout,
    "criterion": str(criterion),
    "lr": lr,
    "weight_decay": weight_decay,
    "patience_sched": patience_sched,
    "factor_sched": factor_sched,
    "verbose_sched": verbose_sched,
    "num_epochs": num_epochs,
    "patience_earlt": patience_earlt,
    "size": size
}

csv_train_path=path_dst+"path_train.csv"
csv_val_path=path_dst+"path_val.csv"
csv_test_path=path_dst+"path_test.csv"

print(config_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

create_df_ipcv_diode(path_dataset,path_dst=path_dst)   
train_val_test_split(csv_path_aug, csv_path_not_aug, path_dst=path_dst,enable_aug_train=enable_aug_train, enable_aug_val=enable_aug_val, enable_aug_test=enable_aug_test)
# faccio clipping sulla depth_map 
if en_clip:
    train_set =CustomDataset(csv_train_path, perc_dataset=percentuale, transform=rgb_transformations_base(size,device), target_transform=depth_map_transformations_clip(size,device))
    val_set = CustomDataset(csv_val_path, perc_dataset=percentuale, transform=rgb_transformations_base(size,device), target_transform=depth_map_transformations_clip(size,device))
else:
    train_set = CustomDataset(csv_train_path, perc_dataset=percentuale, transform=rgb_transformations_base(size,device), target_transform=depth_map_transformations_no_clip(size,device))
    val_set = CustomDataset(csv_val_path, perc_dataset=percentuale, transform=rgb_transformations_base(size,device), target_transform=depth_map_transformations_no_clip(size,device))
test_set = CustomDataset(csv_test_path, perc_dataset=1, transform=rgb_transformations_test(device), target_transform=depth_map_transformations_test(device))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
print('Img train :' ,len(train_set),'\t Img val :', len(val_set),'\t Img test :', len(test_set),"\t Totale :",len(train_set)+len(val_set)+len(test_set))

new_experiment_path=create_folder_experiment(kaggle,config_dict)

while model_type not in ['Dense121','Base','Skip']:
    model_type=input('Inserisci Dense121/Base/Skip :')
if model_type=='Dense121':
    model = Densenet121_Decoder(enable_BatchNorm2d_alllayer=enable_BatchNorm2d_alllayer,enable_Dropout_alllayer=enable_Dropout_alllayer,decrease_dropout=decrease_dropout,value_dropout=value_dropout)
elif model_type=='Base':
    model=Encoder_Decoder()
elif model_type=='Skip':
    model=ED_SkippConnection()


criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience_sched, factor=factor_sched, verbose=True)
list_loss_train,list_loss_val,val_images,val_img_rgbd,val_outputs,train_img,train_true,train_output=train(model, criterion, optimizer, train_loader, val_loader, num_epochs, patience_earlt, scheduler,new_experiment_path)

save_epoch(list_loss_train,list_loss_val,save_path=new_experiment_path)
print_dataset_pred(model,train_loader,rows = 4,offset=0,save_path=new_experiment_path,name='train')
print_dataset_pred(model,train_loader,rows = 4,offset=10,save_path=new_experiment_path,name='train')
print_dataset_pred(model,train_loader,rows = 4,offset=20,save_path=new_experiment_path,name='train')
print_dataset_pred(model,val_loader,rows = 4,offset=0,save_path=new_experiment_path,name='val')
print_dataset_pred(model,val_loader,rows = 4,offset=10,save_path=new_experiment_path,name='val')
print_dataset_pred(model,val_loader,rows = 4,offset=20,save_path=new_experiment_path,name='val')

print("END")