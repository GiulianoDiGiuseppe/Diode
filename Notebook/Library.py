from PIL import Image
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

import cv2
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def create_df_ipcv_diode(path):
    folder=os.listdir(path)
    df = pd.DataFrame(columns=['scene','scan','name','path_rgb','path_depth','path_mask'])
    for scene in folder:
        scans=os.listdir(path+'/'+scene)
        for scan in scans:
            files=os.listdir(path+'/'+scene+'/'+scan)
            for file in files:
                if file[-4:]=='.png':
                    #print(file[:-4])
                    path_base=path+'/'+scene+'/'+scan+'/'+file[:-4]
                    #print(str(path_base+'.png'), str(path_base+'_depth.npy',str(path_base+'_depth_mask.npy'))

                    df.loc[len(df)]={'scene':str(scene),'scan':scan,'name':file[:-4]
                                    ,'path_rgb': str(path_base+'.png'),
                                    'path_depth': str(path_base+'_depth.npy'),
                                    'path_mask':str(path_base+'_depth_mask.npy')}
    df.to_csv("../Csv/path_images.csv",index=False)

class CustomDataset:
    def __init__(self, csv_path, test_size=0.3, random_state=42):
        self.df = pd.read_csv(csv_path)
        self.img_rgbs = []
        self.img_rgbds = []
        self.depth_maps = []
        for index, element in self.df.iterrows():
            img_rgb = Image.open(element['path_rgb'])
            img_rgbd = np.load(element['path_depth'])
            img_norm = np.load(element['path_mask'])
            self.img_rgbs.append(np.array(img_rgb))
            self.img_rgbds.append(img_rgbd)
            self.depth_maps.append(img_norm)
        self.X_train_rgb, self.X_test_rgb, self.y_train_depth, self.y_test_depth, self.y_train_rgbds, self.y_test_rgbds = train_test_split(self.img_rgbs, self.depth_maps, self.img_rgbds, test_size=test_size, random_state=random_state)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.X_train_rgb[idx], self.y_train_depth[idx], self.y_train_rgbds[idx]