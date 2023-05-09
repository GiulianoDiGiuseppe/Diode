from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import cv2
import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


torch.cuda.get_device_name(0)



def create_df_ipcv_diode(path,path_dst='../Csv/',name='path_images'):
    folder=os.listdir(path)
    df = pd.DataFrame(columns=['scene','scan','name','path_rgb','path_depth','path_mask'])
    for scene in folder:
        scans=os.listdir(path+'/'+scene)
        for scan in scans:
            files=os.listdir(path+'/'+scene+'/'+scan)
            for file in files:
                if file[-4:]=='.png':
                    path_base=path+'/'+scene+'/'+scan+'/'+file[:-4]
                    df.loc[len(df)]={'scene':str(scene),'scan':scan,'name':file[:-4]
                                    ,'path_rgb': str(path_base+'.png'),
                                    'path_depth': str(path_base+'_depth.npy'),
                                    'path_mask':str(path_base+'_depth_mask.npy')}
    df.to_csv(path_dst+name+'.csv',index=False)


def print_dataset(train_loader,rows = 6,cols = 6,offset=12):
    fig, axs = plt.subplots(rows, cols, figsize=(12,6))
    axs = axs.flatten()
    i=0
    for index ,imagesRGBD in enumerate(train_loader.dataset.y_train_rgbds[offset:offset+int(rows*cols/3)]):
        image=train_loader.dataset.X_train_rgb[offset+index]
        mask=train_loader.dataset.y_train_depth[offset+index]

        axs[i].imshow(image)
        axs[i].axis('off')
        i+=1
        axs[i].imshow(imagesRGBD)
        axs[i].axis('off')
        i+=1

        image_ = image
        image_ = np.array(image_) / 255.0

        depth_map = imagesRGBD
        mask = mask > 0
        max_depth = min(300, np.percentile(depth_map, 99))    # calcola il massimo valore della mappa di profondità per evitare valori troppo grandi

        depth_map = np.log(depth_map, where=np.expand_dims(mask, axis=-1))# applica la scala logaritmica alla mappa di profondità
        depth_map = np.clip(depth_map, 0.1, np.log(max_depth))

        axs[i].imshow(depth_map)
        axs[i].axis('off')
        i+=1
        
    # Mostra il plot
    plt.show()


class CustomDataset:
    def __init__(self, csv_path, test_size=0.3, random_state=42, perc_dataset=1, resize_shape=(768,1024)):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.iloc[:int(len(self.df)*perc_dataset)]
        self.resize_shape = resize_shape 
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
        print("Element load : ",len(self.img_rgbds))

    def __len__(self):
        return int(len(self.df))
    
    def __getitem__(self, idx):
        img_rgb = self.img_rgbs[idx]
        img_rgbd = self.img_rgbds[idx]
        img_norm = self.depth_maps[idx]
        
        # convertire le immagini in tensori PyTorch
        img_rgb_tensor = torch.from_numpy(img_rgb).float()
        img_rgbd_tensor = torch.from_numpy(img_rgbd).float()
        img_norm_tensor = torch.from_numpy(img_norm).float()
        
        # spostare i tensori sulla GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img_rgb_tensor = img_rgb_tensor.to(device)
        img_rgbd_tensor = img_rgbd_tensor.to(device)
        img_norm_tensor = img_norm_tensor.to(device)
        
        return img_rgb_tensor, img_rgbd_tensor, img_norm_tensor



# class CustomDataset:
#     def __init__(self, csv_path, test_size=0.3, random_state=42, perc_dataset=1):
#         self.df = pd.read_csv(csv_path)
#         self.df = self.df.iloc[:int(len(self.df)*perc_dataset)]
#         self.img_rgbs = []
#         self.img_rgbds = []
#         for index, element in self.df.iterrows():
#             img_rgb = Image.open(element['path_rgb'])
#             img_rgbd = np.load(element['path_depth'])
#             self.img_rgbs.append(np.array(img_rgb))
#             self.img_rgbds.append(img_rgbd)
#         self.X_train_rgb, self.X_test_rgb, self.y_train_rgbds, self.y_test_rgbds = train_test_split(self.img_rgbs, self.img_rgbds, test_size=test_size, random_state=random_state)
#         print("load ", int(len(self.df)*perc_dataset), " on: ", len(self.X_train_rgb) + len(self.X_test_rgb))

#     def __len__(self):
#         return int(len(self.df)*perc_dataset)
    
#     def __getitem__(self, idx):
#         img_rgb = self.X_train_rgb[idx]
#         img_rgbd = self.y_train_rgbds[idx]
#         img_rgb = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0  # convert to tensor and normalize
#         img_rgbd = torch.tensor(img_rgbd, dtype=torch.float32).unsqueeze(0) / 1000.0  # convert to tensor and normalize
#         return img_rgb, img_rgbd

