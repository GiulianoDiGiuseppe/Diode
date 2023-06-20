import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
from PIL import Image
import os

def rgb_transformations(size, device):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x: x.to(device))
    ])
    return transform

def rgb_transformations_base(size, device):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x: x.to(device))
    ])
    return transform

def depth_map_transformations_no_clip(size, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x: x.to(device))
    ])
    return transform

def depth_map_transformations_clip(size, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.clamp_(x.flatten().kthvalue(int(0.02 * x.numel())).values.item(), x.flatten().kthvalue(int(0.98 * x.numel())).values.item())),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x: x.to(device))
    ])
    return transform

def rgb_transformations_test(device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x: x.to(device))
    ])
    return transform

def depth_map_transformations_test(device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x: x.to(device))
    ])
    return transform



def create_df_ipcv_diode(path,path_dst='../Csv/',name='path_images'):
    ''' Create csv for aug and not aug for all dataset
        path: folder of dataset
        path_dst : folder of destination_csv
        name: name of csv'''
    folder=os.listdir(path)
    df = pd.DataFrame(columns=['scene','scan','name','path_rgb','path_depth','path_mask'])
    
    for scene in folder:
        scans=os.listdir(path+'/'+scene)
        for scan in scans:
            files=os.listdir(path+'/'+scene+'/'+scan)
            for file in files:
                if file[-4:]=='.png':
                    path_base=path+'/'+scene+'/'+scan+'/'+file[:-4]
                    last_ = path_base.rfind("_")
                    if path_base.endswith(('F', 'G', 'S', 'C', '_')):
                        df.loc[len(df)]={'scene':str(scene),'scan':scan,'name':file[:-4]
                                        ,'path_rgb': str(path_base+'.png'),
                                        'path_depth': str(path_base[:last_]+'_depth'+path_base[last_:]+'.npy'),
                                        'path_mask':str( path_base[:last_]+'_depth_mask'+path_base[last_:]+'.npy')}
                    else:
                        df.loc[len(df)]={'scene':str(scene),'scan':scan,'name':file[:-4]
                                    ,'path_rgb': str(path_base+'.png'),
                                    'path_depth': str(path_base+'_depth'+'.npy'),
                                    'path_mask':str( path_base+'_depth_mask'+'.npy')}
    df.to_csv(path_dst+name+'_aug'+'.csv',index=False)
    folder=os.listdir(path)
    df = pd.DataFrame(columns=['scene','scan','name','path_rgb','path_depth','path_mask'])
    for scene in folder:
        scans=os.listdir(path+'/'+scene)
        for scan in scans:
            files=os.listdir(path+'/'+scene+'/'+scan)
            for file in files:
                if file[-4:]=='.png':
                    path_base=path+'/'+scene+'/'+scan+'/'+file[:-4]
                    last_ = path_base.rfind("_")
                    if not(path_base.endswith(('F', 'G', 'S', 'C', '_'))):
                        df.loc[len(df)]={'scene':str(scene),'scan':scan,'name':file[:-4]
                                    ,'path_rgb': str(path_base+'.png'),
                                    'path_depth': str(path_base+'_depth'+'.npy'),
                                    'path_mask':str( path_base+'_depth_mask'+'.npy')}                        
    df.to_csv(path_dst+name+'_not_aug'+'.csv',index=False)

def train_val_test_split(csv_path_aug, csv_path_not_aug ,train_scenes = ['scene_00000', 'scene_00001','scene_00006','scene_00004'],
                           val_scenes =['scene_00002', 'scene_00003'], test_scenes = ['scene_00005'], path_dst = 'F:/DeepLearning/IPCV (Diode)/Csv/',
                        enable_aug_train=True,enable_aug_val = False, enable_aug_test = False):
    '''Split dataframe in train val e test and choose the augmentation of 3 dataset
        csv_path_aug: path csv augmentation
        csv_path_not_aug: path of csv without augmentation
        train_scenen : list of scenes folder in train
        valid_scenen : list of scenes folder in valid
        test_scenen : list of scenes folder in test
        path_dst : path save file
        enable_aug_train,ìenable_aug_val,enable_aug_test : enable augmentation
        '''
    if enable_aug_train == True:
        csv_path_train = csv_path_aug
    else:
        csv_path_train = csv_path_not_aug    
    if enable_aug_val == True:
        csv_path_val = csv_path_aug
    else:
        csv_path_val = csv_path_not_aug   
    if enable_aug_test == True:
        csv_path_test = csv_path_aug
    else:
        csv_path_test = csv_path_not_aug  

    df = pd.read_csv(csv_path_train)  # per il training set leggo sempre il path aug
    train_df = df.loc[df['scene'].isin(train_scenes)]
    train_df.to_csv(path_dst+'path_train'+'.csv',index=False)
    
    df = pd.read_csv(csv_path_val)
    val_df = df.loc[df['scene'].isin(val_scenes)]
    val_df.to_csv(path_dst+'path_val'+'.csv',index=False)
    
    df = pd.read_csv(csv_path_test)
    test_df = df.loc[df['scene'].isin(test_scenes)]
    test_df.to_csv(path_dst+'path_test'+'.csv',index=False)  

class CustomDataset:
    '''Class to read value , we applicate transform clap
    csv_path : file to convert in dataframe
    perc_dataset : percent to read dataframe
    transform: make operation input
    target_trasform: make operation output
    '''
    def __init__(self, csv_path, perc_dataset=1, transform = None, target_transform = None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df.index%int(1/perc_dataset)==0]
        self.df = self.df.reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return int(len(self.df))
    
    def __getitem__(self, idx):        
        img_rgb = Image.open(self.df.loc[idx, 'path_rgb'])
        img_rgbd = np.load(self.df.loc[idx, 'path_depth'])
        # Apply the transformations to the image
        if self.transform:
            img_rgb = self.transform(img_rgb)
        if self.target_transform:
            img_rgbd = self.target_transform(img_rgbd)
            
        return img_rgb, img_rgbd