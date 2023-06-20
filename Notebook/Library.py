from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.model_selection import train_test_split

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import imageio.v2 as io 
import json

from skimage.transform import resize
from tqdm.auto import tqdm

import os

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

def print_dataset(train_loader,rows = 6,offset=0,show_clip=True):
    '''Print dataset and you can choose if want see clip or not clip or value of rgbd
    train_loader: tensore
    row: number of row
    offset: offset in tensore
    show_clip: enable to clip up 98-percentile and down 2-percentile'''
    count = 0
    for batch_idx, (img_rgb, img_rgbd) in enumerate(train_loader):     # Iterare sui batch del train_loader
        for sample_idx in range(img_rgb.size(0)):         # Iterare sui singoli campioni nel batch
            img_rgb_np = img_rgb[sample_idx].permute(1,2,0).cpu().numpy()
            img_rgbd_np = img_rgbd[sample_idx].permute(1,2,0).cpu().numpy()

            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(img_rgb_np)
            axes[0].axis('off')
            axes[0].set_title("RGB Image")
            axes[1].imshow(img_rgbd_np, cmap='jet')
            axes[1].axis('off')
            axes[1].set_title("RGBD Image")

            # Calculate the 98th percentile
            percentile_98 = np.percentile(img_rgbd_np, 98)
            percentile_2 = np.percentile(img_rgbd_np, 2)
            img_rgbd_np_tmp=np.copy(img_rgbd_np)
            img_rgbd_np_tmp[img_rgbd_np_tmp >= percentile_98] = percentile_98
            img_rgbd_np_tmp[img_rgbd_np_tmp <= percentile_2] = percentile_2
            axes[2].imshow(img_rgbd_np_tmp, cmap='jet')
            axes[2].axis('off')
            axes[2].set_title("RGBD Image modify")
            plt.show()
        
            count +=1
            if count == rows:
                break
        if count == rows:
            break

class ImageConverter(nn.Module):
    '''model of encoder-decoder'''
    def __init__(self,enable_BatchNorm2d_alllayer=True,enable_Dropout_alllayer=True,decrease_dropout=1,value_dropout=0.5):
        '''use a densnet and determinate a type of encoder
        enable_BatchNorm2d_alllayer=True insert batchNorm
        enable_Dropout_alllayer=True: insert dropout
        decrease_dropout=1 new value is X*decrease_dropout
        value_dropout=0.5 dropout of first layer'''
        super(ImageConverter, self).__init__()
        
        pretrained_model = models.densenet121(pretrained = True)
        
        # Remove the last layer (classifier) of the DenseNet-121 model
        self.encoder = nn.Sequential(*list(pretrained_model.children())[:-1])
        
        if enable_BatchNorm2d_alllayer==True and enable_Dropout_alllayer==True:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=value_dropout/decrease_dropout),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=value_dropout/(decrease_dropout*2)),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=value_dropout/(decrease_dropout*3)),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=value_dropout/(decrease_dropout*4)),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(p=value_dropout/(decrease_dropout*5)),
                nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1),
            )
        elif enable_BatchNorm2d_alllayer==True and enable_Dropout_alllayer==False:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1),
            )
        elif enable_BatchNorm2d_alllayer==False and enable_Dropout_alllayer==True:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=value_dropout/decrease_dropout),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=value_dropout/(decrease_dropout*2)),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=value_dropout/(decrease_dropout*3)),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=value_dropout/(decrease_dropout*4)),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=value_dropout/(decrease_dropout*5)),
                nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1),
            )
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1),
            )
    
    def forward(self, x,):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, criterion, optimizer, train_loader, val_loader, num_epochs, patience, scheduler,new_experiment_path):
    '''Training the model with
    criterion: loss
    optimizer: gradient
    train_loader,val_loader : tensor of image
    num_epochs : num max of epoch
    patience : num epoche when if increase result we stop it
    scheduler: determinate change loss of lr
    new_experiment_path:pathe when experimentxperiment
    '''
    
    best_val_loss = float('inf')
    best_epoch = 0
    if torch.cuda.is_available():
        model.cuda()
    list_loss_train=[]
    list_loss_val=[]
    print ("Training...")

    for epoch in range(num_epochs):
        epoch_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)#training
        train_loss = 0
        for batch, data in enumerate(train_loader):
            images, img_rgbd = data
            model.train()  # forward pass
            outputs = model(images)
            loss = criterion(outputs, img_rgbd)#calcolo loss
            train_loss += loss
            optimizer.zero_grad()#optimizer zero_grad           
            loss.backward()#loss backward
            optimizer.step()#optimizer step
            epoch_bar.set_postfix({"Train Loss": train_loss / (batch + 1)})# Update the current epoch progress bar
            epoch_bar.update(1)
        epoch_bar.close()
        
        train_img,train_true,train_output=images,img_rgbd,outputs
        train_loss /= len(train_loader) # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        list_loss_train.append(float(train_loss)) # print (len(train_loader))   
        
        val_loss = 0 # Setup variables for accumulatively adding up loss and accuracy 
        model.eval()
        val_bar = tqdm(total=len(val_loader), desc="Validation", leave=False)
        with torch.inference_mode():
            for batch, data in enumerate(val_loader):
                images, img_rgbd = data
                val_outputs = model(images) # 1. Forward pass          
                # 2. Calculate loss (accumatively)
                val_loss += criterion(val_outputs, img_rgbd) # accumulatively add up the loss per epoch
                val_bar.set_postfix({"Val Loss": val_loss / (batch + 1)})
                val_bar.update(1)
            # Calculations on test metrics need to happen inside torch.inference_mode()        
            val_loss /= len(val_loader)# Divide total test loss by length of test dataloader (per batch)        
            list_loss_val.append(float(val_loss))# print (len(val_loader))

        val_bar.close()
        scheduler.step(val_loss)
        
        print(f"Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f}") ## Print out what's happening
        
        torch.save(model.state_dict(),os.path.join(new_experiment_path, 'val_model.pt'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(new_experiment_path,'best_model.pt'))
        else:
            if epoch - best_epoch >= patience:
                print(f"\nEarly Stopping at Epoch {epoch+1}")
                break

    return list_loss_train,list_loss_val,images,img_rgbd,val_outputs,train_img,train_true,train_output

class ImageConverter2(nn.Module):
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding ='same'),
                torch.nn.ReLU(),
                # torch.nn.BatchNorm2d(mid_channel),
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding ='same'),
                torch.nn.ReLU(),
                # torch.nn.BatchNorm2d(mid_channel),
                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                )
        return  block        
    
    def __init__(self,enable_BatchNorm2d_alllayer=True,enable_Dropout_alllayer=True,decrease_dropout=1,value_dropout=0.5):
        super(ImageConverter2, self).__init__()
        
        self.pretrained_model = models.densenet121(pretrained = True)
        
        
        # Remove the last layer (classifier) of the DenseNet-121 model
        self.encoder = nn.Sequential(*list(self.pretrained_model.children())[:-1])
        
        self.bottleneck = torch.nn.Sequential(
                            torch.nn.Conv2d (kernel_size=3, in_channels=1024, out_channels=512, padding ='same'),
                            torch.nn.ReLU(),
                            # torch.nn.BatchNorm2d(512)
                            )
            
        self.conv_decode_4 = self.expansive_block(1024, 512, 256)
            
        self.conv_decode_3 = self.expansive_block(512, 256, 128)
            
        self.conv_decode_2 = self.expansive_block(256, 128, 64)
            
        self.conv_decode_1 = self.expansive_block(128, 64, 32)
        
        self.ultima_up = torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # convoluzione con kernel 1x1 su tutti i canali in ingresso che ha come uscita la depth
        self.conv_finale = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        
        
    def concat(self, upsampled, bypass):
        return torch.cat((upsampled, bypass), 1)

    def forward(self, input_x):
        features_encoder = self.pretrained_model.features
        # Encode
        features_densenet_5 = self.encoder(input_x) #1024 features 12x16
        features_densenet_4 = features_encoder[:10](input_x) # 512 features 12x16
        features_densenet_3 = features_encoder[:8](input_x) # 256 features 24x32
        features_densenet_2 = features_encoder[:6](input_x) # 128 features 48x64
        features_densenet_1 = features_encoder[:4](input_x) # 64 features 96x128

        # bottleneck
        bottleneck = self.bottleneck(features_densenet_5)
        
        # decode
        features_4 = self.concat(bottleneck, features_densenet_4)
        decode_4 = self.conv_decode_4(features_4)
        
        features_3 = self.concat(decode_4, features_densenet_3)
        decode_3 = self.conv_decode_3(features_3)
        
        features_2 = self.concat(decode_3, features_densenet_2)
        decode_2 = self.conv_decode_2(features_2)  
        
        features_1 = self.concat(decode_2, features_densenet_1)
        decode_1 = self.conv_decode_1(features_1)
        
        features_0 = self.ultima_up(decode_1)
        
        # conv 1x1
        depth_map = self.conv_finale(features_0)
        
        return depth_map

def train2(model, criterion, optimizer, train_loader, val_loader, num_epochs, patience, scheduler,new_experiment_path):
    '''Training the model with
    criterion: loss
    optimizer: gradient
    train_loader,val_loader : tensor of image
    num_epochs : num max of epoch
    patience : num epoche when if increase result we stop it
    scheduler: determinate change loss of lr
    new_experiment_path:pathe when experimentxperiment
    '''
    
    best_val_loss = float('inf')
    best_epoch = 0
    if torch.cuda.is_available():
        model.cuda()
    list_loss_train=[]
    list_loss_val=[]
    print ("Training...")

    for epoch in range(num_epochs):
        epoch_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)#training
        train_loss = 0
        for batch, data in enumerate(train_loader):
            images, img_rgbd = data
            model.train()  # forward pass
            outputs = model(images)
            loss = criterion(outputs, img_rgbd)#calcolo loss
            train_loss += loss
            optimizer.zero_grad()#optimizer zero_grad           
            loss.backward()#loss backward
            optimizer.step()#optimizer step
            epoch_bar.set_postfix({"Train Loss": train_loss / (batch + 1)})# Update the current epoch progress bar
            epoch_bar.update(1)
        epoch_bar.close()
        
        train_img,train_true,train_output=images,img_rgbd,outputs
        train_loss /= len(train_loader) # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        list_loss_train.append(float(train_loss)) # print (len(train_loader))   
        
        val_loss = 0 # Setup variables for accumulatively adding up loss and accuracy 
        model.eval()
        val_bar = tqdm(total=len(val_loader), desc="Validation", leave=False)
        with torch.inference_mode():
            for batch, data in enumerate(val_loader):
                images, img_rgbd = data
                val_outputs = model(images) # 1. Forward pass          
                # 2. Calculate loss (accumatively)
                val_loss += criterion(val_outputs, img_rgbd) # accumulatively add up the loss per epoch
                val_bar.set_postfix({"Val Loss": val_loss / (batch + 1)})
                val_bar.update(1)
            # Calculations on test metrics need to happen inside torch.inference_mode()        
            val_loss /= len(val_loader)# Divide total test loss by length of test dataloader (per batch)        
            list_loss_val.append(float(val_loss))# print (len(val_loader))

        val_bar.close()
        scheduler.step(val_loss)
        
        print(f"Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f}") ## Print out what's happening
        
        torch.save(model.state_dict(),os.path.join(new_experiment_path, 'val_model.pt'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(new_experiment_path,'best_model.pt'))
        else:
            if epoch - best_epoch >= patience:
                print(f"\nEarly Stopping at Epoch {epoch+1}")
                break

    return list_loss_train,list_loss_val,images,img_rgbd,val_outputs,train_img,train_true,train_output

def iou_loss(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return 1 - iou

class CustomLoss(nn.Module):
    def __init__(self, peso_l1=0.5):
        super(CustomLoss, self).__init__()
        self.peso_l1 = peso_l1

    def forward(self, y_true, y_pred):
        l1_loss = torch.mean(torch.abs(y_true - y_pred))
        ssim_loss_val = 1 - torchvision.functional.ssim(y_pred, y_true)
        loss = self.peso_l1 * l1_loss + (1 - self.peso_l1) * ssim_loss_val
        return loss

def print_epoch(epoch_loss_train, epoch_loss_val):
    plt.plot(range(len(epoch_loss_train)),epoch_loss_train,'r--', label = 'Training Loss')
    plt.plot(range(len(epoch_loss_val)), epoch_loss_val, 'b--', label='Validation Loss')
    plt.legend()
    plt.xticks(range(len(epoch_loss_train)), range(len(epoch_loss_train)))
    plt.show()


def save_epoch(list_loss_train,list_loss_val,save_path=''):
        # Generazione del plot
    if save_path != '':
        plt.plot(list_loss_train, label="Train")
        plt.plot(list_loss_val, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Salvataggio del plot nella cartella di esperimento
        plot_file = os.path.join(save_path, "Epoch.png")
        plt.savefig(plot_file)
        plt.close()
        #plt.show()
    

# Prende i pesi salvati
def print_dataset_pred(model,loader, rows=6, offset=0, model_path='/best_model.pt', save_path='',name=''):
    
    model.load_state_dict(torch.load(save_path+'/best_model.pt'))
    if torch.cuda.is_available():
        model.cuda()

    # Create a figure and axes with 6 rows and 3 columns
    fig, axes = plt.subplots(rows, 3)
    count=0
    # Iterate over the batch of the loader
    for batch, data in enumerate(loader):
        img_rgb, img_rgbd = data

        # Iterate over the individual samples in the batch
        for sample_idx in range(img_rgb.size(0)):
            # Convert tensors to NumPy arrays
            img_rgb_np = img_rgb[sample_idx].permute(1, 2, 0).cpu().numpy()
            img_rgbd_np = img_rgbd[sample_idx].permute(1, 2, 0).cpu().numpy()

            # Display images in the subplot
            axes[count // 3, count % 3].imshow(img_rgb_np)
            axes[count // 3, count % 3].axis('off')
            axes[count // 3, count % 3].set_title("RGB Image")

            model.eval()
            with torch.inference_mode():
                pred = model(img_rgb)
            result = pred[sample_idx].permute(1, 2, 0).cpu().detach().numpy()

            axes[count // 3, count % 3 + 1].imshow(img_rgbd_np, cmap='jet')
            axes[count // 3, count % 3 + 1].axis('off')
            axes[count // 3, count % 3 + 1].set_title("RGBD Image")

            axes[count // 3, count % 3 + 2].imshow(result, cmap='jet')
            axes[count // 3, count % 3 + 2].axis('off')
            axes[count // 3, count % 3 + 2].set_title("Prediction")

            count += 3
            if count >= rows * 3:
                break
        if count >= rows * 3:
            break
    # Adjust the spacing between subplots
    fig.tight_layout()

    # Save the plot if save_path is provided
    if save_path != '':
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, str(name)+str(offset)+".png")
        plt.savefig(save_file_path)
        plt.close()
    else:
        # Display the plot
        plt.show()

# Prende i pesi salvati
def print_dataset_train(train_img,train_true,train_output, rows=6, offset=0, model_path='best_model.pt', save_path='',name=''):
    # Generazione del plot
   
    fig, axes = plt.subplots(rows, 3)
    count = 0
    img_rgb=train_img
    img_rgbd=train_true
    pred=train_output
    # Iterate over the batch of the loader
    for counter in range(rows):

        # Iterate over the individual samples in the batch
        for sample_idx in range(rows):
            # Convert tensors to NumPy arrays
            img_rgb_np = img_rgb[sample_idx].permute(1, 2, 0).cpu().numpy()
            img_rgbd_np = img_rgbd[sample_idx].permute(1, 2, 0).cpu().numpy()
            
            # Display images in the subplot
            axes[count // 3, count % 3].imshow(img_rgb_np)
            axes[count // 3, count % 3].axis('off')
            axes[count // 3, count % 3].set_title("RGB Image")

            result = pred[sample_idx].permute(1, 2, 0).cpu().detach().numpy()

            axes[count // 3, count % 3 + 1].imshow(img_rgbd_np, cmap='jet')
            axes[count // 3, count % 3 + 1].axis('off')
            axes[count // 3, count % 3 + 1].set_title("RGBD Image")

            axes[count // 3, count % 3 + 2].imshow(result, cmap='jet')
            axes[count // 3, count % 3 + 2].axis('off')
            axes[count // 3, count % 3 + 2].set_title("Prediction")

            count += 3
            if count >= rows * 3:
                break

        if count >= rows * 3:
            break

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Save the plot if save_path is provided
    if save_path != '':
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, str(name)+str(offset)+".png")
        plt.savefig(save_file_path)
        plt.close()
        plt.show()
    else:
        # Display the plot
        plt.show()



