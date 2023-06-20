import torch
import torch.nn as nn
import torchvision.models as models
from tqdm.auto import tqdm
import os
import torch.nn.functional as F

class Densenet121_Decoder(nn.Module):
    '''model of encoder-decoder'''
    def __init__(self,enable_BatchNorm2d_alllayer=True,enable_Dropout_alllayer=True,decrease_dropout=1,value_dropout=0.5):
        '''use a densnet and determinate a type of encoder
        enable_BatchNorm2d_alllayer=True insert batchNorm
        enable_Dropout_alllayer=True: insert dropout
        decrease_dropout=1 new value is X*decrease_dropout
        value_dropout=0.5 dropout of first layer'''
        super(Densenet121_Decoder, self).__init__()
        
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

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout2d(p=0.5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = self.pool(F.relu(self.dropout1(self.conv1(x))))
        x2 = self.pool(F.relu(self.dropout2(self.conv2(x1))))
        x3 = self.pool(F.relu(self.dropout3(self.conv3(x2))))
        x4 = self.pool(F.relu(self.dropout4(self.conv4(x3))))
        return x4, (x1, x2, x3)

class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.upconv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(p=0.5)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dropout5 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dropout6 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dropout7 = nn.Dropout2d(p=0.5)

    def forward(self, x, skip_connections):
            x1, x2, x3 = skip_connections
            x = F.relu(self.dropout1(self.upconv1(x)))
            x = torch.cat([x, x3], axis=1)
            x = F.relu(self.dropout5(self.conv1(x)))
            x = F.relu(self.dropout2(self.upconv2(x)))
            x = torch.cat([x, x2], axis=1)
            x = F.relu(self.dropout6(self.conv2(x)))
            x = F.relu(self.dropout3(self.upconv3(x)))
            x = torch.cat([x, x1], axis=1)
            x = F.relu(self.dropout7(self.conv3(x)))
            x = self.upconv4(x)
            x = F.relu(x)  # Use ReLU activation for positive depth values
            return x

class Encoder_Decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Encoder_Decoder, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return x

class ED_SkippConnection(nn.Module):
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3, value_dropout = 0.5):
        block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding ='same'),
                torch.nn.ReLU(),
                # torch.nn.BatchNorm2d(mid_channel),
                torch.nn.Dropout2d(value_dropout),
                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                )
        return  block        
    
    def __init__(self):
        super(ED_SkippConnection, self).__init__()
        
        self.pretrained_model = models.densenet121(pretrained = True)
        
        
        # Remove the last layer (classifier) of the DenseNet-121 model
        self.encoder = nn.Sequential(*list(self.pretrained_model.children())[:-1])
        
        self.bottleneck = torch.nn.Sequential(
                            torch.nn.Conv2d (kernel_size=3, in_channels=1024, out_channels=512, padding ='same'),
                            torch.nn.ReLU(),
                            torch.nn.Dropout2d(0.5),
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
