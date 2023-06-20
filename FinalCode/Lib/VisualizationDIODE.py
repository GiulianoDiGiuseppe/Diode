import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

def create_folder_experiment(kaggle,config_dict):
    if kaggle:
        experiment_dir = "/kaggle/working"
    else:
        experiment_dir = "../Experiment"
    new_experiment_dir = "exp0"

    # Cerca l'ultima cartella di esperimento esistente
    last_exp_num_max = 0
    for folder_name in os.listdir(experiment_dir):
        if folder_name.startswith("exp"):
            new_experiment_dir = folder_name
            last_exp_num_temp = int(new_experiment_dir[3:])
            if (last_exp_num_temp) > (last_exp_num_max):
                last_exp_num_max = last_exp_num_temp
                
    new_exp_num = last_exp_num_max + 1
    new_experiment_dir = f"exp{new_exp_num}"
    new_experiment_path = os.path.join(experiment_dir, new_experiment_dir)
    os.makedirs(new_experiment_path)

    json_data = json.dumps(config_dict) # Convert the dictionary to a JSON string

    json_file_path = os.path.join(new_experiment_path, "config.json") # Save the JSON string to a file in the experiment folder
    with open(json_file_path, "w") as file:
        file.write(json_data)

    return new_experiment_path

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


# Prende i pesi salvati
def print_dataset_pred(model,loader, rows=6,offset=0, model_path='/best_model.pt', save_path='',name=''):
    
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

def clamp(img_rgbd_np):
    lower_percentile = np.percentile(img_rgbd_np, 5)
    upper_percentile = np.percentile(img_rgbd_np, 95)
    img_rgbd_np[img_rgbd_np < lower_percentile] = lower_percentile
    img_rgbd_np[img_rgbd_np > upper_percentile] = upper_percentile
    return img_rgbd_np

# Prende i pesi salvati
def print_comparison(dic_model,loader, rows=6, offset=0, model_path='/best_model.pt', save_path='',name=''):

    # Create a figure and axes with 6 rows and rows columns
    fig, axes = plt.subplots(rows, (2+len(dic_model)))
    count=0
    # Iterate over the batch of the loader
    for batch, data in enumerate(loader):
        img_rgb, img_rgbd = data

        # Convert tensors to NumPy arrays
        img_rgb_np = img_rgb[0].permute(1, 2, 0).cpu().numpy()
        img_rgbd_np = img_rgbd[0].permute(1, 2, 0).cpu().numpy()

        # Display images in the subplot
        axes[count % rows, (2+len(dic_model)) % (2+len(dic_model))].imshow(img_rgb_np)
        axes[count % rows, (2+len(dic_model)) % (2+len(dic_model))].axis('off')
        axes[count % rows, (2+len(dic_model)) % (2+len(dic_model))].set_title("RGB ")

        #print(img_rgbd_np.max())
        img_rgbd_np=clamp(img_rgbd_np)
        
        axes[count % rows, (2+len(dic_model)) % (2+len(dic_model)) + 1].imshow(img_rgbd_np, cmap='jet')
        axes[count % rows, (2+len(dic_model)) % (2+len(dic_model)) + 1].axis('off')
        axes[count % rows, (2+len(dic_model)) % (2+len(dic_model)) + 1].set_title("RGBD ")

        cnt=2
        for titolo, model in dic_model.items():
            model.eval()
            with torch.inference_mode():
                pred = model(img_rgb)
            result = pred[0].permute(1, 2, 0).cpu().detach().numpy()
            result=clamp(result)
            axes[count % rows, (2+len(dic_model)) % (2+len(dic_model)) + cnt].imshow(result, cmap='jet')
            axes[count % rows, (2+len(dic_model)) % (2+len(dic_model)) + cnt].axis('off')
            axes[count % rows, (2+len(dic_model)) % (2+len(dic_model)) + cnt].set_title(titolo)
            cnt+=1

        #print(count,rows)
        count +=1
        if count >= rows:
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