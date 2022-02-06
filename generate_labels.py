from torch.utils.data import DataLoader
from train_base import SegDataset
from model.build_BiSeNet import BiSeNet
import torch
import numpy as np
from PIL import Image
#import os

model_type='DA' #select model type (BASE, DA, DA_LIGHT, FDA)

palette = [128, 64, 128,  # Road - 0
            244, 35, 232,  # Sidewalk - 1
            70, 70, 70,  # Building - 2
            102, 102, 156,  # Wall - 3
            190, 153, 153,  # Fence - 4
            153, 153, 153,  # Pole - 5
            250, 170, 30,  # Light - 6
            220, 220, 0,  # Sign - 7
            107, 142, 35,  # Vegetation - 8
            152, 251, 152,  # Terrain - 9
            70, 130, 180,  # Sky - 10
            220, 20, 60,  # Person - 11
            255, 0, 0,  # Rider - 12
            0, 0, 142,  # Car - 13
            0, 0, 70,  # Truck - 14
            0, 60, 100,  # Bus - 15
            0, 80, 100,  # Train - 16
            0, 0, 230,  # Motocycle - 17
            119, 11, 32]  # Bicycle - 18]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
    

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


num_classes = 19
context_path = 'resnet18'#'resnet101'
model = BiSeNet(num_classes, context_path)


if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()

print('load model from %s ...' % './checkpoints_{}/best_dice_loss.pth'.format(model_type))
model.module.load_state_dict(torch.load('./checkpoints_{}/best_dice_loss.pth'.format(model_type)))
print('Done!')
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


dataset_val = SegDataset(train_or_val='val')
dataloader_val = DataLoader(dataset_val,batch_size=1) #load one image at a time

outputs=[]
with torch.no_grad():
        for images, labels in dataloader_val:

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                output = model(images)

                output = output.cpu().numpy()
                output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8)
                outputs.append(output)
                        
                        
                        

path="./data/Cityscapes/val.txt"
img_files_txt = np.loadtxt(path, dtype=str, delimiter="\n")
for i in range(len(outputs)):
        img_file_name = img_files_txt[i].split("/") 
        
        out=outputs[i]
        outputs[i] = np.squeeze(out, 0)
        outputs[i] = np.asarray(outputs[i], dtype=np.uint8)
        outputs[i] = colorize_mask(outputs[i])
        outputs[i].save('./best_pseudolabels_{}/{}'.format(model_type,img_file_name[1]))
        


