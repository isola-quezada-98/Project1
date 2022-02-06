import numpy as np
from PIL import Image
import os

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
    pass

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

# Label Image Load
path="./data/Cityscapes/val.txt"
img_files_txt = np.loadtxt(path, dtype=str, delimiter="\n")

id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

output_size=(1024,512)
for idx in range(len(img_files_txt)):
          img_file_name = img_files_txt[idx].split("/")  
          mask_file_name = img_file_name[1].rsplit("_",1)
          mask_file_name_complete = "./data/Cityscapes/labels/{}_gtFine_labelIds.png".format(mask_file_name[0]) 
          label= Image.open(mask_file_name_complete)
          label= label.resize(output_size,Image.NEAREST) #changed from ANTIALIAS
          label = np.array(label)

          # re-assign labels to match the format of Cityscapes
          label_copy = 255 * np.ones(label.shape, dtype=np.float32)
          for k, v in id_to_trainid.items():
            label_copy[label == k] = v
          
          label_copy = colorize_mask(label_copy)
          label_copy.save('./data/Cityscapes/color_labels/{}_gtFine_labelIds.png'.format(mask_file_name[0]))
          

