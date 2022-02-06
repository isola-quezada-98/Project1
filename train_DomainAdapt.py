import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import poly_lr_scheduler, adjust_learning_rate_D
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
import torch.cuda.amp as amp
from torchvision.io import read_image
from PIL import Image
from model.discriminator import FCDiscriminator, Light_Discriminator
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

class target_Dataset(data.Dataset):
    def __init__(self,train_or_val):
        path="./data/Cityscapes/{}.txt".format(train_or_val)
        self.img_files_txt = np.loadtxt(path, dtype=str, delimiter="\n")
        self.img_files=[]
        self.crop_size = (1024,512)
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

        for idx in range(len(self.img_files_txt)):
  
          img_file_name = self.img_files_txt[idx].split("/")
          self.img_files.append("./data/Cityscapes/images/{}".format(img_file_name[1]))
 

    def __getitem__(self, index):
        img_path = self.img_files[index]
        image = Image.open(img_path).convert('RGB')
        
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        

        image = np.asarray(image, np.float32)
    
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy()


    def __len__(self):
        return len(self.img_files)


class SegDataset(data.Dataset):
    def __init__(self,train_or_val):
        path="./data/Cityscapes/{}.txt".format(train_or_val)
        self.img_files_txt = np.loadtxt(path, dtype=str, delimiter="\n")
        self.img_files=[]
        self.mask_files=[]
        self.crop_size = (1024,512)
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for idx in range(len(self.img_files_txt)):
  
          img_file_name = self.img_files_txt[idx].split("/")
          self.img_files.append("./data/Cityscapes/images/{}".format(img_file_name[1]))
          mask_file_name = img_file_name[1].rsplit("_",1)
          self.mask_files.append("./data/Cityscapes/labels/{}_gtFine_labelIds.png".format(mask_file_name[0])) 

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
    
        image = Image.open(img_path).convert('RGB')
        label = Image.open(mask_path)

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy()


    def __len__(self):
        return len(self.img_files)





class GTADataset(data.Dataset):
    def __init__(self,train_or_val):
        path="./data/GTA5/{}.txt".format(train_or_val)
        self.img_files_txt = np.loadtxt(path, dtype=str, delimiter="\n")
        self.img_files=[]
        self.mask_files=[]
        self.crop_size = (1024,512)
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for idx in range(len(self.img_files_txt)):
  
          img_file_name = self.img_files_txt[idx]
          self.img_files.append("./data/GTA5/images/{}".format(img_file_name))
          mask_file_name = img_file_name
          self.mask_files.append("./data/GTA5/labels/{}".format(mask_file_name)) 

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
    
        image = Image.open(img_path).convert('RGB')
        label = Image.open(mask_path)

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy()


    def __len__(self):
        return len(self.img_files)






def val(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou


def train(args, model,d_model, optimizer,d_optimizer, dataloader_train, dataloader_val, dataloader_target):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    scaler = amp.GradScaler()
    lambda_disc=0.001 #best results from paper



    if args.loss == 'crossentropy':
      loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)


    #discriminator loss
    bce_loss = torch.nn.BCEWithLogitsLoss()


    # labels for adversarial training
    source_label = 0
    target_label = 1
    
    max_miou = 0
    step = 0


    

    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        lr_d= adjust_learning_rate_D(d_optimizer, i_iter=epoch,learning_rate_d=args.learning_rate_D, num_steps=args.num_epochs, power=0.9)

        model.train()
        d_model.train()

        target_Dataset_iter=iter(dataloader_target) #creates iterable object for the  TARGET dataset


        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data_src, label) in enumerate(dataloader_train): #dataloader_train should be SOURCE DOMAIN

            data_targ = next(target_Dataset_iter) #load next target dataset image
            data_targ = data_targ.cuda()


            # train GENERATOR

            # don't accumulate grads in Discriminator, as we are now training the GENERATOR
            for param in d_model.parameters():
                param.requires_grad = False

            
            data_src = data_src.cuda()
            label = label.long().cuda()

            optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            
            with amp.autocast(): #should automatically cast to float16 or float32 when needed
                #calcualte segmentation loss
                output, output_sup1, output_sup2 = model(data_src)
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss_seg = loss1 + loss2 + loss3 

                scaler.scale(loss_seg).backward()

                #calculate discriminator loss
                output_targ, output_sup1_targ, output_sup2_targ = model(data_targ)
                disc_output=d_model(F.softmax(output_targ))
                loss_disc_raw = bce_loss(disc_output, Variable(torch.FloatTensor(disc_output.data.size()).fill_(source_label)).cuda())

                """
                DISCRIMINATOR LOSS EXPLANATION:
                This loss function will compare the predicted image with a tensor which will always classify as "Source". 
                So, if the discriminator says the image comes from source, the loss goes down.
                As we know, we are only giving target images to the discriminator, so if the discriminator says the image comes from source, it means we have successfully fooled it,
                therefore, the loss goes down.
                """

                loss_disc = loss_disc_raw*lambda_disc

                scaler.scale(loss_disc).backward()

            #Train D

            # bring back requires_grad
            for param in d_model.parameters():
                param.requires_grad = True
            
            with amp.autocast():
                #detach the predictions from generator, we are in discriminator territory now
                src_out=output.detach()
                trg_out=output_targ.detach()

                #train with the source image
                disc_output=d_model(F.softmax(src_out))
                loss_disc_src = bce_loss(disc_output, Variable(torch.FloatTensor(disc_output.data.size()).fill_(source_label)).cuda())
                scaler.scale(loss_disc_src).backward()

                #train with target image
                disc_output=d_model(F.softmax(trg_out))
                loss_disc_trg = bce_loss(disc_output, Variable(torch.FloatTensor(disc_output.data.size()).fill_(target_label)).cuda())
                scaler.scale(loss_disc_trg).backward()               
                
            scaler.step(optimizer)
            scaler.step(d_optimizer)
            scaler.update()
            
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss_seg)
            step += 1
            writer.add_scalar('loss_step', loss_seg, step)
            loss_record.append(loss_seg.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))





        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, 'latest_dice_loss.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os 
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="Cityscapes", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, crossentropy')
    parser.add_argument('--learning_rate_D', type=float, default=1e-4, help='Base learning rate for discriminator.')


    args = parser.parse_args(params)

  
    # Define here your dataloaders
    
    #GTA DATALOADER (SOURCE)
    srcset_train=GTADataset(train_or_val='train')
    dataloader_train = DataLoader(srcset_train,batch_size=args.batch_size)
    
    #CITYSCAPES DATALOADER (TARGET)
    targset_train=target_Dataset(train_or_val='train')
    dataloader_target= DataLoader(targset_train,batch_size=args.batch_size)

    dataset_val = SegDataset(train_or_val='val')
    dataloader_val = DataLoader(dataset_val,batch_size=1)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    

    #discriminator model
    #d_model = FCDiscriminator(num_classes=args.num_classes)
    d_model = Light_Discriminator(num_classes=args.num_classes)

    if torch.cuda.is_available() and args.use_gpu:
            model = torch.nn.DataParallel(model).cuda()
            d_model = torch.nn.DataParallel(d_model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None


    #discriminator optimizer
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))


    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, d_model, optimizer,d_optimizer, dataloader_train, dataloader_val, dataloader_target)

    #val(args, model, dataloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '51',
        '--learning_rate', '2.5e-2',
        '--data', './data/...',
        '--num_workers', '8',
        '--num_classes', '19',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', './checkpoints_DA_LIGHT',
        '--context_path', 'resnet18', #'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',
        '--loss','crossentropy',
        '--learning_rate_D','1e-4',
        #'--validation_step','1' #FOR DEBUGGING ONLY
        #'--pretrained_model_path','./checkpoints_DA_LIGHT/latest_dice_loss.pth'


    ]
    main(params)
