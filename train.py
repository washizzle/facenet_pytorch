import os
import traceback
import numpy as np
from datetime import datetime
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.modules.distance import PairwiseDistance
import torchvision
from torchvision import transforms
from eval_metrics import evaluate, plot_roc
from utils import TripletLoss
from models import FaceNetModel
from data_loader import TripletFaceDataset, get_dataloader
from pathlib import Path
from shutil import copyfile
import pandas as pd

#val==valid==validation, these terms are used interchangeably.
#If start-epoch != 0 and/or pure_validation is given as argument, a load_pth_from needs to be defined.
#If no separate save_dir is defined through "save_pth_to_separate_dir", save_dir = load_pth_from.
#If save_pth_to_separate_dir is given as argument, a new directory new_dir is created (in the standard way) and the pth file from load_pth_from directory is copied to new_dir. log_dir = new_dir,


parser = argparse.ArgumentParser(description = 'Face Recognition using Triplet Loss')

parser.add_argument('--train_dataset_depth', default = 3, type = int,
                    help = 'Defines depth of the images in the train dataset. E.g. Grayscale = 1 and rgb = 3 ')
parser.add_argument('--val_dataset_depth', default = 3, type = int,
                    help = 'Defines depth of the images in the validation dataset. E.g. Grayscale = 1 and rgb = 3 ')                    
parser.add_argument('--train_format', type=str, 
                    help='Format of images for training set', default='.png')
parser.add_argument('--valid_format', type=str, 
                    help='Format of images for validation set', default='.jpg')
parser.add_argument('--logs_base_dir', type=str, 
                    help='Directory where to write event logs.', default='/lustre2/0/wsdarts/facenet_logs/')
parser.add_argument('--start-epoch', default = 0, type = int, metavar = 'SE',
                    help = 'start epoch (default: 0).')
parser.add_argument('--load_pth_from', default = './log/test/', type = str,
                    help = 'directory to load .pth from. Only used when doing pure validation or when --start-epoch > 0')
parser.add_argument('--save_pth_to_separate_dir', 
                    help='only important when load_pth_from is used. Defines whether the save_dir needs to be different than the one defined in load_pth_from.', action='store_true')                    
parser.add_argument('--pure_validation', 
                    help='Only do 1 epoch of validation. Requires .pth file to load from.', action='store_true')
parser.add_argument('--num-epochs', default = 200, type = int, metavar = 'NE',
                    help = 'number of epochs to train (default: 200)')
parser.add_argument('--num-classes', default = 10000, type = int, metavar = 'NC',
                    help = 'number of clases (default: 10000)')
parser.add_argument('--num-train-triplets', default = 10000, type = int, metavar = 'NTT',
                    help = 'number of triplets for training (default: 10000)')
parser.add_argument('--num-valid-triplets', default = 10000, type = int, metavar = 'NVT',
                    help = 'number of triplets for vaidation (default: 10000)')
parser.add_argument('--embedding-size', default = 128, type = int, metavar = 'ES',
                    help = 'embedding size (default: 128)')
parser.add_argument('--batch-size', default = 64, type = int, metavar = 'BS',
                    help = 'batch size (default: 128)')
parser.add_argument('--num-workers', default = 8, type = int, metavar = 'NW',
                    help = 'number of workers (default: 8)')
parser.add_argument('--learning-rate', default = 0.001, type = float, metavar = 'LR',
                    help = 'learning rate (default: 0.001)')
parser.add_argument('--margin', default = 0.5, type = float, metavar = 'MG',
                    help = 'margin (default: 0.5)')
parser.add_argument('--train-root-dir', default = '/run/media/hoosiki/WareHouse2/home/mtb/datasets/vggface2/test_mtcnnpy_182', type = str,
                    help = 'path to train root dir')
parser.add_argument('--valid-root-dir', default = '/run/media/hoosiki/WareHouse2/home/mtb/datasets/lfw/lfw_mtcnnpy_182', type = str,
                    help = 'path to valid root dir')
parser.add_argument('--train-csv-name', default = './datasets/test_vggface2.csv', type = str,
                    help = 'list of training images')
parser.add_argument('--valid-csv-name', default = './datasets/lfw.csv', type = str,
                    help = 'list of validation images')
parser.add_argument('--train_torchvision', 
                    help='Use torchvision dataset for training.', action='store_true')
parser.add_argument('--val_torchvision',
                    help='Use torchvision dataset for validation.', action='store_true')
parser.add_argument('--pure_training',
                    help='Skip validation for each epoch.', action='store_true')
parser.add_argument('--pure_valid_six_time', 
                    help='run validation six times, one for each omniglot alphabet in the array', action='store_true')                    
                    
#global variables
args    = parser.parse_args()
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
l2_dist = PairwiseDistance(2)
num_epochs = args.num_epochs

#add logging and checkpoints
subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
save_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
log_dir = save_dir

if args.start_epoch > 0 or args.pure_validation:
    load_dir = Path(args.load_pth_from)
    if not args.save_pth_to_separate_dir and not args.pure_validation:
        save_dir = load_dir
        log_dir = load_dir
    else:
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)
        #copy pth file to the new directory
        pth_file_src = Path('{}/checkpoint_epoch{}.pth'.format(load_dir, args.start_epoch-1))
        pth_file_dest = Path('{}/checkpoint_epoch{}.pth'.format(log_dir, args.start_epoch-1))
        copyfile(pth_file_src, pth_file_dest)
        print("Copied .pth file to ", pth_file_dest)
    if args.pure_validation:
        num_epochs = 1
print("log_dir: ", log_dir)
print("save_dir: ", save_dir)

def main():
    print("Start date and time: ", time.asctime(time.localtime(time.time())))
    print("arguments: ", args)
    
    #load model
    model     = FaceNetModel(embedding_size = args.embedding_size, num_classes = args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = args.learning_rate/4, gamma = 0.1)
    
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    
    if args.start_epoch != 0:
        checkpoint = torch.load('{}/checkpoint_epoch{}.pth'.format(log_dir, args.start_epoch-1))
        model.load_state_dict(checkpoint['state_dict'])
    
    #if args.torchvision_dataset:   #create csv
    #    dataset_path = os.path.join(os.path.expanduser(datasets_path), args.dataset_name)
    #    dataloaders, dataset_sizes, num_classes = load_torchvision_data(args.dataset_name, dataset_path, data_transforms, dataset_depth)
    array = [
                {"dir": "/lustre2/0/wsdarts/datasets/omniglot_multiple_folders_split/val/Japanese_(hiragana)/", "csv":"/nfs/home4/mhouben/facenet_pytorch/datasets/omniglot_alphabet_csvs/val/Japanese_(hiragana).csv"},
                {"dir": "/lustre2/0/wsdarts/datasets/omniglot_multiple_folders_split/val/Japanese_(katakana)/", "csv":"/nfs/home4/mhouben/facenet_pytorch/datasets/omniglot_alphabet_csvs/val/Japanese_(katakana).csv"},
                {"dir": "/lustre2/0/wsdarts/datasets/omniglot_multiple_folders_split/val/Korean/",              "csv":"/nfs/home4/mhouben/facenet_pytorch/datasets/omniglot_alphabet_csvs/val/Korean.csv"             },
                {"dir": "/lustre2/0/wsdarts/datasets/omniglot_multiple_folders_split/val/Latin/",               "csv":"/nfs/home4/mhouben/facenet_pytorch/datasets/omniglot_alphabet_csvs/val/Latin.csv"              },
                {"dir": "/lustre2/0/wsdarts/datasets/omniglot_multiple_folders_split/val/N_Ko/",                "csv":"/nfs/home4/mhouben/facenet_pytorch/datasets/omniglot_alphabet_csvs/val/N_Ko.csv"               },
                {"dir": "/lustre2/0/wsdarts/datasets/omniglot_multiple_folders_split/val/Greek/",               "csv":"/nfs/home4/mhouben/facenet_pytorch/datasets/omniglot_alphabet_csvs/val/Greek.csv"              }
    ]
    if args.pure_valid_six_time:
        for a in array:
            for epoch in range(args.start_epoch, num_epochs + args.start_epoch):
                t = time.time()
                if not args.pure_validation:
                    print(80 * '=')
                    print('Epoch [{}/{}]'.format(epoch, num_epochs + args.start_epoch - 1))
                # load data (every epoch)
                print("a[dir]", a["dir"])
                print("a[csv]", a["csv"])
                data_loaders, data_size = get_dataloader(args.train_root_dir,     a["dir"],
                                                         args.train_csv_name,     a["csv"],
                                                         args.num_train_triplets, args.num_valid_triplets, 
                                                         args.batch_size,         args.num_workers,
                                                         args.train_format,       args.valid_format,
                                                         args.train_dataset_depth,args.val_dataset_depth,
                                                         args.train_torchvision,  args.val_torchvision,
                                                         224,                     224,
                                                         args.pure_validation,    args.pure_training)
                # training and validation
                train_valid(model, optimizer, scheduler, epoch, data_loaders, data_size, t)
                print("duration of epoch ", epoch, ": ", time.time()-t, " seconds")
            print(80 * '=')
            print("End date and time: ", time.asctime(time.localtime(time.time())))
    else:
        for epoch in range(args.start_epoch, num_epochs + args.start_epoch):
            t = time.time()
            if not args.pure_validation:
                print(80 * '=')
                print('Epoch [{}/{}]'.format(epoch, num_epochs + args.start_epoch - 1))
            # load data (every epoch)
            data_loaders, data_size = get_dataloader(args.train_root_dir,     args.valid_root_dir,
                                                     args.train_csv_name,     args.valid_csv_name,
                                                     args.num_train_triplets, args.num_valid_triplets,   
                                                     args.batch_size,         args.num_workers,
                                                     args.train_format,       args.valid_format,
                                                     args.train_dataset_depth,args.val_dataset_depth,
                                                     args.train_torchvision,  args.val_torchvision,
                                                     224,                     224,
                                                     args.pure_validation,    args.pure_training)
            # training and validation
            train_valid(model, optimizer, scheduler, epoch, data_loaders, data_size, t)
            print("duration of epoch ", epoch, ": ", time.time()-t, " seconds")
        print(80 * '=')
        print("End date and time: ", time.asctime(time.localtime(time.time())))
    
def train_valid(model, optimizer, scheduler, epoch, dataloaders, data_size, start_time):
    
    for phase in ['train', 'valid']:
        if args.pure_validation and phase == 'train':
            continue
        if args.pure_training and phase == 'valid':
            continue
        labels, distances = [], []
        triplet_loss_sum  = 0.0

        if phase == 'train':
            scheduler.step()
            model.train()
        else:
            model.eval()

        #for batch_idx in range(0, data_size[phase], 1):
        for batch_idx, batch_sample in enumerate(dataloaders[phase]):
            #print("batch_idx:", batch_idx)
            try:
                #batch_sample = dataloaders[phase][batch_idx]
                if not 'exception' in batch_sample:
                    anc_img = batch_sample['anc_img'].to(device)
                    pos_img = batch_sample['pos_img'].to(device)
                    neg_img = batch_sample['neg_img'].to(device)
                
                    pos_cls = batch_sample['pos_class'].to(device)
                    neg_cls = batch_sample['neg_class'].to(device)
                                    
                    with torch.set_grad_enabled(phase == 'train'):
                    
                        # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                        anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)
                        #for i in anc_embed:
                        #    print(i.item())
                        if args.num_valid_triplets <= 100:
                            anc_embed_cpu = anc_embed.cpu()
                            pos_embed_cpu = pos_embed.cpu()
                            neg_embed_cpu = neg_embed.cpu()
                            pos_cls_cpu = pos_cls.cpu()
                            neg_cls_cpu = neg_cls.cpu()
                            pd.DataFrame([t.numpy() for t in anc_embed_cpu]).to_csv("./embeddings.csv", mode='a', header=None)
                            pd.DataFrame([t.numpy() for t in pos_embed_cpu]).to_csv("./embeddings.csv", mode='a', header=None)
                            pd.DataFrame([t.numpy() for t in neg_embed_cpu]).to_csv("./embeddings.csv", mode='a', header=None)
                            pd.DataFrame({'type': "anc", 'id': batch_sample['anc_id'], 'class': pos_cls_cpu, 'train_set': args.train_csv_name.split('.')[0], 'val_set': args.valid_csv_name.split('.')[0]}).to_csv("./embeddings_info.csv", mode='a', header=None)
                            pd.DataFrame({'type': "pos", 'id': batch_sample['pos_id'], 'class': pos_cls_cpu, 'train_set': args.train_csv_name.split('.')[0], 'val_set': args.valid_csv_name.split('.')[0]}).to_csv("./embeddings_info.csv", mode='a', header=None)
                            pd.DataFrame({'type': "neg", 'id': batch_sample['neg_id'], 'class': pos_cls_cpu, 'train_set': args.train_csv_name.split('.')[0], 'val_set': args.valid_csv_name.split('.')[0]}).to_csv("./embeddings_info.csv", mode='a', header=None)
                            
                        #print([t.size() for t in anc_embed])
                        # choose the hard negatives only for "training"
                        pos_dist = l2_dist.forward(anc_embed, pos_embed)
                        neg_dist = l2_dist.forward(anc_embed, neg_embed)
                    
                        all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
                        if phase == 'train':
                            hard_triplets = np.where(all == 1)
                            if len(hard_triplets[0]) == 0:
                                continue
                        else:
                            hard_triplets = np.where(all >= 0)
                        
                        anc_hard_embed = anc_embed[hard_triplets].to(device)
                        pos_hard_embed = pos_embed[hard_triplets].to(device)
                        neg_hard_embed = neg_embed[hard_triplets].to(device)
                    
                        anc_hard_img   = anc_img[hard_triplets].to(device)
                        pos_hard_img   = pos_img[hard_triplets].to(device)
                        neg_hard_img   = neg_img[hard_triplets].to(device)
                    
                        pos_hard_cls   = pos_cls[hard_triplets].to(device)
                        neg_hard_cls   = neg_cls[hard_triplets].to(device)
                    
                        anc_img_pred   = model.forward_classifier(anc_hard_img).to(device)
                        pos_img_pred   = model.forward_classifier(pos_hard_img).to(device)
                        neg_img_pred   = model.forward_classifier(neg_hard_img).to(device)
                    
                        triplet_loss   = TripletLoss(args.margin).forward(anc_hard_embed, pos_hard_embed, neg_hard_embed).to(device)
                
                        if phase == 'train':
                            optimizer.zero_grad()
                            triplet_loss.backward()
                            optimizer.step()
                    
                        dists = l2_dist.forward(anc_embed, pos_embed)
                        distances.append(dists.data.cpu().numpy())
                        labels.append(np.ones(dists.size(0))) 

                        dists = l2_dist.forward(anc_embed, neg_embed)
                        distances.append(dists.data.cpu().numpy())
                        labels.append(np.zeros(dists.size(0)))
                    
                        triplet_loss_sum += triplet_loss.item()
            except:
                #traceback.print_exc()
                print("traceback: ", traceback.format_exc())
                print("something went wrong with batch_idx: ", batch_idx, ", batch_sample:", batch_sample, ", neg_img size: ", batch_sample['neg_img'].shape, ", pos_img size: ", batch_sample['pos_img'].shape, ", anc_img size: ", batch_sample['anc_img'].shape)
  
        avg_triplet_loss = triplet_loss_sum / data_size[phase]
        labels           = np.array([sublabel for label in labels for sublabel in label])
        distances        = np.array([subdist for dist in distances for subdist in dist])
        
        nrof_pairs = min(len(labels), len(distances))
        if nrof_pairs >= 10:
            tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
            print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
            print('  {} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))
            duration = time.time()-start_time
       
            with open('{}/{}_log_epoch{}.txt'.format(log_dir, phase, epoch), 'w') as f:
                f.write(str(epoch)             + '\t' +
                        str(np.mean(accuracy)) + '\t' +
                        str(avg_triplet_loss)  + '\t' +
                        str(duration))
                
            if phase == 'train':
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict()},
                            '{}/checkpoint_epoch{}.pth'.format(save_dir, epoch))
            else:
                plot_roc(fpr, tpr, figure_name = '{}/roc_valid_epoch_{}.png'.format(log_dir, epoch))


if __name__ == '__main__':
    
    main()
