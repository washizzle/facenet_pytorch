#!/usr/bin/env python
# coding: utf-8

import os
import glob
import pandas as pd
import time
from PIL import Image
from pathlib import Path

# Omniglot has another layer of folders which thus requires a slightly different way of dealing with 
#creation of csvs, which is why we made it possible to choose whether we are dealing with such a dataset 
#or not

which_dataset = 6
# omniglot, all images are put into 1 folder
omniglot_1_folder = True
# (UNTESTED) omniglot, separate (alphabet) folders. Having this on True while omniglot_1_folder is also 
#True causes it to behave as if omniglot_separate_folders = False.         
omniglot_separate_folders = False       

if   which_dataset == 0:
    root_path = "~/facenet_pytorch/datasets/vggface2_train/aligned/"
elif which_dataset == 1:
    root_path = "~/facenet_pytorch/datasets/vggface2_test/aligned/"
elif which_dataset == 2:
    root_path = "/lustre2/0/wsdarts/jpg_datasets/lfw_aligned/"
elif which_dataset == 3:
    root_path = "/lustre2/0/wsdarts/jpg_datasets/CASIA_aligned/"
elif which_dataset == 4:
    root_path = "./omniglot/python/images_background/"
elif which_dataset == 5:
    root_path = "/lustre2/0/wsdarts/jpg_datasets/omniglot_1_folder_splits/train"
elif which_dataset == 6:
    root_path = "/lustre2/0/wsdarts/jpg_datasets/omniglot_1_folder_splits/val"
else:
    root_path = "/run/media/hoosiki/WareHouse2/home/mtb/datasets/my_pictures/my_pictures_mtcnnpy_182"

time0 = time.time()
df = pd.DataFrame()
#dfs = {}

if omniglot_1_folder:
    files = glob.glob(root_path+"/*/*/*")
    print("The number of files: ", len(files))
    for idx, file in enumerate(files):
        if idx%10000 == 0:
            print("[{}/{}]".format(idx, len(files)-1))
            
        image_id    = os.path.basename(file).split('.')[0]
        character_name = os.path.basename(os.path.dirname(file))
        alphabet_name = os.path.basename(os.path.dirname(os.path.dirname(file)))
        name = alphabet_name + "-" + character_name
        df = df.append({'id': image_id, 'name': name}, ignore_index = True)
    
    df = df.sort_values(by = ['name', 'id']).reset_index(drop = True)
    df['class'] = pd.factorize(df['name'])[0]

elif omniglot_separate_folders:
    files = glob.glob(root_path+"/*/*/*")
    print("The number of files: ", len(files))
    for alphabet_dir in os.listdir(root_path):
        alphabet_path = Path(root_path+alphabet_dir)
        if os.path.isdir(alphabet_path):
            alphabet_df = pd.DataFrame()
            files = glob.glob(alphabet_path+"/*/*")
            print("The number of files: ", len(files))
            for idx, file in enumerate(files):
                if idx%10000 == 0:
                    print("[{}/{}]".format(idx, len(files)-1))
                face_id    = os.path.basename(file).split('.')[0]
                face_label = os.path.basename(os.path.dirname(file))
                alphabet_df = alphabet_df.append({'id': face_id, 'name': face_label}, ignore_index = True)

            alphabet_df = alphabet_df.sort_values(by = ['name', 'id']).reset_index(drop = True)
            alphabet_df['class'] = pd.factorize(alphabet_df['name'])[0]
            #dfs[alphabet_dir] = alphabet_df
            df.to_csv("./omniglot_alphabet_csvs/"+alphabet_dir+".csv", index = False)
else:
    files = glob.glob(root_path+"/*/*")

    print("The number of files: ", len(files))
    for idx, file in enumerate(files):
        if idx%10000 == 0:
            print("[{}/{}]".format(idx, len(files)-1))
        face_id    = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))
        df = df.append({'id': face_id, 'name': face_label}, ignore_index = True)

    df = df.sort_values(by = ['name', 'id']).reset_index(drop = True)
    df['class'] = pd.factorize(df['name'])[0]


if which_dataset == 0:
    df.to_csv("train_vggface2.csv", index = False)
elif which_dataset == 1:
    df.to_csv("test_vggface2.csv", index = False)
elif which_dataset == 2:
    df.to_csv("valid_lfw.csv", index = False)
elif which_dataset == 3:
    df.to_csv("CASIA.csv", index = False)
elif which_dataset == 4:
    df.to_csv("omniglot_1_folder.csv", index = False)
elif which_dataset == 5:
    df.to_csv("omniglot_train_1_folder.csv", index = False)
elif which_dataset == 6:
    df.to_csv("omniglot_val_1_folder.csv", index = False)
else:
    df.to_csv("my_pictures.csv", index = False)

elapsed_time = time.time() - time0
print("elapsted time: ", elapsed_time//3600, "h", elapsed_time%3600//60, "m")

