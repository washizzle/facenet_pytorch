#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import pandas as pd
import time
from PIL import Image


# In[ ]:


verbosity     = 1
which_dataset = 3


# In[ ]:


if   which_dataset == 0:
    root_dir = "~/facenet_pytorch/datasets/vggface2_train/aligned"
elif which_dataset == 1:
    root_dir = "~/facenet_pytorch/datasets/vggface2_test/aligned"
elif which_dataset == 2:
    root_dir = "~/facenet_pytorch/datasets/lfw"
elif which_dataset == 3:
    root_dir = "./CASIA/CASIA-maxpy-clean"
else:
    root_dir = "/run/media/hoosiki/WareHouse2/home/mtb/datasets/my_pictures/my_pictures_mtcnnpy_182"


# In[ ]:


files = glob.glob(root_dir+"/*/*")


# In[ ]:


time0 = time.time()
df = pd.DataFrame()
print("The number of files: ", len(files))
for idx, file in enumerate(files):
    if idx%10000 == 0:
        print("[{}/{}]".format(idx, len(files)-1))
    '''    
    try:
        img = Image.open(file) # open the image file
        img.verify() # verify that it is, in fact, an image
    except (IOError, SyntaxError) as e:
        if verbosity == 1:
            print('Bad file:', file) # print out the names of corrupt files
        pass
    else:
        face_id    = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))
        df = df.append({'id': face_id, 'name': face_label}, ignore_index = True)
    '''    
    face_id    = os.path.basename(file).split('.')[0]
    face_label = os.path.basename(os.path.dirname(file))
    df = df.append({'id': face_id, 'name': face_label}, ignore_index = True)


# In[ ]:


df = df.sort_values(by = ['name', 'id']).reset_index(drop = True)


# In[ ]:


df['class'] = pd.factorize(df['name'])[0]


# In[ ]:


if which_dataset == 0:
    df.to_csv("train_vggface2.csv", index = False)
elif which_dataset == 1:
    df.to_csv("test_vggface2.csv", index = False)
elif which_dataset == 2:
    df.to_csv("valid_lfw.csv", index = False)
elif which_dataset == 3:
    df.to_csv("CASIA.csv", index = False)
else:
    df.to_csv("my_pictures.csv", index = False)

elapsed_time = time.time() - time0
print("elapsted time: ", elapsed_time//3600, "h", elapsed_time%3600//60, "m")


# In[ ]:


df

