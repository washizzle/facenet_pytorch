import os
import traceback
import numpy as np
import pandas as pd
from skimage import io
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from PIL import Image

csv_name = "./CASIA_test_incorrect.csv"
file_name = "./CASIA_test2.csv"
df = pd.read_csv(csv_name, header=0, names=['id', 'name', 'class'], 
                            dtype={'id': str, 'name': str, 'class': int})      
df['class'] = df['class'] - 4233
#classes = df['class'].unique()
#df = df.fillna(-9999)
#df = df.astype(int)
df.to_csv(file_name, index = False)
print(file_name + " created.")