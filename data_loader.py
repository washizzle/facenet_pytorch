import os
import traceback
import numpy as np
import pandas as pd
from skimage import io
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import image_pb2
from pathlib import Path

class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, csv_name, num_triplets, format, transform = None):
        
        self.root_dir          = root_dir
        self.df                = pd.read_csv(csv_name, names=['id', 'name', 'class'], 
                   dtype={'id': str, 'name': str, 'class': str})
        self.num_triplets      = num_triplets
        self.transform         = transform
        self.training_triplets = self.generate_triplets(self.df, self.num_triplets)
        self.format            = format
    
    def get_image(self, person, image):
        for person_file in os.listdir(self.root_dir):
            #print("hit line 25")
            if os.path.isfile(Path(self.root_dir+person_file)) and person_file == person+".pid":
                #print("hit line 27")
                f = open(Path(self.root_dir+person_file), "rb")
                p = image_pb2.Person()
                p.ParseFromString(f.read())
                f.close()
                for i in p.images:
                    #print("hit line 33")
                    if i.name == image:
                        #print(i.name)
                        return i
    
    @staticmethod
    def generate_triplets(df, num_triplets):
        #dictionary to keep track of which images belong to which "faces":
        def make_dictionary_for_face_class(df):

            '''
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            '''
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append(df.iloc[idx, 0])
            return face_classes
        
        triplets    = []
        classes     = df['class'].unique()
        face_classes = make_dictionary_for_face_class(df)
         
        for _ in range(num_triplets):

            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''
        
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)
            
            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size = 2, replace = False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))
            
            triplets.append([face_classes[pos_class][ianc], face_classes[pos_class][ipos], face_classes[neg_class][ineg], 
                             pos_class, neg_class, pos_name, neg_name])
        
        return triplets
    
    def __getitem__(self, idx):
        keep = {}
        try:
            anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]
            
            anc_img   = os.path.join(self.root_dir, str(pos_name), str(anc_id) + self.format)
            pos_img   = os.path.join(self.root_dir, str(pos_name), str(pos_id) + self.format)
            neg_img   = os.path.join(self.root_dir, str(neg_name), str(neg_id) + self.format)
            
            #print("line 107 anc_img: ", anc_img)
            keep = {anc_img, pos_img, neg_img}
            
            anc_id = str(anc_id) + self.format
            pos_id = str(pos_id) + self.format
            neg_id = str(neg_id) + self.format
            
            #print("test", pos_name, anc_id, self.get_image(pos_name, anc_id))
            
            anc_img = self.get_image(pos_name, anc_id).contents
            pos_img = self.get_image(pos_name, pos_id).contents
            neg_img = self.get_image(neg_name, neg_id).contents
            
            #anc_img = self.deserialize_image(anc_img)
            #pos_img = self.deserialize_image(pos_img)
            #neg_img = self.deserialize_image(neg_img)

            anc_img   = io.imread(anc_img, plugin='imageio')
            pos_img   = io.imread(pos_img, plugin='imageio')
            neg_img   = io.imread(neg_img, plugin='imageio')

            pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
            neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))
            
            #print("anc_img: ", anc_img)
            
            sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class, 'neg_class': neg_class}

            if self.transform:
                sample['anc_img'] = self.transform(sample['anc_img'])
                sample['pos_img'] = self.transform(sample['pos_img'])
                sample['neg_img'] = self.transform(sample['neg_img'])
        except:
            print("traceback: ", traceback.format_exc())
            sample = {'exception': True}
            print("exception occurred: ", keep)
        return sample
    
    
    def __len__(self):
        
        return len(self.training_triplets)
    

def get_dataloader(train_root_dir,     valid_root_dir, 
                   train_csv_name,     valid_csv_name, 
                   num_train_triplets, num_valid_triplets, 
                   batch_size,         num_workers,
                   train_format,       valid_format):
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])]),
        'valid': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])}

    face_dataset = {
        'train' : TripletFaceDataset(root_dir     = train_root_dir,
                                     csv_name     = train_csv_name,
                                     num_triplets = num_train_triplets,
                                     format       = train_format,
                                     transform    = data_transforms['train']),
        'valid' : TripletFaceDataset(root_dir     = valid_root_dir,
                                     csv_name     = valid_csv_name,
                                     num_triplets = num_valid_triplets,
                                     format       = valid_format,
                                     transform    = data_transforms['valid'])}

    dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset[x], batch_size = batch_size, shuffle = False, num_workers = num_workers)
        for x in ['train', 'valid']}
    #dataloaders = {
    #    x: face_dataset[x] 
    #    for x in ['train', 'valid']}
    data_size = {x: len(face_dataset[x]) for x in ['train', 'valid']}

    return dataloaders, data_size