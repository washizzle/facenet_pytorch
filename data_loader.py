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
import cv2
from MNIST_color import MNISTColor
from PIL import Image

class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, csv_name, num_triplets, format, dataset_depth, use_torchvision=False, transform = None):
        
        self.root_dir          = root_dir
        self.df                = pd.read_csv(csv_name, header=0, names=['id', 'name', 'class'], 
                   dtype={'id': str, 'name': str, 'class': str})
        self.num_triplets      = num_triplets
        self.transform         = transform
        self.training_triplets = self.generate_triplets(self.df, self.num_triplets)
        self.dataset_depth     = dataset_depth
        self.format            = format
        self.use_torchvision   = use_torchvision
        self.dataset_name      = os.path.basename(os.path.dirname(root_dir)) #fix
        self.dataset           = None
        if use_torchvision:
            if self.dataset_name == 'mnist':
                train_val = os.path.basename(root_dir)
                self.dataset = MNISTColor(os.path.join(root_dir), train='train'==train_val,
                                transform=transform, target_transform=None,
                                download=True, dataset_depth=dataset_depth)
                    
    
    def get_image(self, person, image):
        for person_file in os.listdir(self.root_dir):
            #print("hit line 25")
            person_path = os.path.join(self.root_dir, person_file)
            if os.path.isfile(person_path) and person_file == person+".pid":
                #print("hit line 27")
                f = open(person_path, "rb")
                p = image_pb2.Person()
                p.ParseFromString(f.read())
                f.close()
                for i in p.images:
                    #print("hit line 33")
                    if i.name == image:
                        #print(i.name)
                        return i
        raise Exception("pid file not found. Variables: image: ", image, ", person: ", person)
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
            print("first idx print: ", idx)
            anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]
            
            #anc_img   = os.path.join(self.root_dir, str(pos_name), str(anc_id) + self.format)
            #pos_img   = os.path.join(self.root_dir, str(pos_name), str(pos_id) + self.format)
            #neg_img   = os.path.join(self.root_dir, str(neg_name), str(neg_id) + self.format)
            
            #print("line 107 anc_img: ", anc_img)
            anc_img, pos_img, neg_img = None, None, None
            if not self.use_torchvision:
                anc_id = str(anc_id) + self.format
                pos_id = str(pos_id) + self.format
                neg_id = str(neg_id) + self.format
                keep = {'anc_id': anc_id, 'pos_id': pos_id, 'neg_id': neg_id, 'pos_name': pos_name, 'neg_name': neg_name}
                
                anc_img = self.get_image(pos_name, anc_id).contents
                pos_img = self.get_image(pos_name, pos_id).contents
                neg_img = self.get_image(neg_name, neg_id).contents            

                anc_img   = io.imread(anc_img, plugin='imageio')
                pos_img   = io.imread(pos_img, plugin='imageio')
                neg_img   = io.imread(neg_img, plugin='imageio')
                print("anc_img.shape: ", anc_img.shape)
                
                if self.dataset_depth==1:
                    anc_img = cv2.cvtColor(anc_img, cv2.COLOR_GRAY2RGB)
                    pos_img = cv2.cvtColor(pos_img, cv2.COLOR_GRAY2RGB)
                    neg_img = cv2.cvtColor(neg_img, cv2.COLOR_GRAY2RGB)
                else:
                    if len(anc_img.shape) < 3:
                        anc_img = cv2.cvtColor(anc_img, cv2.COLOR_GRAY2RGB)
                    if len(pos_img.shape) < 3:
                        pos_img = cv2.cvtColor(pos_img, cv2.COLOR_GRAY2RGB)
                    if len(neg_img.shape) < 3:
                        neg_img = cv2.cvtColor(neg_img, cv2.COLOR_GRAY2RGB)
            #else if dataset is mnist    
            elif self.dataset_name == 'mnist':
                print("idx: ", idx)
                #print(", anc_id: ", anc_id)
                #print(", self.dataset.targets: ", self.dataset.targets)
                #print(", self.dataset.data: ", self.dataset.data.item())
                #print(", self.dataset.targets[anc_id]: ", self.dataset.targets.item()[anc_id])
                #print("self.dataset[int(anc_id)]: ", self.dataset[int(anc_id)])
                #print(", self.dataset.data[587]: ", self.dataset.data[587])
                #print(", self.dataset.data.shape: ", self.dataset.data.shape)
                #print(", self.dataset.data[anc_id].shape: ", self.dataset.data[anc_id].shape)
                #print(", self.dataset.data[anc_id]: ", self.dataset.data.item()[anc_id])
                anc_dict = self.dataset[int(anc_id)]
                pos_dict = self.dataset[int(pos_id)]
                neg_dict = self.dataset[int(neg_id)]
                anc_img = anc_dict["image"]
                pos_img = anc_dict["image"]
                neg_img = anc_dict["image"]
                #print("anc_img: ", anc_img)
            #    anc_img = Image.fromarray(anc_img.numpy(), mode='L')
            #    pos_img = Image.fromarray(pos_img.numpy(), mode='L')
            #    neg_img = Image.fromarray(neg_img.numpy(), mode='L')
            #    if self.dataset_depth == 1:
            #        anc_img = np.asarray(anc_img)
            #        pos_img = np.asarray(pos_img)
            #        neg_img = np.asarray(neg_img)                    
            
            #if self.dataset_depth==1:
            #    anc_img = cv2.cvtColor(anc_img, cv2.COLOR_GRAY2RGB)
            #    pos_img = cv2.cvtColor(pos_img, cv2.COLOR_GRAY2RGB)
            #    neg_img = cv2.cvtColor(neg_img, cv2.COLOR_GRAY2RGB)

            pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
            neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))
            
            #print("anc_img: ", anc_img)
            
            sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class, 'neg_class': neg_class}

            if not self.use_torchvision and self.transform:
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
                   train_format,       valid_format,
                   train_dataset_depth,val_dataset_depth,
                   train_torchvision,  val_torchvision,
                   train_input_size,   val_input_size,
                   pure_validation):
    data_transforms = {'train': None, 'valid': None}
    if train_torchvision: 
        data_transforms['train'] = transforms.Compose([
            transforms.ToPILImage(),#extra, not recommended to keep here
            transforms.RandomResizedCrop(train_input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transforms['train'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(train_input_size),#extra, not recommended to keep here
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        ])
    if val_torchvision:
        data_transforms['valid'] = transforms.Compose([
            transforms.ToPILImage(),#extra, not recommended to keep here
            transforms.Resize((val_input_size, val_input_size)),
            transforms.CenterCrop(val_input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transforms['valid'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((val_input_size, val_input_size)),#extra, not recommended to keep here
            transforms.CenterCrop(val_input_size),#extra, not recommended to keep here
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        ])
    #data_transforms = {
    #    'train': transforms.Compose([
    #        transforms.ToPILImage(),
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    #        ]),
    #    'valid': transforms.Compose([
    #        transforms.ToPILImage(),
    #        transforms.ToTensor(),
    #        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    #        ])}
    face_dataset = {
        'train': None, 
        'valid': TripletFaceDataset(root_dir     = valid_root_dir,
                                     csv_name     = valid_csv_name,
                                     num_triplets = num_valid_triplets,
                                     format       = valid_format,
                                     dataset_depth= val_dataset_depth,
                                     use_torchvision = val_torchvision,
                                     transform    = data_transforms['valid'])}
    data_size = {
        'train': None,
        'valid': len(face_dataset['valid'])}
    if not pure_validation:
        face_dataset['train'] = TripletFaceDataset(root_dir     = train_root_dir,
                                     csv_name     = train_csv_name,
                                     num_triplets = num_train_triplets,
                                     format       = train_format,
                                     dataset_depth= train_dataset_depth,
                                     use_torchvision = train_torchvision,
                                     transform    = data_transforms['train'])
        data_size['train'] = len(face_dataset['train'])
    #face_dataset = {
    #    'train' : TripletFaceDataset(root_dir     = train_root_dir,
    #                                 csv_name     = train_csv_name,
    #                                 num_triplets = num_train_triplets,
    #                                 format       = train_format,
    #                                 dataset_depth= train_dataset_depth,
    #                                 use_torchvision = train_torchvision,
    #                                 transform    = data_transforms['train']),
    #    'valid' : TripletFaceDataset(root_dir     = valid_root_dir,
    #                                 csv_name     = valid_csv_name,
    #                                 num_triplets = num_valid_triplets,
    #                                 format       = valid_format,
    #                                 dataset_depth= val_dataset_depth,
    #                                 use_torchvision = val_torchvision,
    #                                 transform    = data_transforms['valid'])}
    #data_size = {x: len(face_dataset[x]) for x in ['train', 'valid']}
    dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset[x], batch_size = batch_size, shuffle = False, num_workers = num_workers)
        for x in ['train', 'valid']}
    #dataloaders = {
    #    x: face_dataset[x] 
    #    for x in ['train', 'valid']}
    

    return dataloaders, data_size