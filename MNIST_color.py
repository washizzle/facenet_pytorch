from PIL import Image
import numpy as np
import torchvision
import cv2

class MNISTColor(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, dataset_depth=1):
        super().__init__(root, train, transform, target_transform, download)
        self.dataset_depth = dataset_depth

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        
        img = Image.fromarray(img.numpy(), mode='L')
        if self.dataset_depth == 1:
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        sample = {"image": img, "class": target}
        return sample