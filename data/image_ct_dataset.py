from data.image_folder import get_custom_file_paths, natural_sort
from data.base_dataset import BaseDataset
from PIL import Image
import random
from torchvision import transforms
import os
import numpy as np
from data.mri_dataset import PadIfNecessary, SpatialFlip, SpatialRotation
from models.networks import setDimensions

class ImageCTDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        setDimensions(2, opt.bayesian)
        
        self.A_paths = natural_sort(get_custom_file_paths(os.path.join(self.opt.dataroot, 'ct', self.opt.phase), '.png'))
        self.B_paths = natural_sort(get_custom_file_paths(os.path.join(self.opt.dataroot, 'mri', self.opt.phase), '.png'))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.transformations = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.type(opt.precision)),
            PadIfNecessary(opt.n_downsampling),
            transforms.Lambda(lambda x: self.center(x, opt.mean, opt.std)),
        ]

        if(opt.phase == 'train'):
            self.transformations += [
                SpatialRotation([(1,2)], [0,1,2,3], auto_update=False),
                SpatialFlip(dims=(1,2), auto_update=False),
            ]
        else:
            self.transformations += [SpatialRotation([(1,2)])]
        self.transform = transforms.Compose(self.transformations)

    def center(self, x, mean, std):
        return (x - mean) / std

    def updateDataAugmentation(self):
        for t in self.transformations[-2:]:
            t.update()

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.paired:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = np.array(Image.open(A_path), dtype=np.float32)
        B_img = np.array(Image.open(B_path), dtype=np.float32)

        if self.opt.paired:
            AB_img = np.stack([A_img,B_img], -1)
            AB = self.transform(AB_img)
            A = AB[0:1]
            B = AB[1:2]
        else:
            A = self.transform(A_img)
            B = self.transform(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
        
        
