from data.image_folder import get_custom_file_paths, natural_sort
from data.base_dataset import BaseDataset
from data.data_augmentation_3D import PadIfNecessary, SpatialFlip, SpatialRotation, ColorJitter3D
import nibabel as nib
import random
from torchvision import transforms
import os
import numpy as np
import torch
from models.networks import setDimensions

class MRIDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.A_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'mri', opt.phase), '.nii.gz'))
        self.B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'ct', opt.phase), '.nii.gz'))
        self.surpress_registration_artifacts = True
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        setDimensions(3, opt.bayesian)
        opt.no_antialias = True
        opt.no_antialias_up = True

        self.transformations = [
            transforms.Lambda(lambda x: self.toGrayScale(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float16 if opt.amp else torch.float32)),
            PadIfNecessary(3),
        ]

        if(opt.phase == 'train'):
            self.transformations += [
                SpatialRotation([(1,2), (1,3), (2,3)], [0,1,2,3], auto_update=False),
                SpatialFlip(dims=(1,2,3), auto_update=False),
            ]
        else:
            self.transformations += [SpatialRotation([(1,2)])]
        self.transform = transforms.Compose(self.transformations)
        self.colorJitter=ColorJitter3D(brightness_min_max=(0.3, 1.5), contrast_min_max=(0.3, 1.5))

    def toGrayScale(self, x):
        x_min = np.amin(x)
        x_max = np.amax(x) - x_min
        x = (x - x_min) / x_max
        return x

    def center(self, x, mean, std):
        return (x - mean) / std

    def updateDataAugmentation(self):
        for t in self.transformations[-2:]:
            t.update()

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # A_path = self.A_paths[102]
        A_img = np.array(nib.load(A_path).get_fdata())
        if self.opt.paired:   # make sure index is within then range
            index_B = index % self.B_size
            B_path = self.B_paths[index_B]
            B_img = np.array(nib.load(B_path).get_fdata())
        else:
            while True: # Prevent big pairs
                index_B = random.randint(0, self.B_size - 1)
                # index_B = 3
                B_path = self.B_paths[index_B]
                B_img = np.array(nib.load(B_path).get_fdata())
                if np.prod(B_img.shape) + np.prod(A_img.shape) < 20000000:
                    break
                else:
                    print('[WARNING]: skipped A: {0} with B:{1}'.format(A_path, B_path))
        if self.opt.paired and self.surpress_registration_artifacts is True:
            A_min = np.min(A_img)
            B_min = np.min(B_img)
            A_img[B_img==0] = A_min
            B_img[B_img==0] = B_min
        
        A = self.transform(A_img[np.newaxis, ...])
        if(self.opt.phase == 'train'):
            A = self.colorJitter(A)
        B = self.transform(B_img[np.newaxis, ...])
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
        
        
