from data.image_folder import get_custom_file_paths, natural_sort
from data.base_dataset import BaseDataset
import nibabel as nib
import random
from torchvision import transforms
from skimage.transform import resize
import os
import numpy as np
import torch
import torch.nn.functional as F
from collections.abc import Sequence

class SpatialRotation():
    def __init__(self, dimensions: Sequence, degree: Sequence = [90.]):
        
        self.dimensions = dimensions
        self.degree = degree

    def invert_permutation_numpy2(self, permutation):
        # https://stackoverflow.com/a/55737198
        inv = np.empty_like(permutation)
        inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
        return inv

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.dimensions[0], Sequence):
            dim1, dim2 = random.choice(self.dimensions)
        else:
            dim1, dim2 = self.dimensions
        permutation = [i for i in range(x.dim()) if (i != dim1 and i != dim2)] + [dim1, dim2]
        reverse_permuation = list(self.invert_permutation_numpy2(permutation))
        x = transforms.functional.rotate(x.permute(permutation), random.choice(self.degree)).permute(reverse_permuation)
        return x

class MRIDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.A_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, opt.phase + 'T1'), 't1.nii.gz'))
        self.B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, opt.phase + 'T2'), 't2.nii.gz'))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        transformations = [
            transforms.Lambda(lambda x: x[48:240,80:240,36:260]), # 192x160x224
            # transforms.Lambda(lambda x: resize(x, (64,64,75), order=1, anti_aliasing=True)),
            transforms.Lambda(lambda x: resize(x, (96,80,112), order=1, anti_aliasing=True)),
            transforms.Lambda(lambda x: self.toGrayScale(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            transforms.Lambda(lambda x: x.unsqueeze(0)),
            transforms.Lambda(lambda x: self.center(x, opt.mean, opt.std)),
            # transforms.Lambda(lambda x: F.pad(x, (0,1,0,0,0,0), mode='constant', value=0)),
        ]

        if(opt.phase == 'train'):
            transformations += [
                SpatialRotation([(1,2), (1,3), (2,3)], [0,90,180,270]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]
        else:
            transformations += [SpatialRotation((1,2))]
        self.transform = transforms.Compose(transformations)

    def toGrayScale(self, x):
        x_min = np.amin(x)
        x_max = np.amax(x)
        x = (x - x_min) / x_max * 255.
        return x

    def center(self, x, mean, std):
        return (x - mean) / std

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = np.array(nib.load(A_path).get_fdata())
        B_img = np.array(nib.load(B_path).get_fdata())
        A = self.transform(A_img)
        B = self.transform(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
        
        
