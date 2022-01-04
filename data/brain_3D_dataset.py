from data.image_folder import get_custom_file_paths, natural_sort
import nibabel as nib
import random
from torchvision import transforms
import os
import numpy as np
import torch
from models.networks import setDimensions
from data.mri_dataset import MRIDataset
from data.data_augmentation_3D import ColorJitter3D, PadIfNecessary, SpatialRotation, SpatialFlip

class brain3DDataset(MRIDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.A1_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 't1', opt.phase), 't1.nii.gz'))
        self.A2_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'flair', opt.phase), 'flair.nii.gz'))
        self.B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'dir', opt.phase), 'dir.nii.gz'))
        self.A_size = len(self.A1_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        setDimensions(3, opt.bayesian)
        opt.input_nc = 2
        opt.output_nc = 1

        transformations = [
            transforms.Lambda(lambda x: x[:,24:168,18:206,8:160]),
            # transforms.Lambda(lambda x: resize(x, (x.shape[0],96,80,112), order=1, anti_aliasing=True)),
            transforms.Lambda(lambda x: self.toGrayScale(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float16 if opt.amp else torch.float32)),
            PadIfNecessary(3),
        ]
        self.updateTransformations = []

        if(opt.phase == 'train'):
            self.updateTransformations += [
                SpatialRotation([(1,2), (1,3), (2,3)], [0,1,2,3], auto_update=False),
                SpatialFlip(dims=(1,2,3), auto_update=False),
                ColorJitter3D((0.3,1.5), (0.3,1,5))
            ]
        else:
            self.updateTransformations += [SpatialRotation([(1,2)], [2])]
            self.updateTransformations += [SpatialRotation([(1,3)], [1])]
        transformations += self.updateTransformations
        self.transform = transforms.Compose(transformations)

    def __getitem__(self, index):
        A1_path = self.A1_paths[index % self.A_size]  # make sure index is within then range
        A1_img = np.array(nib.load(A1_path).get_fdata())
        A2_path = self.A2_paths[index % self.A_size]  # make sure index is within then range
        A2_img = np.array(nib.load(A2_path).get_fdata())
        if self.opt.paired:   # make sure index is within then range
            index_B = index % self.B_size
            B_path = self.B_paths[index_B]
            B_img = np.array(nib.load(B_path).get_fdata())
        else:
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            B_img = np.array(nib.load(B_path).get_fdata())
        A1 = self.transform(A1_img[np.newaxis, ...])
        A2 = self.transform(A2_img[np.newaxis, ...])
        A = torch.concat((A1, A2), dim=0)
        B = self.transform(B_img[np.newaxis, ...])
        return {'A': A, 'B': B, 'A_paths': A1_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
