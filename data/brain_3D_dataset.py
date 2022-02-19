from abc import ABC
from typing import List
from data.base_dataset import BaseDataset
import nibabel as nib
import random
from torchvision import transforms
import numpy as np
import torch
from models.networks import setDimensions
from data.data_augmentation_3D import *
import torchio as tio

class brain3DDataset(BaseDataset, ABC):
    def __init__(self, opt, A_paths: List[list], B_paths: list):
        self.A_paths = A_paths
        self.B_paths = B_paths
        super().__init__(opt)
        self.A_size = len(self.A_paths[0])
        self.B_size = len(self.B_paths)
        setDimensions(3)
        opt.input_nc = len(A_paths)
        opt.output_nc = 1

        transformations = [
            transforms.Lambda(lambda x: getBetterOrientation(x, "IPL")),
            transforms.Lambda(lambda x: np.array(x.get_fdata())[np.newaxis, ...]),
            # transforms.Lambda(lambda x: x[:,24:168,18:206,8:160]),
            # transforms.Lambda(lambda x: resize(x, (x.shape[0],96,80,112), order=1, anti_aliasing=True)),
            transforms.Lambda(lambda x: toGrayScale(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float16 if opt.amp else torch.float32)),
            PadIfNecessary(3),
        ]

        if opt.phase == 'train':
            self.updateTransformations += [
                # SpatialRotation([(1,2), (1,3), (2,3)], [*[0]*12,1,2,3], auto_update=False), # With a probability of approx. 51% no rotation is performed
                # SpatialFlip(dims=(1,2,3), auto_update=False)
            ]
        transformations += self.updateTransformations
        self.transform = transforms.Compose(transformations)

        self.spatialTransforms = [
            tio.RandomAffine(scales=0.1, degrees=10),
            tio.Lambda(lambda x: x.to(dtype=torch.float16 if opt.amp else torch.float32)),
        ]
        if opt.amp:
            self.spatialTransforms.append(tio.Lambda(lambda x: x.half()))
        self.spatialTransforms = tio.Compose(self.spatialTransforms)

        self.styleTransforms = [
            ColorJitter3D(brightness_min_max=(0.8,1.2), contrast_min_max=(0.8,1.2)),
            # RandomNoise(std=(0.02,0.021)),
            # RandomBiasField([0,0.5]),
            # RandomBlur([0,2]),
            # ColorJitter3D(brightness_min_max=(0.7,1.3), contrast_min_max=(0.7,1.3)),
        ]
        self.styleTransforms = transforms.Compose(self.styleTransforms)

    def __getitem__(self, index):
        Ai_paths = [paths[index % self.A_size] for paths in self.A_paths]
        A_imgs: List[nib.Nifti1Image] = [nib.load(Ai_path) for Ai_path in Ai_paths]
        affine = A_imgs[0].affine

        if self.opt.paired:   # make sure index is within then range
            index_B = index % self.B_size
            B_path = self.B_paths[index_B]
            B_img = nib.load(B_path)
        else:
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            B_img = nib.load(B_path)

        Ai = [self.transform(img) for img in A_imgs]
        A = torch.concat(Ai, dim=0)
        B = self.transform(B_img)
        if self.opt.phase == 'train':
            AB = torch.concat((A,B), dim=0)
            AB = self.spatialTransforms(AB.float())
            A,B = AB[:len(A)], AB[-1:]
            A = self.styleTransforms(A)
        return {'A': A, 'B': B, 'affine': affine, 'axis_code': "IPL", 'A_paths': Ai_paths[0], 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return 10#max(self.A_size, self.B_size)
