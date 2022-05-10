from data.image_folder import get_custom_file_paths, natural_sort
from data.base_dataset import BaseDataset
from PIL import Image
import random
from torchvision import transforms
import os
import torch
import numpy as np
from data.data_augmentation_3D import PadIfNecessary, SpatialRotation, toGrayScale, ColorJitter3D, RandomBiasField, RandomBlur, RandomNoise
from models.networks import setDimensions

class ImageCTDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        setDimensions(2)
        
        self.A_paths = natural_sort(get_custom_file_paths(os.path.join(self.opt.dataroot, 'mri', self.opt.phase), '.png'))
        self.B_paths = natural_sort(get_custom_file_paths(os.path.join(self.opt.dataroot, 'ct', self.opt.phase), '.png'))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.surpress_registration_artifacts = False

        self.transformations = [
            # transforms.Lambda(lambda x: toGrayScale(x)),
            transforms.Lambda(lambda x: x / 255.),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.type(torch.float16 if opt.amp else torch.float32)),
            SpatialRotation([(1,2)]),
            PadIfNecessary(3)
        ]

        self.styleTransforms = [
            RandomNoise(std=(0., 0.02)),
            RandomBiasField([0, 0.4]),
            RandomBlur([0, 0.5]),
            ColorJitter3D(brightness_min_max=(0.9, 1.1), contrast_min_max=(0.9, 1.1)),
        ]
        self.styleTransforms = transforms.Compose(self.styleTransforms)

        self.transform = transforms.Compose(self.transformations)
        # self.colorJitter = ColorJitterSphere3D((0.3, 1.5), (0.3,1.5), sigma=0.5, dims=2)

    def center(self, x, mean, std):
        return (x - mean) / std

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.paired:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = np.array(Image.open(A_path), dtype=np.float32)
        B_img = np.array(Image.open(B_path), dtype=np.float32)

        if self.surpress_registration_artifacts and self.opt.direction=="AtoB" and self.opt.phase == 'train':
            if self.opt.paired:
                registration_artifacts_idx = B_img==127
            else:
                registration_artifacts_idx = np.array(Image.open(self.B_paths[index % self.B_size]), dtype=np.float32) == 127
            registration_artifacts_idx = self.transform(1- registration_artifacts_idx*1.)
            B_img[B_img==127] = 0

        if self.opt.paired:
            AB_img = np.stack([A_img,B_img], -1)
            AB = self.transform(AB_img)
            A = AB[0:1]
            B = AB[1:2]
        else:
            A = self.transform(A_img)
            B = self.transform(B_img)
        if self.opt.phase == 'train' and self.opt.direction=='AtoB':
            A = self.styleTransforms(A)
        data = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        if self.surpress_registration_artifacts and self.opt.direction=="AtoB" and self.opt.phase == 'train':
            data['registration_artifacts_idx'] = registration_artifacts_idx
        return data

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
        
        
