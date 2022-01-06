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
        self.mri_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'mri', opt.phase), '.nii.gz'))
        self.ct_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'ct', opt.phase), '.nii.gz'))
        self.surpress_registration_artifacts = True
        self.mri_size = len(self.mri_paths)  # get the size of dataset A
        self.ct_size = len(self.ct_paths)  # get the size of dataset B
        setDimensions(3, opt.bayesian)
        opt.no_antialias = True
        opt.no_antialias_up = True

        transformations = [
            transforms.Lambda(lambda x: self.toGrayScale(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float16 if opt.amp else torch.float32)),
            PadIfNecessary(3),
        ]
        self.updateTransformations = []

        if(opt.phase == 'train'):
            self.updateTransformations += [
                SpatialRotation([(1,2), (1,3), (2,3)], [0,1,2,3], auto_update=False),
                SpatialFlip(dims=(1,2,3), auto_update=False),
                ColorJitter3D(brightness_min_max=(0.3, 1.5), contrast_min_max=(0.3, 1.5))
            ]
        else:
            self.updateTransformations += [SpatialRotation([(1,3)], [3])]
        transformations += self.updateTransformations
        self.mri_transform = transforms.Compose(transformations)
        self.ct_transform = transforms.Compose([
            transforms.Lambda(lambda x: (np.clip(x, -1000., 1000.) + 1000.) / 2000.),
            *transformations[1:-1 if opt.phase == 'train' else None]
        ])

    def toGrayScale(self, x):
        x_min = np.amin(x)
        x_max = np.amax(x) - x_min
        x = (x - x_min) / x_max
        return x

    def center(self, x, mean, std):
        return (x - mean) / std

    def updateDataAugmentation(self):
        for t in self.updateTransformations:
            t.update()

    def __getitem__(self, index):
        mri_path = self.mri_paths[index % self.mri_size]  # make sure index is within then range
        # mri_path = self.mri_paths[102]
        mri_img = np.array(nib.load(mri_path).get_fdata())
        if self.opt.paired:   # make sure index is within then range
            index_ct = index % self.ct_size
            ct_path = self.ct_paths[index_ct]
            ct_img = np.array(nib.load(ct_path).get_fdata())
        else:
            while True: # Prevent big pairs
                index_ct = random.randint(0, self.ct_size - 1)
                # index_B = 3
                ct_path = self.ct_paths[index_ct]
                ct_img = np.array(nib.load(ct_path).get_fdata())
                if np.prod(ct_img.shape) + np.prod(mri_img.shape) < 20000000:
                    break
                else:
                    print('[WARNING]: skipped A: {0} with B:{1}'.format(mri_path, ct_path))
        if self.surpress_registration_artifacts:
            if self.opt.paired:
                registration_artifacts_idx = ct_img==0
            else:
                registration_artifacts_idx = np.array(nib.load(self.ct_paths[index % self.ct_size]).get_fdata()) == 0
            registration_artifacts_idx = self.mri_transform(1- registration_artifacts_idx[np.newaxis, ...]*1.)
            ct_img[ct_img==0] = np.min(ct_img)
        
        # mri_img[mri_img==0] = np.min(mri_img)
        mri = self.mri_transform(mri_img[np.newaxis, ...])
        ct = self.ct_transform(ct_img[np.newaxis, ...])
        if self.surpress_registration_artifacts:
            return {'A': mri, 'B': ct, 'A_paths': mri_path, 'B_paths': ct_path, 'registration_artifacts_idx': registration_artifacts_idx}
        else:
            return {'A': mri, 'B': ct, 'A_paths': mri_path, 'B_paths': ct_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.mri_size, self.ct_size)
        
        
