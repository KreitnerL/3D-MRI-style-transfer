from data.image_folder import get_custom_file_paths, natural_sort
from data.base_dataset import BaseDataset
from data.data_augmentation_3D import PadIfNecessary, SpatialFlip, SpatialRotation, ColorJitterSphere3D, getBetterOrientation
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

        if opt.direction == 'AtoB':
            opt.AtoB = True
            opt.BtoA = False
        else:
            opt.AtoB = False
            opt.BtoA = True

        transformations = [
            transforms.Lambda(lambda x: self.toGrayScale(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float16 if opt.amp else torch.float32)),
            PadIfNecessary(3),
        ]
        self.updateTransformations = []

        if(opt.phase == 'train'):
            self.updateTransformations += [
                SpatialRotation([(1,2), (1,3), (2,3)], [*[0]*12,1,2,3], auto_update=False), # With a probability of approx. 51% no rotation is performed
                SpatialFlip(dims=(1,2,3), auto_update=False)
            ]
        transformations += self.updateTransformations
        self.mri_transform = transforms.Compose(transformations)
        self.ct_transform = transforms.Compose(transformations[1:])
        self.colorJitter = ColorJitterSphere3D((0.3, 1.5), (0.3,1.5), sigma=0.5) # ColorJitter3D(brightness_min_max=(0.3, 1.5), contrast_min_max=(0.3, 1.5))

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
        nifti: nib.Nifti1Image = nib.load(mri_path)
        if self.opt.AtoB:
            affine = nifti.affine
        nifti = getBetterOrientation(nifti, "IPL")
        mri_img = np.array(nifti.get_fdata())

        if self.opt.paired:   # make sure index is within then range
            index_ct = index % self.ct_size
        else:
            index_ct = random.randint(0, self.ct_size - 1)
            # index_B = 3
        ct_path = self.ct_paths[index_ct]
        nifti: nib.Nifti1Image = nib.load(ct_path)
        if self.opt.BtoA:
            affine = nifti.affine
        nifti = getBetterOrientation(nifti, "IPL")
        ct_img = np.array(nifti.get_fdata())
        if self.surpress_registration_artifacts:
            if self.opt.paired or self.opt.BtoA:
                registration_artifacts_idx = ct_img==0
            else:
                registration_artifacts_idx = np.array(getBetterOrientation(nib.load(self.ct_paths[index % self.ct_size]), "IPL").get_fdata()) == 0
            if self.opt.BtoA:
                # mri_img[np.array(getBetterOrientation(nib.load(self.ct_paths[index % self.ct_size]), "IPL").get_fdata()) == 0] = np.min(mri_img)
                registration_artifacts_idx = self.ct_transform(1- registration_artifacts_idx[np.newaxis, ...]*1.)
            else:
                registration_artifacts_idx = self.mri_transform(1- registration_artifacts_idx[np.newaxis, ...]*1.)
            ct_img[ct_img==0] = np.min(ct_img)
        mri = self.mri_transform(mri_img[np.newaxis, ...])
        if(self.opt.phase == 'train') and self.opt.AtoB:
            mri = self.colorJitter(mri)
        ct_img = (np.clip(ct_img, -1000., 1000.) + 1000.) / 2000.
        ct = self.ct_transform(ct_img[np.newaxis, ...])

        data = {'A': mri, 'B': ct, 'affine': affine, 'axis_code': "IPL", 'A_paths': mri_path, 'B_paths': ct_path}
        if self.surpress_registration_artifacts:
            data['registration_artifacts_idx'] = registration_artifacts_idx
        return data

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.mri_size, self.ct_size)
        
        
