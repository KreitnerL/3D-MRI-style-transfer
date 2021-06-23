from data.image_folder import get_custom_file_paths
from data.base_dataset import BaseDataset
import nibabel as nib
import random
from torchvision import transforms

class MRIDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.A_paths = sorted(get_custom_file_paths(opt.dataroot, 't1.nii.gz'))
        self.B_paths = sorted(get_custom_file_paths(opt.dataroot, 't2.nii.gz'))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self.normalize(x))
        ])

    def normalize(self, x):
        x_min = x.amin()
        x_max = x.amax()
        x = (x - x_min) / x_max * 2 -1
        return x

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = nib.load(A_path).get_fdata()
        B_img = nib.load(B_path).get_fdata()
        A = self.transform(A_img)
        B = self.transform(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
        
        
