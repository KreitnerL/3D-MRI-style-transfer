from data.brain_3D_dataset import brain3DDataset
from data.image_folder import get_custom_file_paths, natural_sort
import os

class brain3DSuperResDataset(brain3DDataset):
    def __init__(self, opt):
        A_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'T1_LR', opt.phase), '.nii.gz'))
        B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'T1_HR', opt.phase), '.nii.gz'))
        super().__init__(opt, [A_paths], B_paths)
