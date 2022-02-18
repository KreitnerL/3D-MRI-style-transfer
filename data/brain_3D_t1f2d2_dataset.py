from data.brain_3D_dataset import brain3DDataset
from data.image_folder import get_custom_file_paths, natural_sort
import os

class brain3DT1F2D2Dataset(brain3DDataset):
    def __init__(self, opt):
        A1_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 't1', opt.phase), 't1.nii.gz'))
        A2_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'flair', opt.phase), 'flair.nii.gz'))
        B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'dir', opt.phase), 'dir.nii.gz'))
        super().__init__(opt, [A1_paths, A2_paths], B_paths)
