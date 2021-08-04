import os
import math
from shutil import copyfile
from data.image_folder import get_custom_file_paths
from tqdm import tqdm


t1_set = sorted(get_custom_file_paths('/media/data_4T/linus/', 't1.nii.gz'))
t2_set = sorted(get_custom_file_paths('/media/data_4T/linus/', 't2.nii.gz'))
t3_set = sorted(get_custom_file_paths('/media/data_4T/linus/', 't3.nii.gz'))

assert len(t1_set) == len(t2_set)

test_split = 0.1
train_set_size = math.ceil(len(t1_set) * (1-test_split))
test_set_size = len(t1_set) - train_set_size
print('Train set size: %d, Test set size: %d' % (train_set_size, test_set_size))

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

dataroot = '/media/data_4T/linus/3D_brain_mri/'
mkdir(dataroot)
mkdir(os.path.join(dataroot, 'trainT1/'))
mkdir(os.path.join(dataroot, 'trainT2/'))
mkdir(os.path.join(dataroot, 'trainT3/'))
mkdir(os.path.join(dataroot, 'testT1/'))
mkdir(os.path.join(dataroot, 'testT2/'))
mkdir(os.path.join(dataroot, 'testT3/'))

print('Creating train set...')
for i in tqdm(range(train_set_size)):
    copyfile(t1_set[i], os.path.join(dataroot, 'trainT1/%d_t1.nii.gz'%i))
    copyfile(t1_set[i], os.path.join(dataroot, 'trainT2/%d_t2.nii.gz'%i))
    copyfile(t1_set[i], os.path.join(dataroot, 'trainT3/%d_t3.nii.gz'%i))

print('Creating test set...')
for i in tqdm(range(train_set_size, len(t1_set))):
    copyfile(t1_set[i], os.path.join(dataroot, 'testT1/%d_t1.nii.gz'%i))
    copyfile(t1_set[i], os.path.join(dataroot, 'testT2/%d_t2.nii.gz'%i))
    copyfile(t1_set[i], os.path.join(dataroot, 'testT3/%d_t3.nii.gz'%i))