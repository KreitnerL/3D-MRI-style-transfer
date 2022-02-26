from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm
from util.ssim import SSIM
from util.util import PSNR
from torch.nn import L1Loss
from numpy import mean, ceil



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.paired = True
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results

    validation_functions = {
        'L1': L1Loss(),
        'SSIM': SSIM(),
        'PSNR/100': PSNR(),
        # 'NCC': NCC()
    }
    val_loss = {}
    dataset_len = len(dataset)
    for i, data in enumerate(tqdm(dataset, total=ceil(len(dataset)/opt.batch_size), desc='Validating')):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        for label, fun in validation_functions.items():
            if label in val_loss:
                val_loss[label].append(fun(model.fake_B, model.real_B).mean().item() * model.fake_B.shape[0])
            else:
                val_loss[label] = [fun(model.fake_B, model.real_B).mean().item() * model.fake_B.shape[0]]
    for label, vals in val_loss.items():
        val_loss[label] = sum(vals)/dataset_len
    print(val_loss)