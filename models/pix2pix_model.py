import torch
from models.bayesian import kl_divergence_from_nn
from .base_model import BaseModel
from . import networks
from torchvision.transforms import Compose
from data.data_augmentation_3D import ColorJitter3D, RandomBiasField, RandomBlur, RandomNoise


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan', paired=True)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--perceptual', type=str, default=None, choices=['random', 'D', 'D_aug'], help="Use perceptual loss")
            parser.add_argument('--multitask',  action='store_true', default=False, help="If set learn the weights of the multi-task loss automatically")

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            if opt.perceptual is not None:
                self.loss_names.append('G_perceptual')
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.networks = [self.netG]
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.networks.extend([self.netD])

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, dtype=torch.float16 if opt.amp else torch.float32).to(self.device)
            self.l1 = torch.nn.L1Loss()
            self.layers = []
            self.λ_G = []
    
            if opt.perceptual is not None:
                self.layers = [0, 3, 6, 9]
                self.perceptual_loss = torch.nn.L1Loss()
                λ_G = [opt.lambda_L1, 1.0, 0.3, 0.3, 0.2]
                if opt.multitask:
                    self.to_sigma = lambda x: (1/(2*torch.exp(x)))
                    self.λ_G = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1./(2*λ_i)).log()) for λ_i in λ_G]).to(device=opt.gpu_ids[0])
                    self.stats_names.append('λ_G')
                else:
                    self.to_sigma = lambda x: x
                    self.λ_G = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(λ_i), requires_grad=False) for λ_i in λ_G]).to(device=opt.gpu_ids[0])

                if opt.perceptual == 'random':
                    self.random_D = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
                    self.set_requires_grad(self.random_D, False)

            if opt.perceptual == 'd_aug':
                self.d_aug = Compose([
                    RandomNoise(std=(0.,0.02)),
                    RandomBiasField([0,0.5]),
                    # RandomBlur([0,2]),
                    ColorJitter3D(brightness_min_max=(0.7,1.3), contrast_min_max=(0.7,1.3)),
                ])
            else:
                self.d_aug = lambda x: x

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam([*self.netG.parameters(), *self.λ_G], lr=opt.glr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.dlr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        if self.registration_artifacts_idx is not None:
            self.fake_B = self.fake_B * self.registration_artifacts_idx.to(self.fake_B.device)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            real_A_augmented = self.d_aug(self.real_A.flatten(0,1)).view(self.real_A.shape)
            fake_AB = torch.cat((real_A_augmented, self.d_aug(self.fake_B.flatten(0,1).detach()).view(self.fake_B.shape)), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.netD(fake_AB)
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((real_A_augmented, self.d_aug(self.real_B.flatten(0,1)).view(self.real_B.shape)), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.scaler.scale(self.loss_D).backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            if self.opt.perceptual == 'D' or  self.opt.perceptual == 'D_aug':
                pred_fake, feats_fake = self.netD(fake_AB, layers=self.layers)
            elif self.opt.perceptual == 'random':
                pred_fake = self.netD(fake_AB)
                feats_fake = self.random_D(fake_AB, layers=self.layers, encode_only=True)
            else:
                pred_fake = self.netD(fake_AB, layers=self.layers)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.l1(self.fake_B, self.real_B)
            if self.opt.perceptual is not None:
                self.loss_G_L1 *= self.to_sigma(self.λ_G[0])
                self.loss_G_perceptual = torch.tensor(0, device=self.opt.gpu_ids[0], dtype=torch.float16)
                feats_real = self.netD(torch.cat((self.real_A, self.real_B), 1).detach(), layers=self.layers, encode_only=True)
                for i, λ_i in enumerate(self.λ_G[1:]):
                    self.loss_G_perceptual += self.to_sigma(λ_i) * self.perceptual_loss(feats_fake[i], feats_real[i])
                
                self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceptual
                # Self-learning multitask loss
                if self.opt.multitask:
                    for λ_i in self.λ_G:
                        self.loss_G += λ_i
            else:
                self.loss_G_L1 *= self.opt.lambda_L1
                self.loss_G = self.loss_G_GAN + self.loss_G_L1
            if self.opt.confidence == 'bayesian':
                self.kl_divergence = kl_divergence_from_nn(self.netG)
                self.scaler.scale(self.loss_G + self.kl_divergence).backward()
            else:
                self.scaler.scale(self.loss_G).backward()

    def optimize_parameters(self):
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.scaler.step(self.optimizer_D)  # udpate D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.scaler.step(self.optimizer_G)  # udpate G's weights
        self.scaler.update()                # Updates the scale for next iteration 
