import random
from typing import List, Tuple, Union
import torch.nn.functional as F
from collections.abc import Sequence
import torch
import nibabel as nib
import numpy as np
import torch
import numbers

class SpatialRotation():
    def __init__(self, dimensions: Sequence, k: Sequence = [3], auto_update=True):
        self.dimensions = dimensions
        self.k = k
        self.args = None
        self.auto_update = auto_update
        self.update()

    def update(self):
        self.args = [random.choice(self.k) for dim in self.dimensions]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_update:
            self.update()
        for k, dim in zip(self.args, self.dimensions):
            x = torch.rot90(x, k, dim)
        return x

class SpatialFlip():
    def __init__(self, dims: Sequence, auto_update=True) -> None:
        self.dims = dims
        self.args = None
        self.auto_update = auto_update
        self.update()

    def update(self):
        self.args = tuple(random.sample(self.dims, random.choice(range(len(self.dims)))))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_update:
            self.update()
        x = torch.flip(x, self.args)
        return x

class PadIfNecessary():
    def __init__(self, n_downsampling: int):
        self.mod = 2**n_downsampling

    def __call__(self, x: torch.Tensor):
        padding = []
        for dim in reversed(x.shape[1:]):
            padding.extend([0, (self.mod - dim%self.mod)%self.mod])
        x = F.pad(x, padding)
        return x

    def pad(x, n_downsampling: int = 1):
        mod = 2**n_downsampling
        padding = []
        for dim in reversed(x.shape[1:]):
            padding.extend([0, (mod - dim%mod)%mod])
        x = F.pad(x, padding)
        return x

class ColorJitter3D():
    """
    Randomly change the brightness and contrast an image.
    A grayscale tensor with values between 0-1 and shape BxCxHxWxD is expected.
    Args:
        brightness (float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
    """
    def __init__(self, brightness_min_max: tuple=None, contrast_min_max: tuple=None) -> None:
        self.brightness_min_max = brightness_min_max
        self.contrast_min_max = contrast_min_max
        self.update()

    def update(self):
        if self.brightness_min_max:
            self.brightness = float(torch.empty(1).uniform_(self.brightness_min_max[0], self.brightness_min_max[1]))
        if self.contrast_min_max:
            self.contrast = float(torch.empty(1).uniform_(self.contrast_min_max[0], self.contrast_min_max[1]))

    def __call__(self, x: torch.Tensor, no_update=False) -> torch.Tensor:
        if not no_update:
            self.update()
        if self.brightness_min_max:
            x = (self.brightness * x).float().clamp(0, 1.).to(x.dtype)
        if self.contrast_min_max:
            mean = torch.mean(x.float(), dim=(-4, -3, -2, -1), keepdim=True)
            x = (self.contrast * x + (1.0 - self.contrast) * mean).float().clamp(0, 1.).to(x.dtype)
        return x

class ColorJitterSphere3D():
    """
    Randomly change the brightness and contrast an image.
    A grayscale tensor with values between 0-1 and shape BxCxHxWxD is expected.
    Args:
        brightness (float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
    """
    def __init__(self, brightness_min_max: tuple=None, contrast_min_max: tuple=None, sigma: float=1., dims: int=3) -> None:
        self.brightness_min_max = brightness_min_max
        self.contrast_min_max = contrast_min_max
        self.sigma = sigma
        self.dims = dims
        self.update()

    def update(self):
        if self.brightness_min_max:
            self.brightness = float(torch.empty(1).uniform_(self.brightness_min_max[0], self.brightness_min_max[1]))
        if self.contrast_min_max:
            self.contrast = float(torch.empty(1).uniform_(self.contrast_min_max[0], self.contrast_min_max[1]))
        self.ranges = []
        for _ in range(self.dims):
            r = torch.rand(2) * 10 - 5
            self.ranges.append((r.min().item(), r.max().item()))

    def __call__(self, x: torch.Tensor, no_update=False) -> torch.Tensor:
        if not no_update:
            self.update()

        jitterSphere = torch.zeros(1)
        for i,r in enumerate(self.ranges):
            jitterSphere_i = torch.linspace(*r, steps=x.shape[i + 1])
            jitterSphere_i = (1 / (self.sigma * 2.51)) * 2.71**(-0.5 * (jitterSphere_i/self.sigma) ** 2) # Random section of a normal distribution between (-5,5)
            jitterSphere = jitterSphere.unsqueeze(-1) + jitterSphere_i.view(1, *[1]*i, -1)
        jitterSphere /= torch.max(jitterSphere) # Random 3D section of a normal distribution sphere
        
        if self.brightness_min_max:
            brightness = (self.brightness - 1) * jitterSphere + 1
            x = (brightness * x).float().clamp(0, 1.).to(x.dtype)
        if self.contrast_min_max:
            contrast = (self.contrast - 1) * jitterSphere + 1
            mean =x.float().mean()
            x = (contrast * x + (1.0 - self.contrast) * mean).float().clamp(0, 1.).to(x.dtype)
        return x

class RandomBlur():
    r"""Blur an image using a random-sized Gaussian filter.

    Args:
        std: Tuple :math:`(a_1, b_1, a_2, b_2, a_3, b_3)` representing the
            ranges (in mm) of the standard deviations
            :math:`(\sigma_1, \sigma_2, \sigma_3)` of the Gaussian kernels used
            to blur the image along each axis, where
            :math:`\sigma_i \sim \mathcal{U}(a_i, b_i)`.
            If two values :math:`(a, b)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(a, b)`.
            If only one value :math:`x` is provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x)`.
            If three values :math:`(x_1, x_2, x_3)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x_i)`.
    """
    def __init__(self, std: Union[float, Tuple[float, float]] = (0, 2)):
        self.std_range = std

    def createKernel(self, channels: int, sigma: float, kernel_size: 3, dim=3):
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1. / (std * torch.tensor(2 * torch.pi).sqrt()) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        return kernel

    def __call__(self, x: torch.Tensor):
        std = torch.FloatTensor(1).uniform_(*self.std_range).item()
        dtype = x.dtype
        if x.device.type=='cpu':
            x = x.float()
        kernel = self.createKernel(x.shape[0], sigma=std, kernel_size=3, dim=3).to(device=x.device, dtype=x.dtype)
        x = F.conv3d(x.unsqueeze(0), weight=kernel, groups=x.shape[0]).squeeze(0)
        x = F.pad(x, [1,1]*3, mode='reflect')
        return x.type(dtype)

class RandomNoise():
    r"""Add Gaussian noise with random parameters.

    Add noise sampled from a normal distribution with random parameters.

    Args:
        mean: Mean :math:`\mu` of the Gaussian distribution
            from which the noise is sampled.
            If two values :math:`(a, b)` are provided,
            then :math:`\mu \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`\mu \sim \mathcal{U}(-d, d)`.
        std: Standard deviation :math:`\sigma` of the Gaussian distribution
            from which the noise is sampled.
            If two values :math:`(a, b)` are provided,
            then :math:`\sigma \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`\sigma \sim \mathcal{U}(0, d)`.
    """
    def __init__(
            self,
            mean: float = 0,
            std: Tuple[float, float] = (0, 0.25),
            ):
        self.mean = mean
        self.std_range = std

    def __call__(self, x: torch.Tensor):
        mean = self.mean
        std = torch.FloatTensor(1).uniform_(*self.std_range).item()
        noise = torch.randn(*x.shape, device=x.device) * std + mean
        x = (x + noise).clip(0,1)
        return x

class RandomBiasField():
    r"""Add random MRI bias field artifact.

    MRI magnetic field inhomogeneity creates intensity
    variations of very low frequency across the whole image.

    The bias field is modeled as a linear combination of
    polynomial basis functions, as in K. Van Leemput et al., 1999,
    *Automated model-based tissue classification of MR images of the brain*.

    It was implemented in NiftyNet by Carole Sudre and used in
    `Sudre et al., 2017, Longitudinal segmentation of age-related
    white matter hyperintensities
    <https://www.sciencedirect.com/science/article/pii/S1361841517300257?via%3Dihub>`_.

    Args:
        coefficients: Maximum magnitude :math:`n` of polynomial coefficients.
            If a tuple :math:`(a, b)` is specified, then
            :math:`n \sim \mathcal{U}(a, b)`.
        order: Order of the basis polynomial functions.
    """
    def __init__(self, coefficients: Tuple[float, float] = [0.5,0.5], order: int = 3) -> None:
        self.coefficients = coefficients
        self.order = order
    
    def get_params(
            self,
            order: int,
            coefficients_range: Tuple[float, float],
            ) -> List[float]:
        # Sampling of the appropriate number of coefficients for the creation
        # of the bias field map
        random_coefficients = []
        for x_order in range(0, order + 1):
            for y_order in range(0, order + 1 - x_order):
                for _ in range(0, order + 1 - (x_order + y_order)):
                    number = torch.FloatTensor(1).uniform_(*coefficients_range)
                    random_coefficients.append(number.item())
        return random_coefficients

    @staticmethod
    def generate_bias_field(
            data: torch.Tensor,
            order: int,
            coefficients: List[float],
            ) -> np.ndarray:
        # Create the bias field map using a linear combination of polynomial
        # functions and the coefficients previously sampled
        shape = torch.tensor(data.shape[1:])  # first axis is channels
        half_shape = shape / 2

        ranges = [torch.arange(-n, n, device=data.device) + 0.5 for n in half_shape]

        bias_field = torch.zeros(data.shape[1:], device=data.device)
        meshes = list(torch.meshgrid(*ranges))

        for i in range(len(meshes)):
            mesh_max = meshes[i].max()
            if mesh_max > 0:
                meshes[i] = meshes[i] / mesh_max
        x_mesh, y_mesh, z_mesh = meshes

        i = 0
        for x_order in range(order + 1):
            for y_order in range(order + 1 - x_order):
                for z_order in range(order + 1 - (x_order + y_order)):
                    coefficient = coefficients[i]
                    new_map = (
                        coefficient
                        * x_mesh ** x_order
                        * y_mesh ** y_order
                        * z_mesh ** z_order
                    )
                    bias_field += new_map
                    i += 1
        bias_field = 1./torch.exp(bias_field)
        return bias_field

    def __call__(self, x: torch.Tensor):
        dtype=x.dtype
        coefficients = self.get_params(self.order, self.coefficients)
        bias_field = self.generate_bias_field(x, self.order, coefficients)
        x = x * bias_field
        return x.clip(0,1).type(dtype)

def getBetterOrientation(nifti: nib.Nifti1Image, axisCode="IPL"):
    orig_ornt = nib.io_orientation(nifti.affine)
    targ_ornt = nib.orientations.axcodes2ornt(axisCode)
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    nifti = nifti.as_reoriented(transform)
    return nifti

def toGrayScale(x):
    x_min = np.amin(x)
    x_max = np.amax(x) - x_min
    x = (x - x_min) / x_max
    return x

def center(x, mean, std):
    return (x - mean) / std
