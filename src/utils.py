"""
Utility module.
"""
from typing import Tuple, Any, List, Union

import imgaug
import numpy as np
import torch
from imgaug.augmenters import ElasticTransformation, JpegCompression, Grayscale, OneOf, Augmenter, meta
from models import Tiramisu
from models.modulebank import DEFAULT_MODULE_BANK, ModuleType
from scipy.ndimage import distance_transform_edt
from skimage.color import rgb2gray
from skimage.exposure import adjust_sigmoid
from skimage.morphology import rectangle
from skimage.transform import resize
from torch import nn

from .configuration import ModelName, Configuration, AugmentationType
from .network import discriminator, generators


class ToMicroform(meta.Augmenter):

    def __init__(self, masked: bool = True, height: int = 4600, width: int = 3500):
        super().__init__()
        self.masked = masked
        if self.masked:
            self.MASK = self.precalcMask(height, width)
        self.MASK = np.ones((height, width))

    def precalcMask(self, height: int = 4600, width: int = 3500) -> np.ndarray:
        centerX = width // 2
        centerY = height // 2

        maskWidth = centerX
        maskHeight = centerY

        shapeMask = rectangle(maskHeight, maskWidth)
        mask = np.zeros((height, width))
        x = centerX // 2
        y = centerY // 2
        mask[y:y + shapeMask.shape[0], x:x + shapeMask.shape[1]] = shapeMask

        maskDT = distance_transform_edt(1 - mask)
        maskDT /= maskDT.max()
        result = (1.0 - maskDT.astype(np.float32))
        return result

    def augment_image(self, image: np.ndarray, hooks: Union[None, imgaug.HooksImages] = None) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise ValueError("Expected input to be of type numpy.ndarray but received " + str(type(image)))

        greyscale = rgb2gray(image).astype(np.float32)
        image = adjust_sigmoid(greyscale)

        if self.masked:
            resizedMask = resize(self.MASK, output_shape=image.shape)
            image = resizedMask * image
        image = np.dstack([image, image, image])

        return image

    def get_parameters(self) -> List[Any]:
        return []


def composeAugmentations(config: Configuration) -> Augmenter:
    augmentations = []
    if AugmentationType.MICROFORM in config.augmentationTypes or AugmentationType.ALL in config.augmentationTypes:
        augmentations.append(ToMicroform())
    elif AugmentationType.GREYSCALE in config.augmentationTypes or AugmentationType.ALL in config.augmentationTypes:
        augmentations.append(Grayscale(alpha=1.0))
    if AugmentationType.JPEG in config.augmentationTypes or AugmentationType.ALL in config.augmentationTypes:
        augmentations.append(JpegCompression(compression=(65, 85)))
    if AugmentationType.WARPED in config.augmentationTypes or AugmentationType.ALL in config.augmentationTypes:
        augmentations.append(ElasticTransformation(alpha=(15, 25), sigma=(4, 6)))
    if AugmentationType.ALL in config.augmentationTypes:
        return OneOf(children=augmentations)
    return OneOf(augmentations)


def getDiscriminatorModels() -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Prepares the CycleGAN discriminator models based on the given configuration.

    Returns
    -------
    Tuple[torch.nn.Module, torch.nn.Module]
        clean discriminator, distorted discriminator
    """

    cleanDiscriminator = discriminator.NLayerDiscriminator(input_nc=3)
    distortedDiscriminator = discriminator.NLayerDiscriminator(input_nc=3)
    return cleanDiscriminator, distortedDiscriminator


def getGeneratorModels(config: Configuration) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Prepares the CycleGAN generator models based on the given configuration.

    Parameters
    ----------
    config : Configuration
        experiment configuration

    Returns
    -------
    Tuple[torch.nn.Module, torch.nn.Module]
        generator 'distort', generator 'restore'
    """
    if config.modelName == ModelName.TIRAMISU:
        structure = (config.down, config.bottleneck, config.up)

        module_bank = DEFAULT_MODULE_BANK.copy()
        module_bank[ModuleType.ACTIVATION_FINAL] = nn.Sigmoid

        distort = Tiramisu(3, 3, structure=structure, module_bank=module_bank, checkpoint=True)

        restore = Tiramisu(3, 3, structure=structure, module_bank=module_bank, checkpoint=True)
        return distort, restore
    else:
        distort = generators.DenseGenerator(3, 3, norm_layer=torch.nn.BatchNorm2d, n_blocks=config.bottleneck)

        restore = generators.DenseGenerator(3, 3, norm_layer=torch.nn.BatchNorm2d, n_blocks=config.bottleneck)
        return distort, restore
