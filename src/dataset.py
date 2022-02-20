from pathlib import Path
from typing import Tuple, Iterable, Dict, List, Any

import numpy as np
import torch.cuda
import torchvision.transforms.functional as f
from PIL import Image
from imgaug.augmenters import Augmenter
from numpy.random import default_rng
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomCrop

from src.utils import AugmentationType


class ArtificialNoisePatchDataset(Dataset):

    def __init__(self, rootDir: Path, transforms: Compose = None, alteringAugmentation: Augmenter = None,
                 cropSize: Tuple[int, int] = (256, 256), patchCount: int = 32):
        self.patchCount = patchCount
        self.pages = list(rootDir.glob("*.jpg"))
        if len(self.pages) < 2:
            raise ValueError("directory has to contain at least two images")
        self.transforms = transforms
        self.alteringAugmentation = alteringAugmentation
        self.cropSize = cropSize
        self.referenceCropper = RandomCrop(size=self.cropSize)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self) -> int:
        return self.patchCount

    def __getitem__(self, item: int) -> Dict[str, Any]:
        selection = default_rng().choice(self.pages, 2, replace=False)
        toTensor = ToTensor()

        referencePage = np.asarray(Image.open(selection[0]).convert('RGB'))
        distortedPage = np.asarray(Image.open(selection[1]).convert('RGB'))

        original = distortedPage.copy()

        if self.alteringAugmentation is not None:
            distortedPage = self.alteringAugmentation(image=distortedPage)

        referencePage = toTensor(referencePage)
        distortedPage = toTensor(distortedPage)
        original = toTensor(original)

        i, j, h, w = RandomCrop.get_params(distortedPage, output_size=self.cropSize)
        distortedPatch = f.crop(distortedPage, i, j, h, w)
        original = f.crop(original, i, j, h, w)

        referencePatch = self.referenceCropper(referencePage)

        return {"clean": referencePatch, "distorted": distortedPatch, "original": original}


class ValidationDataset(Dataset):
    AUG_ABBREVIATIONS = {AugmentationType.JPEG: "jpeg", AugmentationType.GREYSCALE: "grey",
                         AugmentationType.MICROFORM: "micro", AugmentationType.WARPED: "warp"}

    def __init__(self, basePath: Path, augmentationTypes: List[AugmentationType] = None):
        originals = basePath / "original"
        augmented = basePath / "augmented"
        self.data = [(patch, originals / "original_{}".format(patch.name)) for patch in augmented.glob("*.png")]
        if augmentationTypes:
            if AugmentationType.ALL not in augmentationTypes:
                self.data = []
                for aug in augmentationTypes:
                    self.data.extend([(patch, originals / "original_{}".format(patch.name)) for patch in
                                      augmented.glob("*{}*.png".format(self.AUG_ABBREVIATIONS[aug]))])
        self.transform = ToTensor()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        augmentedPath, originalPath = self.data[index]

        originalPatch = self.transform(Image.open(originalPath).convert('RGB'))
        augmentedPatch = self.transform(Image.open(augmentedPath).convert('RGB'))

        return {"distorted": augmentedPatch, "original": originalPatch, "name": augmentedPath.name}


class TestDataset(Dataset):
    AUG_ABBREVIATIONS = {AugmentationType.JPEG: "jpeg", AugmentationType.GREYSCALE: "grey",
                         AugmentationType.MICROFORM: "micro", AugmentationType.WARPED: "warp"}

    def __init__(self, basePath: Path, augmentationTypes: Iterable[AugmentationType] = None):
        originals = basePath / "original"
        augmented = basePath / "augmented"
        self.data = [(patch, originals / patch.name) for patch in augmented.glob("*.jpg")]
        if augmentationTypes:
            if AugmentationType.ALL not in augmentationTypes:
                self.data = []
                for aug in augmentationTypes:
                    self.data.extend([(patch, originals / patch.name) for patch in
                                      augmented.glob("*{}*.jpg".format(self.AUG_ABBREVIATIONS[aug]))])
        self.transform = ToTensor()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        augmentedPath, originalPath = self.data[index]

        originalPatch = self.transform(Image.open(originalPath).convert('RGB'))
        augmentedPatch = self.transform(Image.open(augmentedPath).convert('RGB'))

        return {"distorted": augmentedPatch, "original": originalPatch, "name": augmentedPath.name}
