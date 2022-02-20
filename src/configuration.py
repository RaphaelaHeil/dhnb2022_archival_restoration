"""
Contains all code related to the configuration of experiments.
"""
from __future__ import annotations  # needed for class method return-typing

import argparse
import configparser
import random
import time
from configparser import SectionProxy
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple

import torch


class ModelName(Enum):
    """
    Encodes the names of supported base models.
    """
    DENSE = auto()
    TIRAMISU = auto()

    @staticmethod
    def getByName(name: str) -> ModelName:
        """
        Returns the ModelName corresponding to the given string. Returns ModelName.DENSE in case an unknown name is
        provided.

        Parameters
        ----------
        name : str
            string representation that should be converted to a ModelName

        Returns
        -------
            ModelName representation of the provided string, default: ModelName.DENSE
        """
        if name.upper() in [model.name for model in ModelName]:
            return ModelName[name.upper()]
        else:
            return ModelName.DENSE


class AugmentationType(Enum):
    """
    Encodes the available augmentation types.
    """
    JPEG = auto()
    GREYSCALE = auto()
    MICROFORM = auto()
    WARPED = auto()
    ALL = auto()

    @staticmethod
    def getByName(name: str) -> AugmentationType | None:
        """
        Returns the AugmentationType corresponding to the given string. Returns None if no matching type is available.

        Parameters
        ----------
        name : str
            string representation that should be converted to am AugmentationType

        Returns
        -------
            AugmentationType representation of the provided string, or None
        """
        if name.upper() in [aug.name for aug in AugmentationType]:
            return AugmentationType[name.upper()]
        else:
            return None


class Configuration:
    """
    Holds the configuration for the current experiment.
    """

    def __init__(self, parsedConfig: SectionProxy, test: bool = False, fileSection: str = "DEFAULT"):
        self.fileSection = fileSection
        self.parsedConfig = parsedConfig
        if not test:
            self.outDir = Path(parsedConfig.get('outdir')) / '{}_{}_{}'.format(fileSection, str(int(time.time())),
                                                                               random.randint(0, 100000))
            parsedConfig['outdir'] = str(self.outDir)

        if not test and not self.outDir.exists():
            self.outDir.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.epochs = self.getSetInt('epochs', 100)
        self.generatorLearningRate = self.getSetFloat('generator_learning_rate', 0.001)
        self.generatorBetas = self.parseBetas(self.getSetStr("generator_betas", "0.9,0.999"))
        self.discriminatorLearningRate = self.getSetFloat('discriminator_learning_rate', 0.001)
        self.discriminatorBetas = self.parseBetas(self.getSetStr("discriminator_betas", "0.9,0.999"))

        self.batchSize = self.getSetInt('batchsize', 4)
        self.imageHeight = self.getSetInt('imageheight', 256)
        self.imageWidth = self.getSetInt('imagewidth', 256)
        self.modelSaveEpoch = self.getSetInt('modelsaveepoch', 10)
        self.imageSaveInterval = self.getSetInt('imagesaveinterval', 10)
        self.imageSaveCount = self.getSetInt('imagesavecount', 4)
        self.validationEpoch = self.getSetInt('validation_epoch', 10)
        self.trainImageDir = Path(self.getSetStr('trainimgagebasedir'))
        self.testImageDir = Path(self.getSetStr('testimagedir'))

        self.down = self.parseTiramisuConfig(self.getSetStr("down", "4"))
        self.bottleneck = self.getSetInt("bottleneck", 4)
        self.up = self.parseTiramisuConfig(self.getSetStr("up", "4"))

        self.poolSize = self.getSetInt('poolsize', 20)

        self.trainCount = self.getSetInt('train_count', 1000)

        self.cnnLambda = self.getSetFloat('cnn_lambda', 0.5)
        self.identityLambda = self.getSetFloat('identity_lambda', 0.5)
        self.cleanLambda = self.getSetFloat('clean_lambda', 10.0)
        self.distortedLambda = self.getSetFloat('struck_lambda', 10.0)

        self.modelName = ModelName.getByName(self.getSetStr("model", "DENSE"))
        self.augmentationTypes = self.parseAugmentationTypes(self.getSetStr("augmentation_types", "GREYSCALE,JPEG"))

        if not test:
            configOut = self.outDir / 'config.cfg'
            with configOut.open('w+') as cfile:
                parsedConfig.parser.write(cfile)

    def getSetInt(self, key: str, default: int | None = None) -> int:
        """
        Gets an int with the given key from the parsedConfig. Config and return value are set to default if not value
        is given in parsedConfig.

        Parameters
        ----------
        key : str
            String to be parsed.
        default : int, optional
            default value to be used if no value is given in the configuration

        Returns
        -------
        int
            config value for given key
        """
        value = self.parsedConfig.getint(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetFloat(self, key: str, default: float | None = None) -> float:
        """
        Gets a float with the given key from the parsedConfig. Config and return value are set to default if not value
        is given in parsedConfig.

        Parameters
        ----------
        key : str
            String to be parsed.
        default : float, optional
            default value to be used if no value is given in the configuration

        Returns
        -------
        float
            config value for given key
        """
        value = self.parsedConfig.getfloat(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetBoolean(self, key: str, default: bool | None = None) -> bool:
        """
        Gets a boolean with the given key from the parsedConfig. Config and return value are set to default if not value
        is given in parsedConfig.

        Parameters
        ----------
        key : str
            String to be parsed.
        default : bool, optional
            default value to be used if no value is given in the configuration

        Returns
        -------
        bool
            config value for given key
        """
        value = self.parsedConfig.getboolean(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetStr(self, key: str, default: str | None = None) -> str:
        """
        Gets a str with the given key from the parsedConfig. Config and return value are set to default if not value
        is given in parsedConfig.

        Parameters
        ----------
        key : str
            String to be parsed.
        default : str, optional
            default value to be used if no value is given in the configuration

        Returns
        -------
        str
            config value for given key
        """
        value = self.parsedConfig.get(key, default)
        self.parsedConfig[key] = str(value)
        return value

    @staticmethod
    def parseBetas(betaString: str) -> Tuple[float, float]:
        """
        Parses a comma-separated string to a list of floats.

        Parameters
        ----------
        betaString : str
            String to be parsed.

        Returns
        -------
        Tuple[float, float]
            parsed beta values

        Raises
        ------
        ValueError
            if fewer or more than two values are specified
        """
        betas = betaString.split(',')
        if len(betas) < 2:
            raise ValueError("found fewer than two values for betas")
        if len(betas) > 2:
            raise ValueError("found more than two values for betas")
        return float(betas[0]), float(betas[1])

    @staticmethod
    def parseAugmentationTypes(strokeString: str) -> List[AugmentationType]:
        """
        Parses a comma-separated string to a list of augmentation types.

        Parameters
        ----------
        strokeString : str
            string to be parsed

        Returns
        -------
        List[StrikeThroughType]
            list of stroke type strings
        """
        allTypes = [stroke for stroke in AugmentationType]

        if '|' in strokeString:
            splitTypes = strokeString.split('|')  # for backward compatibility
        else:
            splitTypes = strokeString.split(',')
        augmentationTypes = []
        if "all" in splitTypes:
            augmentationTypes = allTypes
        else:
            for item in splitTypes:
                aug = AugmentationType.getByName(item.strip().upper())
                if aug:
                    augmentationTypes.append(aug)
        if len(augmentationTypes) < 1:
            augmentationTypes = allTypes
        return augmentationTypes

    @staticmethod
    def parseTiramisuConfig(configString: str) -> List[int]:
        """
        Parses a comma-separated string of numbers to a list of ints.

        Parameters
        ----------
        configString : str
            string to be parsed

        Returns
        -------
        List[int]
            list of parsed integer values
        """
        split = configString.split(",")
        result = [int(s.strip()) for s in split]
        return result


def getConfiguration() -> Configuration:
    """
    Reads the required arguments from command line and parses the respective configuration file/section. The parsed
    configuration will be copied to the specified output directory.

    Returns
    -------
    Configuration
        parsed configuration
    """
    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-section", required=False, help="section of config-file to use")
    cmdParser.add_argument("-file", required=False, help="path to config-file")
    args = vars(cmdParser.parse_args())
    fileSection = 'DEFAULT'
    fileName = 'config.cfg'
    if args["section"]:
        fileSection = args["section"]

    if args['file']:
        fileName = args['file']
    configParser = configparser.ConfigParser()
    configParser.read(fileName)
    parsedConfig = configParser[fileSection]
    sections = configParser.sections()
    for s in sections:
        if s != fileSection:
            configParser.remove_section(s)
    return Configuration(parsedConfig, fileSection=fileSection)
