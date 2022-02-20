"""
Code to test a previously trained neural network regarding its performance of removing distortions from a manuscript
image.
"""
import argparse
import configparser
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from skimage.metrics import normalized_root_mse, structural_similarity
from torch.utils.data import DataLoader

from .configuration import Configuration, ModelName
from .dataset import TestDataset
from .utils import getGeneratorModels

INFO_LOGGER_NAME = "removal"
RESULTS_LOGGER_NAME = "results"


def initLoggers(config: Configuration) -> None:
    """
    Utility function initialising a default info logger, as well as a results logger.

    Parameters
    ----------
    config : Configuration
        experiment configuration, to obtain the output location for file loggers

    Returns
    -------
        None
    """
    logger = logging.getLogger(INFO_LOGGER_NAME)
    logger.setLevel(logging.INFO)

    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    formatter = logging.Formatter('%(asctime)s - %(message)s', '%d-%b-%y %H:%M:%S')

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(config.outDir / "info.log", mode='w')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    resultsLogger = logging.getLogger(RESULTS_LOGGER_NAME)
    resultsLogger.setLevel(logging.INFO)

    while resultsLogger.hasHandlers():
        resultsLogger.removeHandler(resultsLogger.handlers[0])

    fileHandler = logging.FileHandler(config.outDir / "{}.log".format(RESULTS_LOGGER_NAME), mode='w')
    fileHandler.setLevel(logging.INFO)
    resultsLogger.addHandler(fileHandler)


class TestRunner:
    """
    Utility class that wraps the initialisation and testing of a neural network.
    """

    def __init__(self, configuration: Configuration, saveCleanedImages: bool = True,
                 model_name: str = "genRestore_best_rmse.pth", filterAugmentations: bool = False):
        self.logger = logging.getLogger(INFO_LOGGER_NAME)
        self.resultsLogger = logging.getLogger(RESULTS_LOGGER_NAME)
        self.config = configuration
        self.saveCleanedImages = saveCleanedImages

        if filterAugmentations:
            testDataset = TestDataset(self.config.testImageDir, augmentationTypes=self.config.augmentationTypes)
        else:
            testDataset = TestDataset(self.config.testImageDir)
        if self.config.modelName == ModelName.DENSE:
            batch_size = 32
        else:
            batch_size = 8

        self.testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=1)

        _, self.genStrikeToClean = getGeneratorModels(self.config)

        modelBasePath = self.config.outDir.parent
        state_dict = torch.load(modelBasePath / model_name, map_location=torch.device(self.config.device))
        if "model_state_dict" in state_dict.keys():
            state_dict = state_dict['model_state_dict']

        self.genStrikeToClean.load_state_dict(state_dict)
        self.genStrikeToClean = self.genStrikeToClean.to(self.config.device)

        self.logger.info('Data dir: %s', str(self.config.testImageDir))
        self.logger.info('Test dataset size: %d', len(testDataset))

    def run(self) -> None:
        """
        Inititates the testing process.

        Returns
        -------
            None
        """
        self.genStrikeToClean.eval()

        to_image = torchvision.transforms.ToPILImage()

        imgDir = self.config.outDir / "images"
        imgDir.mkdir(exist_ok=True, parents=True)

        results = []
        distortedSsims = []
        distortedRmses = []
        ssims = []
        rmses = []

        with torch.no_grad():
            for batch, datapoints in enumerate(self.testDataloader):
                print(batch)
                distortedImages = datapoints["distorted"].to(self.config.device)
                groundTruthImages = datapoints["original"]
                names = datapoints["name"]

                cleanedImages = self.genStrikeToClean(distortedImages)

                for idx in range(distortedImages.shape[0]):
                    distortedImage = distortedImages[idx]

                    distortedImage = distortedImage.permute(1, 2, 0).cpu().numpy()

                    cleanedImage = cleanedImages[idx]
                    if self.saveCleanedImages:
                        img = to_image(cleanedImage)
                        img.save(imgDir / '{}.png'.format(names[idx]))
                    cleanedImage = cleanedImage.permute(1, 2, 0).cpu().numpy()
                    groundTruth = groundTruthImages[idx].permute(1, 2, 0).cpu().numpy()

                    distortedRmse = normalized_root_mse(groundTruth, distortedImage)
                    distortedRmses.append(distortedRmse)
                    distortedSsim = structural_similarity(groundTruth, distortedImage, multichannel=True)
                    distortedSsims.append(distortedSsim)
                    cleanedRmse = normalized_root_mse(groundTruth, cleanedImage)
                    rmses.append(cleanedRmse)
                    cleanedSsim = structural_similarity(groundTruth, cleanedImage, multichannel=True)
                    ssims.append(cleanedSsim)

                    results.append({"image": names[idx], "distortedRmse": distortedRmse, "distortedSsim": distortedSsim,
                                       "cleanedRmse": cleanedRmse, "cleanedSsim": cleanedSsim})
        pd.DataFrame.from_records(results).to_csv(self.config.outDir / "results.csv", index=False)
        self.resultsLogger.info('Mean distorted SSIM: %f\t Stdev: %f', np.mean(distortedSsims), np.std(distortedSsims))
        self.resultsLogger.info('Mean distorted RMSE: %f\t Stdev: %f', np.mean(distortedRmses), np.std(distortedRmses))
        self.resultsLogger.info('Mean SSIM: %f\t Stdev: %f', np.mean(ssims), np.std(ssims))
        self.resultsLogger.info('Mean RMSE: %f\t Stdev: %f', np.mean(rmses), np.std(rmses))


if __name__ == "__main__":
    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-configfile", required=True, help="path to config file")
    cmdParser.add_argument("-data", required=True, help="path to data directory")
    cmdParser.add_argument("-model_name", required=False, help="name of the model checkpoint file, default: "
                                                               "genRestore_best_rmse.pth", default="rmse")
    cmdParser.add_argument("-save_images", required=False, help="saves cleaned images if given", default=False,
                           action="store_true")
    cmdParser.add_argument("-filter_aug", required=False, help="filters test images based on augmentations specified in"
                                                               " the config, otherwises uses all", default=False,
                           action="store_true")
    args = vars(cmdParser.parse_args())

    configPath = Path(args['configfile'])
    dataPath = Path(args['data'])

    configParser = configparser.ConfigParser()
    configParser.read(configPath)

    section = "DEFAULT"

    sections = configParser.sections()
    if len(sections) == 1:
        section = sections[0]
    else:
        logging.getLogger("st_recognition").warning(
            "Found %s than 1 section in config file. Using 'DEFAULT' as fallback.",
            'more' if len(sections) > 1 else 'fewer')

    parsedConfig = configParser[section]
    conf = Configuration(parsedConfig, test=True, fileSection=section)
    conf.testImageDir = dataPath

    out = configPath.parent / "{}_{}".format(dataPath.parent.name, dataPath.name)
    out.mkdir(exist_ok=True)
    conf.outDir = out

    initLoggers(conf)
    logger = logging.getLogger(INFO_LOGGER_NAME)
    logger.info(conf.outDir)

    runner = TestRunner(conf, saveCleanedImages=args["save_images"], filterAugmentations=args["filter_aug"],
                        model_name=args["model_name"])

    runner.run()
