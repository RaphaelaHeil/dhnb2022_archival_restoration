"""
Code to train a CycleGAN to remove degradations from manuscript images.
"""
import itertools
import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
from skimage.metrics import structural_similarity, normalized_root_mse
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from . import configuration, dataset
from .configuration import getConfiguration
from .network import image_pool
from .network.initialise import init_weights
from .utils import getGeneratorModels, getDiscriminatorModels, composeAugmentations

INFO_LOGGER_NAME = "st_removal"
CLEAN_DISC_LOGGER_NAME = "cdLoss"
DISTORTED_DISC_LOGGER_NAME = "ddLoss"
C_TO_D_GEN_LOGGER_NAME = "ctodLoss"
D_TO_C_GEN_LOGGER_NAME = "dtocLoss"
VALIDATION_LOGGER_NAME = "validation"


def initLoggers(config: configuration.Configuration) -> None:
    """
    Utility function initialising a default info logger, as well as several loss loggers.

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

    formatter = logging.Formatter('%(asctime)s - %(message)s', '%d-%b-%y %H:%M:%S')

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(config.outDir / "info.log")
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    for network in [CLEAN_DISC_LOGGER_NAME, DISTORTED_DISC_LOGGER_NAME, C_TO_D_GEN_LOGGER_NAME, D_TO_C_GEN_LOGGER_NAME,
        VALIDATION_LOGGER_NAME]:
        networkLogger = logging.getLogger(network)
        networkLogger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler(config.outDir / "{}.log".format(network))
        fileHandler.setLevel(logging.INFO)
        networkLogger.addHandler(fileHandler)


class TrainRunner:
    """
    Utility class that wraps the initialisation, training and validation steps of the training run.
    """

    def __init__(self, config: configuration.Configuration):
        self.logger = logging.getLogger(INFO_LOGGER_NAME)
        self.ctosLogger = logging.getLogger(C_TO_D_GEN_LOGGER_NAME)
        self.stocLogger = logging.getLogger(D_TO_C_GEN_LOGGER_NAME)
        self.cdLogger = logging.getLogger(CLEAN_DISC_LOGGER_NAME)
        self.sdLogger = logging.getLogger(DISTORTED_DISC_LOGGER_NAME)
        self.valLogger = logging.getLogger(VALIDATION_LOGGER_NAME)

        self.config = config

        augmentations = composeAugmentations(self.config)

        trainDataset = dataset.ArtificialNoisePatchDataset(self.config.trainImageDir,
                                                           alteringAugmentation=augmentations,
                                                           cropSize=(self.config.imageWidth, self.config.imageHeight),
                                                           patchCount=self.config.trainCount)

        validationDataset = dataset.ValidationDataset(self.config.testImageDir,
                                                      augmentationTypes=self.config.augmentationTypes)

        self.trainDataLoader = DataLoader(trainDataset, batch_size=self.config.batchSize, shuffle=True, num_workers=0)
        self.validationDataloader = DataLoader(validationDataset, batch_size=10, shuffle=False, num_workers=0)

        self.logger.info('Model: %s', self.config.modelName.name)
        self.logger.info('Data dir: %s', str(self.config.trainImageDir))
        self.logger.info('Train dataset size: %d', len(trainDataset))
        self.logger.info('Validation dataset size: %d', len(validationDataset))

        self.distort, self.restore = getGeneratorModels(self.config)

        self.distort = self.distort.to(self.config.device)
        init_weights(self.distort)
        self.restore = self.restore.to(self.config.device)
        init_weights(self.restore)

        self.cleanDiscriminator, self.distortedDiscriminator = getDiscriminatorModels()
        self.distortedDiscriminator = self.distortedDiscriminator.to(self.config.device)
        init_weights(self.distortedDiscriminator)
        self.cleanDiscriminator = self.cleanDiscriminator.to(self.config.device)
        init_weights(self.cleanDiscriminator)

        self.generatorOptimiser = torch.optim.Adam(
            itertools.chain(self.distort.parameters(), self.restore.parameters()),
            lr=self.config.generatorLearningRate, betas=self.config.generatorBetas)

        self.discriminatorOptimiser = torch.optim.Adam(
            itertools.chain(self.distortedDiscriminator.parameters(), self.cleanDiscriminator.parameters()),
            lr=self.config.discriminatorLearningRate, betas=self.config.discriminatorBetas)

        self.discriminator_criterion = nn.MSELoss()
        self.image_l1_criterion = nn.L1Loss()

        self.fake_clean_pool = image_pool.ImagePool(self.config.poolSize)
        self.fake_struck_pool = image_pool.ImagePool(self.config.poolSize)

        self.cnn_loss_criterion = nn.L1Loss()

        self.bestRmse = float('inf')
        self.bestRmseEpoch = 0
        self.bestSSIM = float('-inf')
        self.bestSSIMEpoch = 0

    def run(self) -> None:
        """
        Initiates the training process.

        Returns
        -------
            None
        """
        self.logger.info(self.config.device)
        self.valLogger.info("rmse, ssim")

        self.logger.info('-- Started training --')

        for epoch in range(1, self.config.epochs + 1):
            self.trainOneEpoch(epoch)
            if self.config.validationEpoch > 0 and epoch % self.config.validationEpoch == 0:
                self.validateOneEpoch(epoch)
        self.logger.info("best rmse: %f (%d), best ssim: %f (%d)", self.bestRmse, self.bestRmseEpoch,
                         self.bestSSIM, self.bestSSIMEpoch)

    def validateOneEpoch(self, epoch: int) -> None:
        """
        Validates the neural network at the current training stage.

        Parameters
        ----------
        epoch : int
            current epoch number

        Returns
        -------
            None
        """
        rmses = []
        ssims = []
        epochdir = None

        imageSaveCounter = 0

        self.distort.eval()
        self.restore.eval()
        self.distortedDiscriminator.eval()
        self.cleanDiscriminator.eval()
        with torch.no_grad():
            for batch_id, datapoints in enumerate(self.validationDataloader):
                imgOutputTensor = torch.Tensor().to(self.config.device)
                cleanImages = datapoints["original"].to(self.config.device)
                distortedImages = datapoints["distorted"].to(self.config.device)

                generatedDistorted = self.distort(cleanImages)
                generatedClean = self.restore(distortedImages)

                for i in range(len(datapoints["distorted"])):
                    currentClean = cleanImages[i]
                    currentDistorted = distortedImages[i]
                    currentGeneratedClean = generatedClean[i]
                    currentGeneratedDistorted = generatedDistorted[i]

                    tmpOut = torch.cat(
                        (currentClean, currentDistorted, currentGeneratedClean, currentGeneratedDistorted), 1)

                    currentClean = currentClean.permute(1, 2, 0)
                    currentGeneratedClean = currentGeneratedClean.permute(1, 2, 0)

                    imgOutputTensor = torch.cat((imgOutputTensor, tmpOut), 2).to(self.config.device)

                    currentClean = currentClean.cpu().numpy()
                    currentGeneratedClean = currentGeneratedClean.cpu().numpy()
                    rmses.append(normalized_root_mse(currentClean, currentGeneratedClean))
                    ssims.append(structural_similarity(currentClean, currentGeneratedClean, multichannel=True))

                if self.config.imageSaveInterval > 0 and epoch % (
                        self.config.imageSaveInterval * self.config.validationEpoch) == 0:
                    if not epochdir:
                        epochdir = self.config.outDir / str(epoch)
                        epochdir.mkdir(exist_ok=True, parents=True)
                    save_image(imgOutputTensor, epochdir / "e{}_b{}.png".format(epoch, batch_id),
                               nrow=1)
                    imageSaveCounter += 1

        meanRMSE = np.mean(rmses)
        meanSSIM = np.mean(ssims)
        self.logger.info('val [%d/%d], rmse: %f, ssim: %f', epoch, self.config.epochs, meanRMSE, meanSSIM)

        self.valLogger.info("%f,%f", meanRMSE, meanSSIM)

        if meanRMSE < self.bestRmse:
            self.bestRmseEpoch = epoch
            torch.save({'epoch': epoch, 'model_state_dict': self.restore.state_dict(), },
                       self.config.outDir / Path('genRestore_best_rmse.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.distort.state_dict(), },
                       self.config.outDir / Path('genDistort_best_rmse.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.distortedDiscriminator.state_dict(), },
                       self.config.outDir / Path('distortedDiscriminator_best_rmse.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.cleanDiscriminator.state_dict(), },
                       self.config.outDir / Path('cleanDiscriminator_best_rmse.pth'))
            self.logger.info('%d: Updated best rmse model', epoch)
            self.bestRmse = meanRMSE

        if meanSSIM > self.bestSSIM:
            self.bestSSIMEpoch = epoch
            torch.save({'epoch': epoch, 'model_state_dict': self.restore.state_dict(), },
                       self.config.outDir / Path('genRestore_best_ssim.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.distort.state_dict(), },
                       self.config.outDir / Path('genDistort_best_ssim.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.distortedDiscriminator.state_dict(), },
                       self.config.outDir / Path('distortedDiscriminator_best_ssim.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.cleanDiscriminator.state_dict(), },
                       self.config.outDir / Path('cleanDiscriminator_best_ssim.pth'))
            self.logger.info('%d: Updated best ssim model', epoch)
            self.bestSSIM = meanSSIM

    def trainOneEpoch(self, epoch: int) -> None:
        """
        Trains the neural network for one epoch.

        Parameters
        ----------
        epoch : int
            current epoch number

        Returns
        -------
            None
        """
        self.distort.train()
        self.restore.train()
        self.distortedDiscriminator.train()
        self.cleanDiscriminator.train()
        epochStartTime = time.time()
        totalDiscriminatorLossClean = []
        totalDiscriminatorLossDistorted = []
        totalCleanToDistortedGeneratorLoss = []
        totalDistortedToCleanGeneratorLoss = []
        for batch_id, datapoints in enumerate(self.trainDataLoader):
            genDistortedToCleanCycleLoss, genCleanToDistortedCycleLoss, discriminatorLossClean, \
                discriminatorLossDistorted = (self.trainOneBatch(batch_id, datapoints, epoch))
            totalDiscriminatorLossClean.append(discriminatorLossClean)
            totalDiscriminatorLossDistorted.append(discriminatorLossDistorted)
            totalDistortedToCleanGeneratorLoss.append(genDistortedToCleanCycleLoss)
            totalCleanToDistortedGeneratorLoss.append(genCleanToDistortedCycleLoss)

        run_time = time.time() - epochStartTime
        self.logger.info('epoch [%d/%d], discriminator losses: clean %f, distorted %f,'
                         ' generator losses: clean to distorted %f, distorted to clean %f, time:%f', epoch,
                         self.config.epochs, np.mean(totalDiscriminatorLossClean),
                         np.mean(totalDiscriminatorLossDistorted), np.mean(totalCleanToDistortedGeneratorLoss),
                         np.mean(totalDistortedToCleanGeneratorLoss), run_time)

        if epoch > 1 and self.config.modelSaveEpoch > 0 and epoch % self.config.modelSaveEpoch == 0:
            torch.save(self.distort.state_dict(),
                       self.config.outDir / Path('genDistort_epoch_{}.pth'.format(epoch)))
            torch.save(self.restore.state_dict(),
                       self.config.outDir / Path('genRestore_epoch_{}.pth'.format(epoch)))
            torch.save(self.cleanDiscriminator.state_dict(),
                       self.config.outDir / Path('cleanDiscriminator_e_{}.pth'.format(epoch)))
            torch.save(self.distortedDiscriminator.state_dict(),
                       self.config.outDir / Path('distortedDiscriminator_e_{}.pth'.format(epoch)))

    def trainOneBatch(self, batchID: int, datapoints: Dict[str, Any], epoch: int) -> Tuple[float, float, float, float]:
        """
        Trains the neural network on a single batch.

        Parameters
        ----------
        batchID : int
            current batch number
        datapoints : Dict[str, Any]
            current batch of datapoints
        epoch : int
            current epoch number

        Returns
        -------
        Tuple[float, float, float, float]
            distortedToCleanCycleLoss, cleanToDistortedCycleLoss, discriminatorLossClean, discriminatorLossDistorted
        """
        self.generatorOptimiser.zero_grad()

        clean = datapoints["clean"]
        struck = datapoints["distorted"]

        clean = clean.to(self.config.device)
        struck = struck.to(self.config.device)

        (generatedClean, generatedDistorted, distortedToCleanCycleLoss,
        cleanToDistortedCycleLoss) = self.trainGenerators(clean, struck)

        self.stocLogger.info("%d,%d,%f", epoch, batchID, distortedToCleanCycleLoss)
        self.ctosLogger.info("%d,%d,%f", epoch, batchID, cleanToDistortedCycleLoss)

        discriminatorLossDistorted, discriminatorLossClean = self.trainDiscriminators(generatedClean, clean,
                                                                                      generatedDistorted, struck)

        self.sdLogger.info("%d,%d,%f", epoch, batchID, discriminatorLossDistorted)
        self.cdLogger.info("%d,%d,%f", epoch, batchID, discriminatorLossClean)

        return distortedToCleanCycleLoss, cleanToDistortedCycleLoss, discriminatorLossClean, discriminatorLossDistorted

    def trainGenerators(self, clean: torch.Tensor, distorted: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Trains the CycleGAN generators for one batch.

        Parameters
        ----------
        clean : torch.Tensor
            batch of clean images
        distorted : torch.Tensor
            batch of distorted image

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, float, float]
                generated clean manuscript image, generated distorted manuscript image, distortedToCleanCycleLoss,
                cleanToDistortedCycleLoss
        """
        # forward first part:
        generatedDistorted = self.distort(clean)

        # forward second part:
        generatedClean = self.restore(distorted)

        # resconstruction:
        cycledClean = self.restore(generatedDistorted)
        cycledDistorted = self.distort(generatedClean)

        # identity:
        if self.config.identityLambda > 0.0:
            cleanIdentity = self.restore(clean)
            disortedIdentity = self.distort(distorted)
            cleanIdentityLoss = self.image_l1_criterion(cleanIdentity,
                                                        clean) * self.config.identityLambda * self.config.distortedLambda
            distortedIdentityLoss = self.image_l1_criterion(disortedIdentity,
                                                            distorted) * self.config.identityLambda * self.config.cleanLambda
        else:
            cleanIdentityLoss = 0.0
            distortedIdentityLoss = 0.0

        self.setDiscriminatorsRequiresGrad(False)

        distortedDiscrimination = self.distortedDiscriminator(generatedDistorted)

        cleanDiscrimination = self.cleanDiscriminator(generatedClean)

        lossDistortedDiscriminator = self.discriminator_criterion(distortedDiscrimination,
                                                                  torch.ones_like(distortedDiscrimination).to(
                                                                      self.config.device))
        lossCleanDiscriminator = self.discriminator_criterion(cleanDiscrimination,
                                                              torch.ones_like(cleanDiscrimination).to(
                                                                  self.config.device))

        distortedToCleanCycleLoss = self.image_l1_criterion(cycledClean, clean) * self.config.cleanLambda

        cleanToDistortedCycleLoss = self.image_l1_criterion(cycledDistorted, distorted) * self.config.distortedLambda

        totalGeneratorLoss = (cleanToDistortedCycleLoss + distortedToCleanCycleLoss + distortedIdentityLoss +
                              cleanIdentityLoss + lossDistortedDiscriminator + lossCleanDiscriminator)

        totalGeneratorLoss.backward()
        self.generatorOptimiser.step()

        return generatedClean, generatedDistorted, distortedToCleanCycleLoss.item(), cleanToDistortedCycleLoss.item()

    def trainDiscriminators(self, generatedClean: torch.Tensor, cleanImages: torch.Tensor,
                            generatedDistorted: torch.Tensor, distortedImages: torch.Tensor) -> Tuple[float, float]:
        """
        Trains the discriminators for one batch.

        Parameters
        ----------
        generatedClean : torch.Tensor
            batch of generated clean images
        cleanImages : torch.Tensor
            batch of original clean input images
        generatedDistorted : torch.Tensor
            batch of generated distorted images
        distortedImages : torch.Tensor
            batch of original distorted input images

        Returns
        -------
        Tuple[float, float]
            distorted discriminator loss, clean discriminator loss
        """
        self.setDiscriminatorsRequiresGrad(True)
        self.discriminatorOptimiser.zero_grad()

        fakeDistorted = self.fake_struck_pool.query(generatedDistorted.detach())
        realDistortedPrediction = self.distortedDiscriminator(distortedImages)
        fakeDistortedPrediction = self.distortedDiscriminator(fakeDistorted)

        fakeDistortedLoss = self.discriminator_criterion(fakeDistortedPrediction,
                                                         torch.zeros_like(fakeDistortedPrediction).to(
                                                             self.config.device))
        realDistortedLoss = self.discriminator_criterion(realDistortedPrediction,
                                                         torch.ones_like(realDistortedPrediction).to(
                                                             self.config.device))
        discriminatorLossDistorted = (fakeDistortedLoss + realDistortedLoss) * 0.5
        discriminatorLossDistorted.backward()

        fakeClean = self.fake_clean_pool.query(generatedClean.detach())

        realCleanPrediction = self.cleanDiscriminator(cleanImages)
        fakeCleanPrediction = self.cleanDiscriminator(fakeClean)

        fakeCleanLoss = self.discriminator_criterion(fakeCleanPrediction,
                                                     torch.zeros_like(fakeCleanPrediction).to(self.config.device))
        realCleanLoss = self.discriminator_criterion(realCleanPrediction,
                                                     torch.ones_like(realCleanPrediction).to(self.config.device))
        discriminatorLossClean = (fakeCleanLoss + realCleanLoss) * 0.5
        discriminatorLossClean.backward()

        self.discriminatorOptimiser.step()

        return discriminatorLossDistorted.item(), discriminatorLossClean.item()

    def setDiscriminatorsRequiresGrad(self, requiresGrad: bool) -> None:
        """
        Switches 'requires_grad' flag both discriminators according to :param:`requiresGrad`

        Parameters
        ----------
        requiresGrad : bool
            value to be propagated to 'requires_grad' of both discriminators

        Returns
        -------
        None
        """
        self.distortedDiscriminator.requires_grad = requiresGrad
        self.cleanDiscriminator.requires_grad = requiresGrad


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    conf = getConfiguration()
    initLoggers(conf)
    logger = logging.getLogger(INFO_LOGGER_NAME)
    logger.info(conf.fileSection)
    runner = TrainRunner(conf)
    runner.run()
