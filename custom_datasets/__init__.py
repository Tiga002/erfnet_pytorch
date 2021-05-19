from custom_datasets import rellis
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def setup_loaders(args):
    """
    Input: argument passed by user
    Outpu: training data loader, validation data loader, testing data loader
    """
    train_batch_size = args.batch_size
    validate_batch_size = 1
    mean_std = ((0.496588, 0.59493099, 0.53358843), (0.496588, 0.59493099, 0.53358843))

    # Geometric Image Transforms
    train_image_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size,(args.scale_min,args.scale_max)),
        transforms.Resize(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(args.color_aug, args.color_aug, args.color_aug, args.color_aug),
        #transforms.GaussianBlur(4,),
        transforms.ToTensor(),
        transforms.Normalize((0.496588, 0.59493099, 0.53358843), (0.496588, 0.59493099, 0.53358843))])

    validate_image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.496588, 0.59493099, 0.53358843), (0.496588, 0.59493099, 0.53358843))])

    test_image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.496588, 0.59493099, 0.53358843), (0.496588, 0.59493099, 0.53358843))])

    """
    Create Dataset Instances:
    If: mode = train THEN
        return train loader and validate loader
    Else if: mode = test THEN,
        return test loader
    Else:
        raise error
    """

    if  args.mode == 'train':
        training_set = rellis.Rellis(
            'train',
            datadir = args.datadir,
            image_transforms = train_image_transforms,
            cv_split = args.cv)

        validation_set = rellis.Rellis(
            'val',
            datadir = args.datadir,
            image_transforms = validate_image_transforms,
            cv_split = args.cv)

        training_set_loader = DataLoader(training_set,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=True)

        validation_set_loader = DataLoader(validation_set,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           shuffle=False)
        return training_set_loader, validation_set_loader

    elif args.mode == 'test':
        testing_set = rellis.Rellis(
            mode = 'test',
            datadir = args.datadir,
            image_transforms = test_image_transforms,
            cv_split = args.cv)

        testing_set_loader = DataLoader(testing_set,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=False)
        return testing_set_loader
    else:
        raise
