from custom_datasets import rellis
from custom_datasets.custom_transformation import JointAugmentation, LabelTransform
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def setup_loaders(args, enc=False):
    ## DEBUG:
    print("---enc = {}".format(enc))
    """
    Input: argument passed by user
    Outpu: training data loader, validation data loader, testing data loader
    """
    validate_batch_size = 1
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
        ##########################  Training Set ##########################
        train_image_label_transforms = JointAugmentation(enc, augment=True, height=args.height)
        train_image_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=args.color_aug, contrast=args.color_aug,
                saturation=args.color_aug, hue=args.color_aug),
            #transforms.GaussianBlur(4,),
            transforms.ToTensor(),
            transforms.Normalize((0.496588, 0.59493099, 0.53358843), (0.496588, 0.59493099, 0.53358843))])
        train_label_transforms = LabelTransform(enc, height=args.height)

        training_set = rellis.Rellis(
            'train',
            datadir = args.datadir,
            joint_image_label_transforms = train_image_label_transforms,
            image_transforms = train_image_transforms,
            label_transforms = train_label_transforms,
            cv_split = args.cv)

        training_set_loader = DataLoader(training_set,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=True)

        ##########################  Validation Set ##########################
        validate_image_label_transforms = JointAugmentation(enc, augment=False, height=args.height)
        validate_image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.496588, 0.59493099, 0.53358843), (0.496588, 0.59493099, 0.53358843))])
        validate_label_transforms = LabelTransform(enc, height=args.height)

        validation_set = rellis.Rellis(
            'val',
            datadir = args.datadir,
            joint_image_label_transforms = validate_image_label_transforms,
            image_transforms = validate_image_transforms,
            label_transforms = validate_label_transforms,
            cv_split = args.cv)

        validation_set_loader = DataLoader(validation_set,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           shuffle=False)

        return training_set_loader, validation_set_loader

    elif args.mode == 'test':
        ##########################  Testing Set ##########################
        test_image_label_transforms = JointAugmentation(enc, augment=False, height=args.height)
        test_image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.496588, 0.59493099, 0.53358843), (0.496588, 0.59493099, 0.53358843))])
        test_label_transforms = LabelTransform(enc, height=args.height)

        testing_set = rellis.Rellis(
            mode = 'test',
            datadir = args.datadir,
            joint_image_label_transforms = test_image_label_transforms,
            image_transforms = test_image_transforms,
            label_transforms = test_label_transforms,
            cv_split = args.cv)

        testing_set_loader = DataLoader(testing_set,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=False)
        return testing_set_loader
    else:
        raise
