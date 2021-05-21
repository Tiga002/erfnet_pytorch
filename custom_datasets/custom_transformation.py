from PIL import Image, ImageOps
from torchvision.transforms import Compose, CenterCrop, Resize, Normalize, Pad, RandomHorizontalFlip
import torchvision.transforms as transforms
import random
import torch
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class JointAugmentation(object):
    def __init__(self, training_encoder, augment=True, height=512):
        self.training_encoder = training_encoder
        self.augment = augment
        self.height = height
        pass

    def __call__(self, image, label):
        # 1: Resize Both image and label || Training/Validation/Test all need resize
        image = transforms.Resize(self.height, Image.BILINEAR)(image) # (819, 512)
        label = transforms.Resize(self.height, Image.NEAREST)(label)    # (819, 512)

        if(self.augment): # For training set only
            # 1: Random Horizontal Flip
            prob_hflip = random.random()
            if (prob_hflip < 0.5):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

            # 2: Random Resized and crop (819+-2, 512+-2)
            transX = random.randint(-2,2)
            transY = random.randint(-2,2)

            image = ImageOps.expand(image, border=(transX,transY,0,0), fill=0)
            label = ImageOps.expand(label, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            image = image.crop((0, 0, image.size[0]-transX, image.size[1]-transY))
            label = label.crop((0, 0, label.size[0]-transX, label.size[1]-transY))

        return image, label

class LabelTransform(object):
    def __init__(self, training_encoder, height=512):
        self.training_encoder = training_encoder
        self.height = height

    def __call__(self, label):
        # If training the encoder with rellis-3d dataset (separately)
        if (self.training_encoder):
            # Further resize the label into (102, 64)
            label = Resize(int(self.height/8), Image.NEAREST)(label)

        # Convert label (PIL image) into Tensor
            #torch.Size([1, 512+-2, 89+-2]) or torch.Size([1, 64, 102])
        label = torch.from_numpy(np.array(label)).long().unsqueeze(0)
        #label = transforms.ToTensor()(label)

        # Replace 255 with 19 on the label
        label = Relabel(255,19)(label)

        return label
