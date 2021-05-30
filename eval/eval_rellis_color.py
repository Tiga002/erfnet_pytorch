# Code to produce colored segmentation output in Pytorch for RELLIS-3D Test Set
# May 2021
# Tiga Leung
#######################

import numpy as np
import torch
import os
import time
import importlib

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize

import sys
sys.path.append("..")
import custom_datasets

NUM_CHANNELS = 3
NUM_CLASSES = 20 #pascal=22, cityscapes=20
time_train = []

def main(args):
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")
    else:
        print ("Loading model: " + modelpath)
        print ("Loading weights: " + weightspath)

    # Import ERFNET
    model = ERFNet(NUM_CLASSES)
    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights LOADED successfully")

    # Set model to Evaluation mode
    model.eval()

    # Setup the dataset loader
    ### RELLIS-3D Dataloader
    enc = False
    loader_test = custom_datasets.setup_loaders(args, enc)
    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if (args.visualize):
        vis = visdom.Visdom()

    for step, (images, labels, img_name, _) in enumerate(loader_test):
        start_time = time.time()
        if (not args.cpu):
            images = images.cuda()
            #labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        label = outputs[0].max(0)[1].byte().cpu().data
        label_color = Colorize()(label.unsqueeze(0))

        eval_save_path = "./save_colour_rellis/"
        if not os.path.exists(eval_save_path):
            os.makedirs(eval_save_path)

        _, file_name = os.path.split(img_name[0])
        file_name = file_name + ".png"

        #image_transform(label.byte()).save(filenameSave)
        label_save = ToPILImage()(label_color)
        label_save.save(os.path.join(eval_save_path, file_name))

        if (args.visualize):
            vis.image(label_color.numpy())
        if step!=0:    #first run always takes some time for setup
            fwt = time.time() - start_time
            time_train.append(fwt)
            print ("Forward time per img (b=%d): %.3f (Mean: %.3f)" % (args.batch_size, fwt/args.batch_size, sum(time_train) / len(time_train) / args.batch_size))

        print (step, os.path.join(eval_save_path, file_name))

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--datadir', default=os.getenv("HOME") + "/dataset/Rellis-3D/")
    parser.add_argument('--loadDir',default="../save/erfnet_training_remote_v3/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="test")  #can be val, test, train, demoSequence
    parser.add_argument('--mode',type=str,default="test")

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--height', type=int, default=(360,640))


    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
