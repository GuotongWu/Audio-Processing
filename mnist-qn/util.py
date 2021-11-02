import torch
import torchvision
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset


class PostIDLocator():
    """
    This is used to locate postid in a full picture
    """
    def __init__(self, threshold=0.45, shrink=0):
        self.threshold = threshold
        self.shrink = shrink

    def locate(self, im):
        """
        Function takes raw cv2.imread() and binar-ified
        (H, W)
        """

        # Focus on left-upper half
        im = im[:im.shape[0] // 2, :im.shape[1] // 2, :]

        # Cast into gray
        im = im.mean(axis=2) / 255.0
        
        # Cast into Black and White (Black as Background)
        im = cv2.threshold(im, self.threshold, 1, cv2.THRESH_BINARY)[1]

        # Right Up edge Detection using Boolean Approach

        im_right = np.roll(im, shift=(1,), axis=(1,))
        im_up  = np.roll(im, shift=(-1,), axis=(0,))

        im_right = np.logical_or(im, im_right)
        im_up = np.logical_or(im, im_up)

        im_right = np.logical_xor(im, im_right)
        im_up = np.logical_xor(im, im_up)

        im = np.logical_or(im_right, im_up)
        
        idxW = np.sort(im.sum(axis=0).argsort()[-12:])
        idxH = np.sort(im.sum(axis=1).argsort()[-2:])

        return [
            (
                idxW[i]+self.shrink, 
                idxH[0]+self.shrink, 
                idxW[i+1]-self.shrink, 
                idxH[1]-self.shrink
            ) for i in range(0, len(idxW), 2)]

class Binarify(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super(Binarify, self).__init__()
        self.threshold = threshold
    
    def forward(self, img):
        return Image.fromarray(1-cv2.threshold(np.asarray(img)/255.0, self.threshold, 1, cv2.THRESH_BINARY)[1])
        
class EnvelopDataset(Dataset):
    def __init__(self, dir:str, threshold=0.45, shrink=0):
        self.dir = dir
        self.file_list = os.listdir(dir)
        self.locator = PostIDLocator(threshold=threshold, shrink=shrink)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.Grayscale(num_output_channels=1),
            Binarify(threshold=0.5),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.file_list[idx])
        im = Image.open(path)
        points = self.locator.locate(cv2.imread(path))
        return torch.stack([self.transform(im.crop(item)) for item in points]), path

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.Sequence = torch.nn.Sequential(
            # (1, 28, 28)
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),

            # (32, 28, 28)
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # (32, 14, 14)
            torch.nn.ReLU(),

            # (32, 14, 14)
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),

            # (64, 14, 14)
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # (64, 7, 7)
            torch.nn.ReLU(),

            # (64, 7, 7)
            torch.nn.Flatten(start_dim=1),

            # (64 * 7 * 7 = 3136)
            torch.nn.Linear(in_features=3136, out_features=1024),

            # (1024, )
            torch.nn.ReLU(),

            # (1024, )
            torch.nn.Linear(in_features=1024, out_features=10),

            # (10, )
            torch.nn.Softmax(),
        )
        

    def forward(self, x):
        return self.Sequence(x)