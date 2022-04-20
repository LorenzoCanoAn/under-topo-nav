import pickle
import torch.utils.data as data_utils
from torchvision import transforms
import torch
import os
import random
import numpy as np

#-------------------------------------------------------------------
#	 definition of the  class
#-------------------------------------------------------------------
class ImageDataset(data_utils.Dataset):
    def __init__(self, path_to_dataset, do_augment=True):
        self.do_augment = do_augment
        self.transforms = transforms.RandomApply(torch.nn.ModuleList([
            transforms.RandomErasing(p=0.5, scale=(0.0, 0.01), ratio=(0.5, 1), value=0, inplace=False),
            transforms.RandomErasing(p=0.5, scale=(0.0, 0.01), ratio=(0.5, 1), value=0, inplace=False)
            ]), p=0.5)

        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)

        self.device = torch.device(
            "cuda:0")
        self.load_dataset(path_to_dataset)

    def load_dataset(self, dataset_folder):
        self.folder = dataset_folder
        self.dir_elements = os.listdir(dataset_folder)
        self.len = self.dir_elements.__len__()
        print(f"Loading dataset: {self.__len__()} elements")
        for idx in range(self.len):
            print("\r", end="")
            print(f"{idx + 1}", end="")
            img_path = os.path.join(self.folder, "{}.pickle".format(idx))
            with open(img_path, "rb") as f:
                x, y = pickle.load(f)
            
            if idx == 0:
                s = x.shape
                self.new_image_shape = (1, s[0], s[1])
                self.images = torch.zeros((self.len, 1, s[0], s[1])).float()
                s = y.shape
                self.labels = torch.zeros((self.len, s[0])).float()

            x = torch.reshape(x, self.new_image_shape)
            self.images[idx, ...] = x
            self.labels[idx, ...] = y
        print()


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = self.images[idx, ...].float()
        result = self.labels[idx, ...]

        if self.do_augment:
            image, result = self.augment(image, result)

        return image.float(), result

    def augment(self, image, result):
        # HORIZONTAL FLIP
        if random.randint(0, 100) < 20:
            image = self.horizontal_flip(image)
            result = self.horizontal_flip(
                torch.reshape(result, (1, -1))).flatten()

        # HORIZONTAL SHIFT
        n = random.randint(-20, 20)
        image = torch.roll(image, n, dims=2)
        result = torch.roll(result, n)

        # VERTICAL SHIFT
        n = random.randint(-2, 2)
        image = torch.roll(image, n, dims=0)

        # GAUSSIAN NOISE
        image += np.random.normal(0, random.uniform(0,
                                  0.02), (16, 720)).astype(float)
        image /= torch.max(image)
        image[image < 0] = 0

        # TORCH DATA AUG
        image = self.transforms(image)

        return image, result

