import numpy as np
import os
import cv2
import math
from typing import Tuple


class frameload(object):
    def __init__(self, folder: str):
        self.folder = folder
        self.frame: int = 1


    def __iter__(self):
        return self


    def __next__(self) -> Tuple[int, np.ndarray]:

        fname = self.folder + str(self.frame).zfill(4) + ".jpg"
        if os.path.exists(fname):
            frameix = self.frame
            self.frame += 1
            return frameix, cv2.imread(fname)
        else:
            raise StopIteration()


def image_as_matrix(folder):
    data = []
    for (_, image) in frameload(folder):
        data.append(image)

    nmatrix = np.zeros([len(data), *data[0].shape])
    for i in range(0, len(data)):
        nmatrix[i, :, :, :] = data[i]
    return nmatrix


def image_color_to_gray(video: np.ndarray) -> np.ndarray:
    newarr = np.zeros(video.shape[:3])
    for imnum in range(0, video.shape[0]):
        frame = video[imnum].astype(np.uint8)
        newarr[imnum, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return newarr


def normalize(ker):
    return (1. / np.sum(ker)) * ker


def make_2d_gaussian(sigma):
    size = math.ceil(5*sigma)
    if size % 2 == 0:
        size += 1
    gauss = np.zeros([size, size])
    stepx = int((size - 1) / 2)
    for i in range(-stepx, stepx + 1):
        for j in range(-stepx, stepx + 1):
            mag = i ** 2 + j ** 2
            gauss[i+stepx, j+stepx] = np.exp(-(mag / (2 * (sigma ** 2))))
    # print(gauss)
    return normalize(gauss)
