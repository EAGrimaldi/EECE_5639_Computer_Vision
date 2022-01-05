from frameload import *
import numpy as np
import cv2
import os
from PIL import Image

office = os.path.dirname(__file__) + '/Office/img01_'
redchair = os.path.dirname(__file__) + "/RedChair/advbgst1_21_"

def spatial_smoothing(video, box_size=0, ssigma=0):
    if (box_size==0) and (ssigma==0):
        print("you need to pick a smoothing filter")
        return
    elif (box_size != 0) and (ssigma!=0):
        print("you need to pick only one smoothing filter")
        return
    elif (box_size<0) or (ssigma<0):
        print("you need to use positive values")
        return
    elif box_size>0:
        smooth_ker = normalize(np.ones([box_size, box_size]))
        filter_type = "box"+str(box_size)+"x"+str(box_size)
    elif ssigma>0:
        smooth_ker = make_2d_gaussian(ssigma)
        filter_type = "gaussian"+str(ssigma)

    data = image_as_matrix(video)
    gray_unsmoothed = image_color_to_gray(data)

    (head, tail) = os.path.split(video)
    new_head = head+'/'+filter_type+'/'
    os.mkdir(new_head)
    for i in range(0, gray_unsmoothed.shape[0]):
        gray = cv2.filter2D(gray_unsmoothed[i], -1, smooth_ker)
        gray_im = Image.fromarray(gray)
        fp=new_head+tail+str(i+1).zfill(4)+'.jpg'
        rgb_im = gray_im.convert('RGB')
        rgb_im.save(fp)


def main():
    video = office
    spatial_smoothing(video=video, box_size=3)
    spatial_smoothing(video=video, box_size=5)
    spatial_smoothing(video=video, ssigma=1.0)
    spatial_smoothing(video=video, ssigma=1.4)
    spatial_smoothing(video=video, ssigma=1.8)
    spatial_smoothing(video=video, ssigma=2.2)

if __name__ == "__main__":
    main()
