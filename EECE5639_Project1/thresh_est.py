from frameload import *
import numpy as np
import cv2

office = os.path.dirname(__file__) + '/Office/img01_'
redchair = os.path.dirname(__file__) + "/RedChair/advbgst1_21_"


def std_dev_est(video):
    data = image_as_matrix(video)
    gray = image_color_to_gray(data)

    gray_td = np.zeros(shape=(gray.shape[0]-2,gray.shape[1],gray.shape[2]))

    for i in range(1,gray.shape[0]-1):
        for j in range(gray.shape[1]):
            for k in range(gray.shape[2]):
                gray_td[i-1,j,k] = 0.5*gray[i-1,j,k] - 0.5*gray[i+1,j,k]

    # gray_td = np.diff(a=gray, n=1, axis=0) # much faster, but not quite the same calculation

    std_dev_per_pixel = np.std(a=gray_td, axis=0)

    ave_std_dev = np.average(std_dev_per_pixel)

    print("std dev of data: %d" %ave_std_dev)

    return ave_std_dev

def main():
    video = office

    sigma = std_dev_est(video)
    thresh_lower = 2*sigma
    thresh_upper = 3*sigma
    print("suggested threshold range: %d, %d" %(thresh_upper, thresh_lower))

if __name__ == "__main__":
    main()
