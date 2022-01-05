from frameload import *
import cv2
import math
import os


office = os.path.dirname(__file__) + '/Office/img01_'
redchair = os.path.dirname(__file__) + "/RedChair/advbgst1_21_"


def gaussian(x, mu, sigma):
    g = (1. / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / sigma ** 2)
    return g


def derivgaussian(x, mu, sigma):
    return (-x / (sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def dgaussiank(k, mu, sigma):
    box = np.zeros([k])
    for i in range(0, k):
        offset = (k - 1) / 2
        val = i - offset
        # print(val)
        box[i] = derivgaussian(val, mu, sigma)
    return box


def dog_filter(video, tsigma=1.0, thresh=8, show_video=True, save_frame=-1):
    data = image_as_matrix(video)

    gray = image_color_to_gray(data)

    size = math.ceil(5*tsigma)
    if size % 2 == 0:
        size += 1

    derivg = dgaussiank(size, 0, tsigma)

    lost_frames = size - 1
    newsize = [gray.shape[0] - lost_frames, gray.shape[1], gray.shape[2]]
    filteredim = np.zeros(newsize)
    maskedim = np.zeros(newsize)
    graysliced = gray[int(lost_frames / 2):-int(lost_frames / 2), :, :]

    for px in range(0, gray.shape[1]):
        for py in range(0, gray.shape[2]):
            histogram = gray[:, px, py]
            filtered = np.abs(np.convolve(derivg, histogram, "valid"))
            filteredim[:, px, py] = filtered

            bthreshed = np.where(filtered > thresh, 1, 0)
            maskedim[:, px, py] = bthreshed

    #print(graysliced.shape)
    #print(maskedim.shape)
    #print(filteredim.shape)

    maskedcombined = maskedim * graysliced

    print("filtering video")
    for i in range(0, graysliced.shape[0]):
        nframe = graysliced[i, :, :]
        mask = 255 * maskedim[i, :, :]
        combined = maskedcombined[i, :, :]

        if show_video:
            cv2.imshow('gray source', nframe.astype(np.uint8))
            cv2.imshow('bit mask', mask.astype(np.uint8))
            cv2.imshow('motion', combined.astype(np.uint8))
            cv2.waitKey(20)

        if i == save_frame:
            #print("saving frame %d" %i)
            (head,tail) = os.path.split(video)
            file = head+"DoG"+str(tsigma)+"_TH"+str(thresh)+"_test_frame_"+str(save_frame)
            cv2.imwrite(file+'_gray_source.jpg', nframe.astype(np.uint8))
            cv2.imwrite(file+'_bit_mask.jpg', mask.astype(np.uint8))
            cv2.imwrite(file+'_masked_source.jpg', combined.astype(np.uint8))
            
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("done")


def main():
    dog_filter(video=office,save_frame=69)


if __name__ == "__main__":
    main()
