import cv2
import numpy as np
from frameload import frameload
import os


office = os.path.dirname(__file__) + '/Office/img01_'
redchair = os.path.dirname(__file__) + "/RedChair/advbgst1_21_"


def temp_diff_filter(video, thresh=8, show_video=True, save_frame=-1):
    colorframes = {}
    grayframes = {}

    print("filtering video")
    i = 0
    for (frameix, image) in frameload(video):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayframes[frameix] = gray
        colorframes[frameix] = image
        if frameix < 3:
            continue

        preframe = grayframes[frameix - 2]
        nowframe = grayframes[frameix]

        difference = 0.5 * (-1 * preframe + nowframe)
        diffabs = np.abs(difference)

        bthreshed = np.where(diffabs > thresh, 1, 0)
        bthreshed3 = np.repeat(bthreshed[None, ...], 3, axis=0).transpose(1, 2, 0)

        graythresh = (128 * bthreshed).astype(np.uint8)
        maskedOriginal = bthreshed3 * colorframes[frameix - 1]
        maskedIm = maskedOriginal.astype(np.uint8)

        if show_video:
            cv2.imshow('gray source', grayframes[frameix - 1])
            cv2.imshow('bit mask', graythresh)  # derivative for frameix-1
            cv2.imshow('motion', maskedIm)
            cv2.waitKey(20)

        if i == save_frame:
            #print("saving frame %d" %i)
            (head,tail) = os.path.split(video)
            file = head+"_TD"+"_TH"+str(thresh)+"_test_frame_"+str(save_frame)
            cv2.imwrite(file+'_gray_source.jpg', grayframes[frameix - 1])
            cv2.imwrite(file+'_bit_mask.jpg', graythresh)
            cv2.imwrite(file+'_masked_source.jpg', maskedIm)
        i += 1

    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("done")


def main():
    temp_diff_filter(office, save_frame=69)


if __name__ == "__main__":
    main()