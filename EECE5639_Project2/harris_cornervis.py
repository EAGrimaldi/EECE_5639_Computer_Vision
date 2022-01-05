from copy import deepcopy
import cv2
import p2_main
import os

fp = os.path.dirname(os.path.realpath(__file__))


def main():
    files = \
        [
            "DanaHallWay1/DSC_0281.JPG",
            "DanaHallWay1/DSC_0282.JPG",
            "DanaOffice/DSC_0309.JPG",
            "DanaOffice/DSC_0308.JPG",
            "DanaOffice/DSC_0310.JPG"
        ]
    for file in files:
        i1 = cv2.imread(f"{fp}/{file}")
        gdata1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        gdata1 = cv2.GaussianBlur(gdata1, (5, 5), cv2.BORDER_DEFAULT)

        (data, hf) = p2_main.harrisFeatures(gdata1)

        grayim_with_features = cv2.cvtColor(cv2.cvtColor(deepcopy(i1), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        for (ir, ic) in p2_main.nonzero_idx(hf):
            a = 1  # radius of feature dot in figure
            grayim_with_features[ir - a:ir + a, ic - a:ic + a, :] = [0, 255, 0]

        full_name = file.replace("/", "_")
        # cv2.imshow(f"harris features: {full_name}", colorim)
        cv2.imwrite(f"{fp}/figures/harris/{full_name}_features.jpg", grayim_with_features)
        print(f"{fp}/figures/harris/{full_name}_features.jpg")
    # cv2.waitKey(0)


if __name__ == '__main__':
    main()
