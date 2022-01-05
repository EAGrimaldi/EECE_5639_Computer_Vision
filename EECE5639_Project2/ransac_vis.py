from copy import deepcopy
from typing import Tuple, List
import colorsys
import cv2
import p2_main
import numpy as np
import os


fp = os.path.dirname(os.path.realpath(__file__))


def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))


def colorHash(correspond: Tuple[Tuple[int, int], Tuple[int, int]]):
    ang = abs(hash(correspond)) % 360
    return hsv2rgb(ang / 360, 0.5, 0.5)


def main():
    fpairs = [
        ("hallway", ("DanaHallWay1/DSC_0281.JPG", "DanaHallWay1/DSC_0282.JPG")),
        ("office", ("DanaOffice/DSC_0308.JPG", "DanaOffice/DSC_0309.JPG"))]

    for (ftag, (f1, f2)) in fpairs:
        i1 = cv2.imread(fp+'/'+f1)
        gdata1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gdata1, (5, 5), cv2.BORDER_DEFAULT)

        i2 = cv2.imread(fp+'/'+f2)
        gdata2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gdata2, (5, 5), cv2.BORDER_DEFAULT)

        g1, h1 = p2_main.harrisFeatures(gray1)
        g2, h2 = p2_main.harrisFeatures(gray2)
        corre: List[Tuple[int, int, int, int, float]] \
            = p2_main.norm_cross_corr(h1, h2, g1, g2, window_size=11, threshold=0.3)

        corre_reshaped: List[Tuple[Tuple[int, int], Tuple[int, int]]] = list(
            map(lambda cr: ((cr[0], cr[1]), (cr[2], cr[3])), corre))
        (sc0, hg0) = p2_main.ransac_homogrpahy(corre_reshaped, trials=1000, radius=20)
        # get rid of outliers
        (sc1, hg1) = p2_main.ransac_homogrpahy(
            list(filter(lambda pp: p2_main.satisfied(hg0, pp, tolerance=20), corre_reshaped)),
            trials=1000, radius=4)
        (sc2, hg2) = p2_main.ransac_homogrpahy(
            list(filter(lambda pp: p2_main.satisfied(hg1, pp, tolerance=4), corre_reshaped)),
            trials=500, radius=2)

        xoffset = i1.shape[1]

        bigim_ncorr = np.concatenate((i1, i2), axis=1)
        bigim_r20 = deepcopy(bigim_ncorr)
        bigim_r4 = deepcopy(bigim_ncorr)
        bigim_r2 = deepcopy(bigim_ncorr)

        for pointpair in corre_reshaped:
            (l, r) = pointpair
            cv2.line(bigim_ncorr, (l[1], l[0]), (r[1] + xoffset, r[0]), list(colorHash(pointpair)), thickness=1)

        for pointpair in filter(lambda pp: p2_main.satisfied(hg0, pp, tolerance=20), corre_reshaped):
            (l, r) = pointpair
            cv2.line(bigim_r20, (l[1], l[0]), (r[1] + xoffset, r[0]), list(colorHash(pointpair)), thickness=1)

        for pointpair in filter(lambda pp: p2_main.satisfied(hg1, pp, tolerance=4), corre_reshaped):
            (l, r) = pointpair
            cv2.line(bigim_r4, (l[1], l[0]), (r[1] + xoffset, r[0]), list(colorHash(pointpair)), thickness=1)

        for pointpair in filter(lambda pp: p2_main.satisfied(hg2, pp, tolerance=2), corre_reshaped):
            (l, r) = pointpair
            cv2.line(bigim_r2, (l[1], l[0]), (r[1] + xoffset, r[0]), list(colorHash(pointpair)), thickness=1)

        cv2.imwrite(f"{fp}/figures/ransac/{ftag}_ncorr.jpg", bigim_ncorr)
        cv2.imwrite(f"{fp}/figures/ransac/{ftag}_r20.jpg", bigim_r20)
        cv2.imwrite(f"{fp}/figures/ransac/{ftag}_r4.jpg", bigim_r4)
        cv2.imwrite(f"{fp}/figures/ransac/{ftag}_r2.jpg", bigim_r2)


if __name__ == '__main__':
    main()
