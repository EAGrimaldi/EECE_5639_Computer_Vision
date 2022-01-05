import colorsys
from copy import deepcopy
from typing import Tuple, List, Iterator
from scipy import linalg
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import scipy.ndimage
import os

fp = os.path.dirname(os.path.realpath(__file__))

PC = Tuple[Tuple[int, int], Tuple[int, int]]


def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))


def colorHash(correspond: PC):
    ang = abs(hash(correspond)) % 360
    return hsv2rgb(ang / 360, 0.5, 0.5)


def sobel(inp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    ix = scipy.ndimage.convolve(inp, sx)
    iy = scipy.ndimage.convolve(inp, sy)
    return ix, iy


def cmatrix(inp: np.ndarray, neighborhood: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    (ix, iy) = sobel(inp)

    ixy = ix * iy
    ixx = ix * ix
    iyy = iy * iy

    window = np.ones([neighborhood, neighborhood])

    return (scipy.ndimage.convolve(ixx, window),
            scipy.ndimage.convolve(ixy, window),
            scipy.ndimage.convolve(iyy, window))


def harris(inp: np.ndarray, k=0.04) -> np.ndarray:
    (sumixx, sumixy, sumiyy) = cmatrix(inp, 9)

    trace = sumixx + sumiyy
    det = (sumixx * sumiyy) - (sumixy ** 2)
    return det - k * (trace ** 2)


def nms(inp: np.ndarray, window=5) -> np.ndarray:
    nmsed = np.zeros(inp.shape)
    (rmax, cmax) = inp.shape
    q = np.amin(inp)
    loss: int = int((window - 1) / 2)
    for r in range(loss, rmax - loss):
        for c in range(loss, cmax - loss):
            area = inp[r - loss:r + loss + 1, c - loss:c + loss + 1]
            preamax = np.amax(area)
            if preamax == inp[r, c]:
                nmsed[r, c] = 1
    return nmsed


def nonzero_idx(arr: np.ndarray) -> List[Tuple[int, int]]:
    idx = []
    x, y = arr.shape
    for i in range(x):
        for j in range(y):
            if arr[i, j] > 0:
                idx.append((i, j))
    return idx


def norm_cross_corr(
        corners_a: np.ndarray,
        corners_b: np.ndarray, im_a: np.ndarray,
        im_b: np.ndarray,
        window_size: int = 5, threshold: int = 0
) -> List[Tuple[int, int, int, int, float]]:
    correlations = []
    x, y = corners_a.shape
    halfwin = int((window_size - 1) / 2)
    c_idx_a = nonzero_idx(corners_a)
    c_idx_b = nonzero_idx(corners_b)
    for ia, ja in c_idx_a:
        if halfwin <= ia < x - halfwin and halfwin <= ja < y - halfwin:
            window_a = deepcopy(im_a[ia - halfwin:ia + halfwin + 1, ja - halfwin:ja + halfwin + 1])
            window_a = window_a / np.linalg.norm(window_a)
            tempcorrelations = []
            for ib, jb in c_idx_b:
                if halfwin <= ib < x - halfwin and halfwin <= jb < y - halfwin:
                    window_b = deepcopy(im_b[ib - halfwin:ib + halfwin + 1, jb - halfwin:jb + halfwin + 1])
                    window_b = window_b / np.linalg.norm(window_b)
                    score = np.sum(window_a * window_b)
                    if score > threshold:
                        tempcorrelations.append((ia, ja, ib, jb, float(score)))
            # keep the top 30 correlations
            tempcorrelations.sort(key=lambda cor: -cor[4])
            correlations.extend(tempcorrelations[:1])
    return correlations


def ncorr(im1: np.ndarray, im2: np.ndarray) -> float:
    aux1 = deepcopy(im1)
    aux2 = deepcopy(im2)
    aux1 = aux1 / np.linalg.norm(aux1)
    aux2 = aux2 / np.linalg.norm(aux2)
    return float(np.sum(aux1 * aux2))


def findBestPatch(template1: np.ndarray, c_lo: int, c_hi: int, g2: np.ndarray, line: np.ndarray, width: int = 11):
    if width % 2 != 1:
        raise ValueError("width must be odd, got %s" % width)
    norm_template = deepcopy(template1)
    norm_template = norm_template / np.linalg.norm(norm_template)

    best_match: Tuple[float, Tuple[int, int]] = (-100, (-1, -1))

    hw = int((width - 1) / 2)

    for col in range(max(hw, c_lo), min(g2.shape[1] - hw - 1, c_hi), 1):
        tr = int((-line[2] - (col * line[1])) / line[0])
        if tr - hw >= 0 and tr + hw + 1 < g2.shape[0]:

            cnr = g2[tr - hw:tr + hw + 1, col - hw:col + hw + 1]
            patch_score = ncorr(norm_template, cnr)
            # print(tr, col, patch_score)
            if patch_score > best_match[0]:
                best_match = (patch_score, (tr, col))

    return best_match


def harrisFeatures(gdata: np.ndarray, thresh: int = 22116052) -> Tuple[np.ndarray, np.ndarray]:
    gdatanorm = gdata.astype(np.int64)
    e = harris(gdatanorm)
    eharris = np.where(e < thresh, 0, 128).astype(np.uint8)
    harrisOut = nms(e)

    harrisOut = np.logical_and(harrisOut, eharris)
    return gdatanorm, harrisOut


def homogenize(vec):
    q = deepcopy(vec)
    q[0] /= q[2]
    q[1] /= q[2]
    q[2] /= q[2]
    return q


def ransac_fundamental(
        corre: List[PC],
        trials: int = 500,
        radius: float = 1) -> Tuple[np.ndarray, int]:
    from random import shuffle
    bestscore = -1
    bestFundamental: np.ndarray = None
    for trial in tqdm(range(0, trials), desc="fundamental RANSAC[tolerance: %s]" % radius):
        shuffle(corre)
        points = corre[:8]
        fundamental = getFundamental(points)
        score = scoreFundamental(fundamental, corre, tolerance=radius)

        if bestscore < score:
            bestscore = score
            bestFundamental = fundamental

    return bestFundamental, bestscore


def distance_to_fundamental(fmatr: np.ndarray, pc: PC) -> float:
    ((xl, yl), (xr, yr)) = pc
    line = np.matmul(np.asarray([xr, yr, 1]), fmatr)
    line = line / np.linalg.norm(line)

    return np.abs(np.matmul(line, np.transpose(np.asarray([xl, yl, 1])))) / np.sqrt(
        np.square(line[0]) + np.square(line[1]))


def filter_pc_to_tolerance(fmatr: np.ndarray, corre: List[PC], tolerance: float = 1) -> Iterator[PC]:
    return filter(lambda pc: distance_to_fundamental(fmatr, pc) <= tolerance, corre)


def scoreFundamental(fmatr: np.ndarray, corre: List[PC], tolerance: float = 1) -> int:
    return len(list(filter_pc_to_tolerance(fmatr, corre, tolerance=tolerance)))


def getFundamental(points: List[PC]) -> np.ndarray:
    if len(points) != 8:
        raise ValueError("need 8 points for fundamental est, got %s" % len(points))
    rrefmatr = []
    for ((x1, y1), (x1p, y1p)) in points:
        rrefmatr.append([x1 * x1p, x1 * y1p, x1, y1 * x1p, y1 * y1p, y1, x1p, y1p, 1])

    u, s, vh = np.linalg.svd(np.asarray(rrefmatr).astype(np.float64), full_matrices=True)

    vh = vh.T

    last_col = vh[:, vh.shape[1] - 1]  # last col of v

    if len(last_col) != 9:
        raise ValueError("last col is not of right size :/ %s " % len(last_col))

    F = np.asarray([[last_col[0], last_col[3], last_col[6]],
                    [last_col[1], last_col[4], last_col[7]],
                    [last_col[2], last_col[5], last_col[8]]])

    u, s, vh = np.linalg.svd(F, full_matrices=True)
    s[s.shape[0] - 1] = 0
    F_reconstructed = np.matmul(np.matmul(u, np.diag(s)), vh)

    return F_reconstructed


def correlate_fundamental(i1, i2, gray1: np.ndarray, gray2: np.ndarray, header : str = "unk") -> Tuple[np.ndarray, int]:
    g1, h1 = harrisFeatures(gray1)
    g2, h2 = harrisFeatures(gray2)
    corre: List[Tuple[int, int, int, int, float]] \
        = norm_cross_corr(h1, h2, g1, g2, window_size=11, threshold=0.3)
    corre_reshaped: List[Tuple[Tuple[int, int], Tuple[int, int]]] = list(
        map(lambda cr: ((cr[0], cr[1]), (cr[2], cr[3])), corre))

    (f_matr_v1, ign) = ransac_fundamental(corre_reshaped, 200, 4)
    print("best fundamental matrix: [%s satisfied]" % ign)
    (f_matr_v2, ign_v2) = ransac_fundamental(list(filter_pc_to_tolerance(f_matr_v1, corre_reshaped, 4)),
                                             500, 0.5)
    print("best fundamental matrix: [%s satisfied]" % ign_v2)
    (f_matr_v3, best_score) = ransac_fundamental(list(filter_pc_to_tolerance(f_matr_v2, corre_reshaped, 1)),
                                                 1000, 0.05)
    print("best fundamental matrix: [%s satisfied]" % best_score)

    xoffset = i1.shape[1]

    bigim_postF = np.concatenate((i1, i2), axis=1)
    bigim_preF = deepcopy(bigim_postF)

    for pointpair in corre_reshaped:
        (l, r) = pointpair
        cv2.line(bigim_preF, (l[1], l[0]), (r[1] + xoffset, r[0]), list(colorHash(pointpair)), thickness=1)

    for pointpair in list(filter_pc_to_tolerance(f_matr_v3, corre_reshaped, 0.05)):
        (l, r) = pointpair
        cv2.line(bigim_postF, (l[1], l[0]), (r[1] + xoffset, r[0]), list(colorHash(pointpair)), thickness=1)

    cv2.imwrite(f"ransac_figs/{header}_preFundamental.png", bigim_preF)
    cv2.imwrite(f"ransac_figs/{header}_postFundamental.png", bigim_postF)
    return f_matr_v3, best_score


def compute_disparity(f_matrix: np.ndarray, gray1: np.ndarray, gray2: np.ndarray, header: str = "unk"):
    win_width = 11
    hw = int((win_width - 1) / 2)

    x_dispa = np.zeros(gray1.shape)
    y_dispa = np.zeros(gray1.shape)
    patch_acc = np.zeros(gray1.shape)
    invalid = np.zeros(gray1.shape)

    for tr in tqdm(range(hw, gray1.shape[0] - hw - 1), "Computing disparities (rows)"):
        prior_disp = None
        for tc in range(hw, gray1.shape[1] - hw - 1):
            tmp = deepcopy(gray1[tr - hw: tr + hw + 1, tc - hw: tc + hw + 1])
            line = np.matmul(f_matrix, np.asarray([tr, tc, 1]))
            line = line / np.linalg.norm(line)
            if prior_disp is None:
                (sc, (i2r, i2c)) = findBestPatch(tmp, tc - 100, tc + 100, gray2, line, win_width)
            else:
                prior_disp = min(max(prior_disp, -100), 100)
                # optimize to 60 instead of 200
                (sc, (i2r, i2c)) = findBestPatch(tmp, tc + prior_disp - 15, tc + prior_disp + 15, gray2, line,
                                                 win_width)
                # print(i2c - tc, prior_disp)
                if abs((i2c - tc) - prior_disp) > 10:  # much further from our intial guess
                    if i2c < tc + prior_disp:
                        (sc, (i2r, i2c)) = findBestPatch(tmp, i2c - 50, i2c + 10, gray2, line, win_width)
                    else:
                        (sc, (i2r, i2c)) = findBestPatch(tmp, i2c - 10, i2c + 50, gray2, line, win_width)

                if sc < 0.995:
                    (sc, (i2r, i2c)) = findBestPatch(tmp, tc - 200, tc + 200, gray2, line, win_width)

            if sc < 0.995:
                invalid[tr, tc] = 1
            patch_acc[tr, tc] = sc
            x_dispa[tr, tc] = tr - i2r
            y_dispa[tr, tc] = tc - i2c
            # prior_disp = i2c - tc

    plt.imsave("figures/%s_patch_disparities.png" % header, patch_acc, cmap='hot')
    plt.imsave("figures/%s_x_disparities.png" % header, x_dispa, cmap='hot')
    plt.imsave("figures/%s_y_disparities.png" % header, y_dispa, cmap='hot')
    with open("disparity_map_data/%s_dispdata.npy" % header, "wb") as f:
        np.save(f, patch_acc)
        np.save(f, x_dispa)
        np.save(f, y_dispa)
        np.save(f, invalid)
        f.close()


def draw_pixelpair(f_matrix: np.ndarray,
                   im_r0: int,
                   im_c0: int,
                   i1: np.ndarray,
                   i2: np.ndarray,
                   gray1: np.ndarray,
                   gray2: np.ndarray,
                   width: int = 11,
                   dens_lo: int = -100,
                   dens_hi: int = 100,
                   header: str = "unk"):
    hw = int((width - 1) / 2)
    tmp = deepcopy(gray1[im_r0 - hw: im_r0 + hw + 1, im_c0 - hw: im_c0 + hw + 1])
    line = np.matmul(f_matrix, np.asarray([im_r0, im_c0, 1]))
    line = line / np.linalg.norm(line)
    (sc, (i2r, i2c)) = findBestPatch(tmp, im_c0 + dens_lo, im_c0 + dens_hi, gray2, line, width)

    xoffset = i1.shape[1]
    imstacks = np.concatenate((i1, i2), axis=1)

    c_init = -line[2] / line[0]
    c_fin = (-line[2] - (line[1] * i2.shape[1])) / line[0]
    cv2.line(imstacks, (xoffset, int(c_init)), (xoffset + i2.shape[1], int(c_fin)), (0, 255, 255), thickness=1)
    cv2.rectangle(imstacks, (im_c0 - hw, im_r0 - hw), (im_c0 + hw, im_r0 + hw), (255, 255), thickness=1)
    cv2.rectangle(imstacks, (i2c - hw + xoffset, i2r - hw), (i2c + hw + xoffset, i2r + hw), (255, 255), thickness=1)
    print(im_r0, im_c0, i2r, i2c, sc, im_c0 - i2c)
    cv2.imwrite(f"figures/{header}_pixpair_at_{im_r0}_{im_c0}.png", imstacks)
    return im_r0 - i2r, im_c0 - i2c


def main():
    fpairs = [
        ("cones", ("assets/Cones_im2.jpg", "assets/Cones_im6.jpg", -70, -5, 15, 60)),
        ("prison", ("assets/cast-left-1.jpg", "assets/cast-right.jpg", -80, -40, 20, 80))
    ]

    for (ftag, (lfile, rfile, dl, dh, bl, bh)) in fpairs:
        print(f"{ftag}")
        i1 = cv2.imread(f"{fp}/{lfile}")
        gdata1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        gdata1 = cv2.GaussianBlur(gdata1, (5, 5), cv2.BORDER_DEFAULT)

        i2 = cv2.imread(f"{fp}/{rfile}")
        gdata2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        gdata2 = cv2.GaussianBlur(gdata2, (5, 5), cv2.BORDER_DEFAULT)

        (f_matr, sat) = correlate_fundamental(i1, i2, gdata1, gdata2, header=ftag)
        # compute_disparity(f_matr, gdata1, gdata2, header=ftag)

        ps = 5
        a = np.zeros((int(i1.shape[0] / ps) + 2, int(i1.shape[1] / ps) + 2))
        b = np.zeros(a.shape)
        c = np.zeros((int(i1.shape[0] / ps) + 2, int(i1.shape[1] / ps) + 2, 3))
        for i in range(ps, i1.shape[0] - ps, ps):
            for j in range(ps, i1.shape[1] - ps, ps):
                (rowdisp, coldisp) = draw_pixelpair(f_matr, i, j, i1, i2, gdata1, gdata2,
                                                    dens_lo=dl, dens_hi=dh, header=ftag)
                a[a.shape[0] - int(i / ps), int(j / ps)] = coldisp
                b[b.shape[0] - int(i / ps), int(j / ps)] = rowdisp
                from math import atan2, pi, hypot
                ang = int(atan2(rowdisp, coldisp) * (180 / pi))
                mag = hypot(rowdisp, coldisp)

                c[int(i / ps), int(j / ps)] = hsv2rgb(ang, mag / bh, 0.8)

        plt.colorbar(plt.pcolor(np.clip(a, bl, bh)))
        plt.savefig(f"disparity_figures/{ftag}_dispmap_horiz.png", dpi=100)
        plt.close()

        plt.colorbar(plt.pcolor(np.clip(b, bl, bh)))
        plt.savefig(f"disparity_figures/{ftag}_dispmap_vert.png", dpi=100)
        cv2.imwrite(f"disparity_figures/{ftag}_dispmap_vec.png", c.astype(np.uint8))

        # draw_pixelpair(f_matr, 320, 300, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 33, 284, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 75, 60, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 139, 106, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 300, 20, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 200, 350, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 400, 400, i1, i2, gdata1, gdata2, header=ftag)

        # draw_pixelpair(f_matr, 25, 164, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 42, 114, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 219, 5, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 219, 25, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 219, 45, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 219, 65, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 200, 200, i1, i2, gdata1, gdata2, header=ftag)
        # draw_pixelpair(f_matr, 150, 150, i1, i2, gdata1, gdata2, header=ftag)


if __name__ == '__main__':
    main()
