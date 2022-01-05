from copy import deepcopy
from typing import Tuple, List
from scipy import linalg
from tqdm import tqdm
import cv2
import numpy as np
import scipy.ndimage
import os

fp = os.path.dirname(os.path.realpath(__file__))


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
        corners_b: np.ndarray,
        im_a: np.ndarray,
        im_b: np.ndarray,
        window_size: int = 5,
        threshold: int = 0
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
                        tempcorrelations.append((ia, ja, ib, jb, score))
            # keep the top 30 correlations
            tempcorrelations.sort(key=lambda cor: -cor[4])
            correlations.extend(tempcorrelations[:1])
    return correlations


def harrisFeatures(gdata: np.ndarray, pref="unknown") -> Tuple[np.ndarray, np.ndarray]:
    gdatanorm = gdata.astype(np.int64)
    e = harris(gdatanorm)
    eharris = np.where(e < 22116052, 0, 128).astype(np.uint8)
    harrisOut = nms(e)

    harrisOut = np.logical_and(harrisOut, eharris)
    return gdatanorm, harrisOut


def getHomography(
        x1p: Tuple[Tuple[int, int], Tuple[int, int]],
        x2p: Tuple[Tuple[int, int], Tuple[int, int]],
        x3p: Tuple[Tuple[int, int], Tuple[int, int]],
        x4p: Tuple[Tuple[int, int], Tuple[int, int]]
) -> np.ndarray:
    rrefmatr = []
    avec = []
    for pointpair in [x1p, x2p, x3p, x4p]:
        ((x1, y1), (x2, y2)) = pointpair
        rrefmatr.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        rrefmatr.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
        avec.extend([x2, y2])

    rrefmatr = np.asarray(rrefmatr)
    avec = np.asarray(avec)
    try:
        sol = np.matmul(linalg.inv(rrefmatr), np.transpose(avec))
    except:
        return np.zeros([3, 3])
    return np.asarray([[sol[0], sol[1], sol[2]],
                       [sol[3], sol[4], sol[5]],
                       [sol[6], sol[7], 1]])


def scoreHomography(hg: np.ndarray, criteria: List[Tuple[Tuple[int, int], Tuple[int, int]]], tolerance: float = 1):
    return len(list(filter(lambda pointpair: satisfied(hg, pointpair, tolerance=tolerance), criteria)))


def least_squares_homography(pointpairs: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> np.ndarray:
    rrefmatr = []
    avec = []
    for pointpair in pointpairs:
        ((x1, y1), (x2, y2)) = pointpair
        rrefmatr.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        rrefmatr.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
        avec.extend([x2, y2])

    A = np.asarray(rrefmatr).astype(np.float64)
    B = np.asarray(avec).astype(np.float64)

    AtA = np.matmul(np.transpose(A), A).astype(np.float64)
    AtB = np.matmul(np.transpose(A), B).astype(np.float64)

    if linalg.det(AtA) == 0:
        raise AssertionError("Least Squares failed AtA is singular.")
    ## rrefmatr: (2N x 8)
    ## avec : (2N x 1)
    ## AT = (8 x 2N)
    ## ATA = (8x8)
    ## ATB = (8 x 1)
    sol = np.matmul(np.linalg.inv(AtA), AtB)

    q = np.asarray([[sol[0], sol[1], sol[2]],
                    [sol[3], sol[4], sol[5]],
                    [sol[6], sol[7], 1]])
    return q


def ransac_homogrpahy(
        correspondence: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        trials: int = 500,
        mindet_null: float = 0.2,
        radius: float = 1
):
    from random import shuffle
    bestscore = -1
    besthg: np.ndarray = None
    for trial in tqdm(range(0, trials), desc="computing homography"):
        shuffle(correspondence)
        points = correspondence[:4]
        homography = getHomography(*points)
        if linalg.det(homography) < mindet_null:
            continue  # the det sanity check failed probably a bad homography
        score = scoreHomography(homography, correspondence, tolerance=radius)
        if score > bestscore:
            bestscore = score
            besthg = homography
    print("best homography [det: %s], [matches: %s of %s] on radius of %s" % (
        linalg.det(besthg), bestscore, len(correspondence), radius))
    return bestscore, besthg


def homogenize(vec):
    q = deepcopy(vec)
    q[0] /= q[2]
    q[1] /= q[2]
    q[2] /= q[2]
    return q


def satisfied(hg: np.ndarray, pointpair: Tuple[Tuple[int, int], Tuple[int, int]], tolerance: float = 1) -> bool:
    source = np.asarray((*pointpair[0], 1))
    target = np.asarray((*pointpair[1], 1))
    transformed = homogenize(np.matmul(hg, source))
    return linalg.norm(transformed - target) < tolerance


def correlate(gray1: np.ndarray, gray2: np.ndarray, pref1='left', pref2='right'):
    g1, h1 = harrisFeatures(gray1, pref=pref1)
    g2, h2 = harrisFeatures(gray2, pref=pref2)
    corre: List[Tuple[int, int, int, int, float]] \
        = norm_cross_corr(h1, h2, g1, g2, window_size=11, threshold=0.3)

    corre_reshaped: List[Tuple[Tuple[int, int], Tuple[int, int]]] = list(
        map(lambda cr: ((cr[0], cr[1]), (cr[2], cr[3])), corre))
    (sc0, hg0) = ransac_homogrpahy(corre_reshaped, trials=1000, radius=20)
    # get rid of outliers
    (sc1, hg1) = ransac_homogrpahy(list(filter(lambda pp: satisfied(hg0, pp, tolerance=20), corre_reshaped)),
                                   trials=1000, radius=4)
    (sc2, hg2) = ransac_homogrpahy(list(filter(lambda pp: satisfied(hg1, pp, tolerance=4), corre_reshaped)),
                                   trials=500, radius=2)

    hg3 = least_squares_homography(list(filter(lambda pp: satisfied(hg2, pp, tolerance=2), corre_reshaped)))

    return hg3


def get_warped_boundaries(input_im_hg_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[
    Tuple[int, int, int, int], int]:
    bound_b, bound_r, bound_t, bound_l = (0, 0, 0, 0)
    for src, transform in input_im_hg_pairs:
        max_row, max_col, num_channels = src.shape
        corners = [(0, 0), (max_row - 1, 0), (0, max_col - 1), (max_row - 1, max_col - 1)]
        tf_inv = linalg.inv(transform)
        for corner_i, corner_j in corners:
            new_i, new_j, _ = homogenize(np.matmul(tf_inv, np.transpose(np.asarray([corner_i, corner_j, 1]))))
            if new_i > bound_b:
                bound_b = int(new_i)
            elif new_i < bound_t:
                bound_t = int(new_i)
            if new_j > bound_r:
                bound_r = int(new_j)
            elif new_j < bound_l:
                bound_l = int(new_j)
    return (bound_b, bound_r, bound_t, bound_l), num_channels


def mergeImages(input_im_hg_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    (bound_b, bound_r, bound_t, bound_l), num_channels = get_warped_boundaries(input_im_hg_pairs)
    num_srcs = len(input_im_hg_pairs)

    out_mat = np.zeros((bound_b - bound_t + 1, bound_r - bound_l + 1, num_channels))
    temp_mat = np.zeros((bound_b - bound_t + 1, bound_r - bound_l + 1, num_srcs, num_channels))
    feather_weights = np.zeros((bound_b - bound_t + 1, bound_r - bound_l + 1, num_srcs, num_channels))
    for warp_row in tqdm(range(bound_t, bound_b), desc="merging images"):
        for warp_col in range(bound_l, bound_r):
            out_row = warp_row - bound_t
            out_col = warp_col - bound_l
            for src_idx, (src, transform) in enumerate(input_im_hg_pairs):
                (from_row, from_col, _) = homogenize(
                    np.matmul(transform, np.transpose(np.asarray([warp_row, warp_col, 1]))))
                if (0 <= from_row < src.shape[0]) and (0 <= from_col < src.shape[1]):
                    # no interp
                    # out_mat[out_row, out_col] = src[int(fr)][int(from_col)]

                    # bilinear interp
                    i = int(from_row)
                    j = int(from_col)
                    a = from_row - i
                    b = from_col - j
                    p00 = (1 - a) * (1 - b) * src[i, j]
                    if i + 1 < src.shape[0] and j + 1 < src.shape[1]:
                        p10 = a * (1 - b) * src[i + 1, j]
                        p01 = (1 - a) * b * src[i, j + 1]
                        p11 = a * b * src[i + 1, j + 1]
                    elif i + 1 >= src.shape[0] and j + 1 < src.shape[1]:
                        p10 = a * (1 - b) * src[i, j]
                        p01 = (1 - a) * b * src[i, j + 1]
                        p11 = a * b * src[i, j + 1]
                    elif i + 1 < src.shape[0] and j + 1 >= src.shape[1]:
                        p10 = a * (1 - b) * src[i + 1, j]
                        p01 = (1 - a) * b * src[i, j]
                        p11 = a * b * src[i + 1, j]
                    else:
                        p10 = a * (1 - b) * src[i, j]
                        p01 = (1 - a) * b * src[i, j]
                        p11 = a * b * src[i, j]
                    temp_mat[out_row, out_col, src_idx] = p00 + p10 + p01 + p11

                    # feathering
                    feather_weights[out_row, out_col, src_idx] = np.min([
                        from_row,
                        src.shape[0] - 1 - from_row,
                        from_col,
                        src.shape[1] - 1 - from_col
                    ])
    # feathering, cont
    out_mat = np.divide(np.sum(np.multiply(temp_mat, feather_weights), axis=2), np.sum(feather_weights, axis=2))

    return out_mat


def main():
    fpairs = [
        ("hallway", ("DanaHallWay1/DSC_0281.JPG", "DanaHallWay1/DSC_0282.JPG")),
        ("office", ("DanaOffice/DSC_0308.JPG", "DanaOffice/DSC_0309.JPG"))]

    for (ftag, (f1, f2)) in fpairs:
        i1 = cv2.imread(f"{fp}/{f1}")
        gdata1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        gdata1 = cv2.GaussianBlur(gdata1, (5, 5), cv2.BORDER_DEFAULT)

        i2 = cv2.imread(f"{fp}/{f2}")

        gdata2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        gdata2 = cv2.GaussianBlur(gdata2, (5, 5), cv2.BORDER_DEFAULT)

        hg = correlate(gdata1, gdata2)

        warped_image = mergeImages([[i1, linalg.inv(hg)], [i2, np.identity(3)]])

        # cv2.imshow(f"warped {ftag} images", warped_image.astype(np.uint8))
        cv2.imwrite(f'{fp}/figures/mosaic/{ftag}_mosaic_example.jpg', warped_image.astype(np.uint8))

    # cv2.waitKey(0)


if __name__ == "__main__":
    main()
