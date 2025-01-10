import cv2
import numpy as np
from typing import Tuple

def get_rect_maps(
    lcam_mat = None, 
    rcam_mat = None, 
    rmat = None,
    tvec = None,
    ldist_coeffs = None,
    rdist_coeffs = None,
    img_size: Tuple[int, int] = (1280, 1024),
    triangular_intrinsics: bool = False,
    mode: str = 'conventional'
    ) -> dict:
    if mode == 'conventional':
        if triangular_intrinsics:
            lcam_mat = np.array([[lcam_mat[0, 0], 0, lcam_mat[0, 2]], [0, lcam_mat[1, 1], lcam_mat[1, 2]], [0, 0, 1]], dtype=np.float64)
            rcam_mat = np.array([[rcam_mat[0, 0], 0, rcam_mat[0, 2]], [0, rcam_mat[1, 1], rcam_mat[1, 2]], [0, 0, 1]], dtype=np.float64)

        # compute pixel mappings
        img_size = [int(img_size[0]), int(img_size[1])]
        r1, r2, p1, p2, q, valid_pix_roi1, valid_pix_roi2 = cv2.stereoRectify(cameraMatrix1=lcam_mat.astype('float64'), distCoeffs1=ldist_coeffs.astype('float64'),
                                                                            cameraMatrix2=rcam_mat.astype('float64'), distCoeffs2=rdist_coeffs.astype('float64'),
                                                                            imageSize=tuple(img_size), R=rmat.astype('float64'), T=tvec.T.astype('float64'),
                                                                            alpha=0)

        lmap1, lmap2 = cv2.initUndistortRectifyMap(cameraMatrix=lcam_mat, distCoeffs=ldist_coeffs, R=r1, newCameraMatrix=p1, size=tuple(img_size), m1type=cv2.CV_32FC1)
        rmap1, rmap2 = cv2.initUndistortRectifyMap(cameraMatrix=rcam_mat, distCoeffs=ldist_coeffs, R=r2, newCameraMatrix=p2, size=tuple(img_size), m1type=cv2.CV_32FC1)
        maps = {'lmap1': lmap1,
                'lmap2': lmap2,
                'rmap1': rmap1,
                'rmap2': rmap2}
    elif mode == 'pseudo':
        maps = {}
        p1 = lcam_mat.astype('float64')
        p2 = rcam_mat.astype('float64')
    else:
        raise NotImplementedError

    return maps, p1, p2, q


def rectify_pair(limg, rimg, maps, method='nearest'):

    cv_interpol = cv2.INTER_NEAREST if method == 'nearest' else cv2.INTER_CUBIC


    limg_rect = cv2.remap(np.copy(limg), maps['lmap1'], maps['lmap2'], interpolation=cv_interpol)
    rimg_rect = cv2.remap(np.copy(rimg), maps['rmap1'], maps['rmap2'], interpolation=cv_interpol)

    return limg_rect, rimg_rect

def pseudo_rectify(rimg, x0, x1):

    tmat = np.array(((1, 0, x0-x1), (0, 1, 0))).astype(np.float32)
    rimg_rect = cv2.warpAffine(rimg, tmat, (rimg.shape[1], rimg.shape[0]))

    return rimg_rect

def pseudo_rectify_2d(rimg, x0, x1, y0, y1):

    tmat = np.array(((1, 0, x0-x1), (0, 1, y0-y1))).astype(np.float32)
    rimg_rect = cv2.warpAffine(rimg, tmat, (rimg.shape[1], rimg.shape[0]))

    return rimg_rect
