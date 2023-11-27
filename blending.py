import cv2 as cv
import numpy as np
import sys

P_LEVEL = 4


def g_pyr(img):
    img_ = img.copy()
    g = [img_]
    for i in range(P_LEVEL):
        img_ = cv.pyrDown(img_)
        g.append(img_)
    return g


def l_pyr(g):
    l = [g[-1]]
    for i in range(P_LEVEL, 0, -1):
        size = (g[i - 1].shape[1], g[i - 1].shape[0])
        ext = cv.pyrUp(g[i], dstsize=size)
        lap = cv.subtract(g[i-1], ext)
        l.append(lap)
    return l


def blend(L1, L2, GM):
    LS = []
    for la, lb, gm in zip(L1, L2, GM):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls.astype(np.uint8))
    return LS


def reconstruct(L):
    img_ = L[0]
    for i in range(P_LEVEL):
        size = (L[i+1].shape[1], L[i+1].shape[0])
        ext = cv.pyrUp(img_, dstsize=size)
        img_ = cv.add(L[i+1], ext)
    return img_


if __name__ == '__main__':
    img1 = cv.imread('images/apple.jpg')
    img2 = cv.imread('images/orange.jpg')

    roi = cv.selectROI(img1)
    mask = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
    mask[roi[1]: roi[1]+roi[3], roi[0]: roi[0]+roi[2]] = 1

    img2 = cv.resize(img2, dsize=(img1.shape[1], img1.shape[0]))

    gm = g_pyr(mask)
    gm.reverse()
    
    g1 = g_pyr(img1)
    g2 = g_pyr(img2)

    l1 = l_pyr(g1)
    l2 = l_pyr(g2)

    blended = blend(l1, l2, gm)
    final = reconstruct(blended)

    cv.imwrite("final1.jpg", final)
    cv.imshow("final_1", final)
    cv.waitKey(0)
    cv.destroyAllWindows()