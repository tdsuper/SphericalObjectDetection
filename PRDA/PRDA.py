import cv2
import numpy as np
import os
from lib.ComputerPoint import get_corner
from lib.tools import ro_Shpbbox, roPoint, rotate_image
from lib.ImageRecorder import ImageRecorder

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def pxpy2xyz(point, w=1024, h=512):
    point[0] = ((point[0] + 0.5) / w - 0.5) * 2 * np.pi
    point[1] = ((point[1] + 0.5) / h - 0.5) * np.pi

    sin_u = np.sin(point[0])
    cos_u = np.cos(point[0])
    sin_v = np.sin(point[1])
    cos_v = np.cos(point[1])

    x = cos_v * sin_u
    y = sin_v
    z = cos_v * cos_u
    return np.array([x, y, z])


def get_angle(a, b):
    M = np.dot(a, b)
    N = np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) * np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
    theta = np.arccos(M / N)
    return theta / np.pi * 180

def get_new_angle(pl, pm, pn):
    B = pxpy2xyz(pl)
    A_ = pxpy2xyz(pm)
    C_ = pxpy2xyz(pn)

    V_ = np.dot(np.dot(B, C_), C_)
    V = np.dot(np.dot(A_, C_), C_)

    BV = B - V
    A_V = A_ - V
    k = np.cross(BV, A_V)
    k = np.true_divide(k, np.linalg.norm(k) + 0.01)
    C_ = np.true_divide(C_, np.linalg.norm(C_) + 0.01)

    angle = get_angle(BV, A_V)

    if k[0] > 0:
        if C_[0] > 0:
            angle = -angle
        else:
            angle = angle
    elif k[0] <= 0:
        if C_[0] > 0:
            angle = angle
        else:
            angle = -angle

    return angle


def get_new_RBFoV(gt, ro_an, n, erp_w=1024, erp_h=512):
    corner = get_corner(gt)
    corner = roPoint(corner, -ro_an, n, erp_w, erp_h)

    cx = (gt[0][0] + np.pi) / (2 * np.pi) * erp_w
    cy = -((gt[0][1] + np.pi / 2) / np.pi * erp_h) + erp_h
    px, py = roPoint([cx, cy], -ro_an, n, erp_w, erp_h)
    theta = px / erp_w * (2 * np.pi) - np.pi
    phi = -py / erp_h * np.pi + np.pi / 2
    gt_new = np.array([[theta, phi, gt[0][2], gt[0][3], 0]])
    pl = get_corner(gt_new)
    pm = corner
    if pl[0] == pm[0] and pl[1] == pm[1]:
        return gt

    pn = [px, py]

    new_angle = get_new_angle(pl, pm, pn)
    gt[..., 0] = theta
    gt[..., 1] = phi
    gt[..., 4] = new_angle
    return gt

def visualization(img, new_gt, erp_w=1024, erp_h=512):
    BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=new_gt[0][2], view_angle_h=new_gt[0][3],
                         long_side=erp_w)
    Px, Py = BFoV._sample_points(new_gt[0][0], new_gt[0][1], border_only=True)
    Px, Py = ro_Shpbbox(new_gt, Px, Py, erp_w=erp_w, erp_h=erp_h)
    BFoV.draw_BFoV(img, Px, Py, border_only=True, color=[0, 0, 255])


if __name__ == '__main__':
    # Get image and RBFoV
    img = cv2.imread('input/test.jpg')
    img_size = img.shape
    erp_w, erp_h = img_size[1], img_size[0]
    gt = np.array([[-1.32, -0.16, 35, 54, 0]])
    print('The original RBFoV: ', gt)

    # Set rotation angle and rotation axis
    rotate_anlge = 20
    n_axis = [1, 1, 1]
    n_axis = n_axis / np.sqrt(n_axis[0] ** 2 + n_axis[1] ** 2 + n_axis[2] ** 2)

    # Rotate image
    img = rotate_image(img, rotate_anlge, n_axis)

    # Get the new RBFoV
    new_gt = get_new_RBFoV(gt, rotate_anlge, n_axis, erp_w=erp_w, erp_h=erp_h)
    print('The new RBFoV: ', new_gt)

    # Visualization results
    visualization(img, new_gt, erp_w=erp_w, erp_h=erp_h)

    cv2.imwrite('output/test.jpg', img)
