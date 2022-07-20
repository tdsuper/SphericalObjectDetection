import numpy as np
from calculate_RoIoU import Sph
from libs.tools import ro_Shpbbox

def transFormat(gt):
    '''
    Change the format and range of the RBFoV Representations.
    Input:
    - gt: the last dimension: [center_x, center_y, fov_x, fov_y, angle]
          center_x : [-180, 180]
          center_y : [90, -90]
          fov_x    : [0, 180]
          fov_y    : [0, 180]
          angle    : [90, -90]
          All parameters are angles.
    Output:
    - ann: the last dimension: [center_x', center_y', fov_x', fov_y', angle]
           center_x' : [0, 2 * pi]
           center_y' : [0, pi]
           fov_x'    : [0, pi]
           fov_y'    : [0, pi]
           angle     : [90, -90]
           All parameters are radians.
    '''
    import copy
    ann = copy.copy(gt)
    ann[..., 2] = ann[..., 2] / 180 * np.pi
    ann[..., 3] = ann[..., 3] / 180 * np.pi
    ann[..., 0] = ann[..., 0] / 180 *np.pi+ np.pi
    ann[..., 1] = np.pi / 2 - ann[..., 1] / 180 * np.pi
    return ann

def drawSphBBox(ann, b):
    '''Draw RBFoVs'''
    from libs.ImageRecorder import ImageRecorder
    import cv2
    
    img = np.zeros((960, 1920, 3))
    erp_w, erp_h = 1920, 960
    BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=ann[2], view_angle_h=ann[3], long_side=erp_w)
    Px, Py = BFoV._sample_points(ann[0] / 180 * np.pi, ann[1] / 180 * np.pi, border_only=True)
    Px, Py = ro_Shpbbox(ann, Px, Py, erp_w=erp_w, erp_h=erp_h)
    BFoV.draw_Sphbbox(img, Px, Py, border_only=True)
    
    BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=b[2], view_angle_h=b[3], long_side=erp_w)
    Px, Py = BFoV._sample_points(b[0] / 180 * np.pi, b[1] / 180 * np.pi, border_only=True)
    Px, Py = ro_Shpbbox(b, Px, Py, erp_w=erp_w, erp_h=erp_h)
    BFoV.draw_Sphbbox(img, Px, Py, border_only=True, color=(0, 255, 0))

    cv2.imshow('output', img)
    cv2.waitKey()


if __name__ == '__main__':
    '''
    Some Examples for IoU Calculation between two RBFoVs.
    Note: the input range for pred and gt (angles)
          [center_x, center_y, fov_x, fov_y]
          center_x : [-180, 180]
          center_y : [90, -90]
          fov_x    : [0, 180]
          fov_y    : [0, 180]
          angle    : [90, -90]
    The input range for our unbiased IoU: (radians)
          [center_x, center_y, fov_x, fov_y]
          center_x : [0, 360]
          center_y : [0, 180]
          fov_x    : [0, 180]
          fov_y    : [0, 180]
          angle    : [90, -90]
    We use "transFormat" function to change the format.
    '''
    pred = np.array([
        [-33.76, 43.99, 61.65, 43.88, -11],
        [-51.96, 19.61, 61.65, 43.88, 33],
        [-88.25, 7.33, 61.65, 43.88, 15],
        [-109.89, -13.51, 61.65, 43.88, -22],
        [-60.44, 14.7, 61.65, 43.88, -17],
        [0, 0, 61.65, 43.88, 17]
    ])

    gt = np.array([
        [-37.9, 19.33, 64.09, 48.89, -30],
        [-75.68, 12.1, 64.09, 48.89, 18],
        [-97.17, -8.95, 64.09, 48.89, 29],
        [-51.24, -29.18, 40.65, 42.58, -12],
        [0, -1, 58.42, 40.32, 12],
    ])

    _gt = transFormat(gt)
    _pred = transFormat(pred)
    
    sphIoU = Sph().sphIoU(_pred, _gt)
    
    # Present the values for the IoU  calculation results in float format.
    np.set_printoptions(suppress=True)
    print(sphIoU)
    
    # Draw RBFoVs. (Use the fifth of pred and the first of gt as an example)
    drawSphBBox(pred[-2], gt[0])
