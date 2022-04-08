import numpy as np
from calculate_IoU import Sph


def transFormat(gt):
    '''
    Change the format and range of the Spherical Rectangle Representations.
    Input:
    - gt: the last dimension: [center_x, center_y, fov_x, fov_y]
          center_x : [-180, 180]
          center_y : [90, -90]
          fov_x    : [0, 180]
          fov_y    : [0, 180]
          All parameters are angles.
    Output:
    - ann: the last dimension: [center_x', center_y', fov_x', fov_y']
           center_x' : [0, 2 * pi]
           center_y' : [0, pi]
           fov_x'    : [0, pi]
           fov_y'    : [0, pi]
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
    '''Draw Spherical Rectangles'''
    from ImageRecorder import ImageRecorder
    import cv2
    
    img = np.zeros((960, 1920, 3))
    erp_w, erp_h = 1920, 960
    BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=ann[2], view_angle_h=ann[3], long_side=erp_w)
    Px, Py = BFoV._sample_points(ann[0] / 180 * np.pi, ann[1] / 180 * np.pi, border_only=True)
    BFoV.draw_Sphbbox(img, Px, Py, border_only=True)
    
    BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=b[2], view_angle_h=b[3], long_side=erp_w)
    Px, Py = BFoV._sample_points(b[0] / 180 * np.pi, b[1] / 180 * np.pi, border_only=True)
    BFoV.draw_Sphbbox(img, Px, Py, border_only=True, color=(0, 255, 0))
    cv2.imshow('outputa', img)
    cv2.waitKey()


if __name__ == '__main__':
    '''
    Some Unbiased Spherical IoU Computation Examples.
    Note: the input range for pred and gt (angles)
          [center_x, center_y, fov_x, fov_y]
          center_x : [-180, 180]
          center_y : [90, -90]
          fov_x    : [0, 180]
          fov_y    : [0, 180]
    The input range for our unbiased IoU: (radians)
          [center_x, center_y, fov_x, fov_y]
          center_x : [0, 360]
          center_y : [0, 180]
          fov_x    : [0, 180]
          fov_y    : [0, 180]
    We use "transFormat" function to change the format.
    '''
    pred = np.array([
        [-33.76, 43.99, 61.65, 43.88],
        [-51.96, 19.61, 61.65, 43.88],
        [-88.25, 7.33, 61.65, 43.88],
        [-109.89, -13.51, 61.65, 43.88],
        [-60.44, 14.7, 61.65, 43.88],
        [0, 0, 61.65, 43.88]
    ])

    gt = np.array([
        [-37.9, 19.33, 64.09, 48.89],
        [-75.68, 12.1, 64.09, 48.89],
        [-97.17, -8.95, 64.09, 48.89],
        [-51.24, -29.18, 40.65, 42.58],
        [0, -1, 58.42, 40.32],
    ])

    _gt = transFormat(gt)
    _pred = transFormat(pred)
    
    sphIoU = Sph().sphIoU(_pred, _gt)
    
    # Present the values for the IoU  calculation results in float format.
    np.set_printoptions(suppress=True)
    print(sphIoU)
    
    # Draw spherical bounding boxes. (Use the fifth of pred and the first of gt as an example)
    drawSphBBox(pred[-2], gt[0])
