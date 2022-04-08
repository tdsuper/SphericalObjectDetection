#!/usr/bin/env python
# encoding: utf-8

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator as interp2d


class ImageRecorder(object):

    """
    Draw spherical rectangles on 360 degree images.
    """

    # def __init__(self, sphereW, sphereH, view_angle=65.5, imgW=640):
    def __init__(self, sphereW, sphereH, view_angle_w=64, view_angle_h=64, long_side=640):
        """
        sphereW, sphereH：the width and height of spherical image in ERP format
        view_angle_w, view_angle_h：FOVs
        long_side：the longest side of ERP
        """
        self.sphereW = sphereW
        self.sphereH = sphereH
        fov_w, fov_h = view_angle_w, view_angle_h
        self._long_side = long_side

        if fov_w >= fov_h:
            self._imgW = long_side
            self._imgH = int(np.tan(fov_h / 360 * np.pi) *
                             self._imgW / float(np.tan(fov_w / 360 * np.pi)))
        else:
            self._imgH = long_side
            self._imgW = int(np.tan(fov_w / 360 * np.pi) *
                             self._imgH / float(np.tan(fov_h / 360 * np.pi)))

        TX, TY = self._meshgrid()
        R, ANGy = self._compute_radius(view_angle_w, TY)

        self._R = R
        self._ANGy = ANGy
        self._Z = TX

    def _meshgrid(self):
        """
        Construct mesh point
        :returns: TX, TY
        """
        if self._imgW >= self._imgH:
            offset = int((self._imgW - self._imgH)/2)
            TX, TY = np.meshgrid(range(self._imgW), range(
                offset, self._imgH + offset))
        else:
            offset = int((self._imgH - self._imgW)/2)
            TX, TY = np.meshgrid(
                range(offset, self._imgW + offset), range(self._imgH))

        TX = TX.astype(np.float64) - 0.5
        TX -= self._long_side/2

        TY = TY.astype(np.float64) - 0.5
        TY -= self._long_side/2
        return TX, TY

    def _compute_radius(self, view_angle, TY):
        _view_angle = np.pi * view_angle / 180.
        r = self._imgW/2 / np.tan(_view_angle/2)
        R = np.sqrt(np.power(TY, 2) + r**2)
        ANGy = np.arctan(-TY/r)
        return R, ANGy

    def catch(self, x, y, image):
        Px, Py = self._sample_points(x, y)
        warped_image = self._warp_image(Px, Py, image)
        return warped_image

    def _sample_points(self, x, y, border_only=False):
        """
        Sample necessary points.
        x, y: the coordinate of the center point
        """
        angle_x, angle_y = self._direct_camera(x, y, border_only)
        Px = (angle_x + np.pi) / (2*np.pi) * self.sphereW + 0.5
        Py = (np.pi/2 - angle_y) / np.pi * self.sphereH + 0.5
        INDx = Px < 1
        Px[INDx] += self.sphereW
        return Px, Py

    def _direct_camera(self, rotate_x, rotate_y, border_only=False):
        """
        rotate_x, rotate_y: the coordinate of the center point (FOV)
        """
        if border_only:
            angle_y = np.hstack([self._ANGy[0, :], self._ANGy[-1, :],
                                 self._ANGy[:, 0], self._ANGy[:, -1]]) + rotate_y
            Z = np.hstack([self._Z[0, :], self._Z[-1, :],
                           self._Z[:, 0], self._Z[:, -1]])
            R = np.hstack([self._R[0, :], self._R[-1, :],
                           self._R[:, 0], self._R[:, -1]])
        else:
            angle_y = self._ANGy + rotate_y
            Z = self._Z
            R = self._R

        X = np.sin(angle_y) * R
        Y = - np.cos(angle_y) * R

        INDn = np.abs(angle_y) > np.pi/2

        angle_x = np.arctan(Z / -Y)
        RZY = np.linalg.norm(np.stack((Y, Z), axis=0), axis=0)
        angle_y = np.arctan(X / RZY)

        angle_x[INDn] += np.pi
        angle_x += rotate_x

        INDy = angle_y < -np.pi/2
        angle_y[INDy] = -np.pi - angle_y[INDy]
        angle_x[INDy] = angle_x[INDy] + np.pi

        INDx = angle_x <= -np.pi
        angle_x[INDx] += 2*np.pi
        INDx = angle_x > np.pi
        angle_x[INDx] -= 2*np.pi
        return angle_x, angle_y

    def _warp_image(self, Px, Py, frame):
        minX = max(0, int(np.floor(Px.min())))
        minY = max(0, int(np.floor(Py.min())))

        maxX = min(int(self.sphereW), int(np.ceil(Px.max())))
        maxY = min(int(self.sphereH), int(np.ceil(Py.max())))

        im = frame[minY:maxY, minX:maxX, :]
        Px -= minX
        Py -= minY
        warped_images = []

        y_grid = np.arange(im.shape[0])
        x_grid = np.arange(im.shape[1])
        samples = np.vstack([Py.ravel(), Px.ravel()]).transpose()
        for c in range(3):
            full_image = interp2d((y_grid, x_grid), im[:, :, c],
                                  bounds_error=False,
                                  method='linear',
                                  fill_value=None)
            warped_image = full_image(samples).reshape(Px.shape)
            warped_images.append(warped_image)
        warped_image = np.stack(warped_images, axis=2)
        return warped_image

    def draw_bbox(self, frame, Px, Py):
        '''
        Draw a planar bounding box.
        '''
        Px_min, Px_max = Px.min(), Px.max()
        Py_min, Py_max = Py.min(), Py.max()
        if (Px_max-Px_min) < self._long_side-1:
            cv2.rectangle(frame, (round(Px_min), round(Py_min)),
                          (round(Px_max), round(Py_max)), (255, 0, 0), 4)
        else:

            Px_right = Px[Px >= (self._long_side/2-1)]
            Px_left = Px[Px < (self._long_side / 2 - 1)]
            if round(Px_right.min()) == round(Px_left.max()):
                cv2.rectangle(frame, (int(Px_min), int(Py_min)),
                              (round(Px_max), round(Py_max)), (255, 0, 0), 4)
            else:
                Px_right_min, Px_right_max = Px_right.min(), Px_right.max()
                cv2.rectangle(frame, (round(Px_right_min), round(Py_min)), (round(
                    Px_right_max), round(Py_max)), (255, 0, 0), 4)
                Px_left_min, Px_left_max = Px_left.min(), Px_left.max()
                cv2.rectangle(frame, (round(Px_left_min), round(Py_min)), (round(
                    Px_left_max), round(Py_max)), (255, 0, 0), 4)
        return frame

    def draw_Sphbbox(self, frame, Px, Py, border_only=False, color=(0, 0, 255)):
        '''
        Draw a spherical bounding box on the spherical image in ERP format.
        '''
        if border_only:
            for j in range(Px.shape[0]):
                cv2.circle(frame, (int(Px[j]), int(Py[j])), 1, color, 4)
        else:
            for i in range(Px.shape[0]):
                for j in range(Px.shape[1]):
                    cv2.circle(
                        frame, (int(Px[i][j]), int(Py[i][j])), 1, color, 1)
        return frame

    def boxinfo(self, Px, Py):
        '''
        Record bounding boxes information.
        '''
        Px_min, Px_max = Px.min(), Px.max()
        Py_min, Py_max = Py.min(), Py.max()
        if (Px_max - Px_min) < self._long_side - 1:
            box = [round(Px_min), round(Py_min), round(Px_max), round(Py_max)]
        else:
            Px_right = Px[Px >= (self._long_side / 2 - 1)]
            Px_left = Px[Px < (self._long_side / 2 - 1)]
            if round(Px_right.min()) == round(Px_left.max()):
                box = [round(Px_min), round(Py_min),
                       round(Px_max), round(Py_max)]
            else:
                Px_right_min, Px_right_max = Px_right.min(), Px_right.max()

                Px_left_min, Px_left_max = Px_left.min(), Px_left.max()
                w_right = round(Px_right_max) - round(Px_right_min)
                w_left = round(Px_left_max) - round(Px_left_min)
                if w_right >= w_left:
                    box = [round(Px_right_min), round(Py_min),
                           round(Px_right_max), round(Py_max)]
                else:
                    box = [round(Px_left_min), round(Py_min),
                           round(Px_left_max), round(Py_max)]
        return box
