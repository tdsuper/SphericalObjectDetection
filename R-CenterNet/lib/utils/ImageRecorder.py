#!/usr/bin/env python
# encoding: utf-8

import os
import cv2
import numpy as np

from scipy.interpolate import RegularGridInterpolator as interp2d
import pdb

class ImageRecorder(object):

    """
    Record normal view video given 360 degree video.
    """

    #def __init__(self, sphereW, sphereH, view_angle=65.5, imgW=640):
    def __init__(self, sphereW, sphereH, view_angle_w=64, view_angle_h=64, long_side=640):
        self.sphereW = sphereW
        self.sphereH = sphereH
        # pdb.set_trace()
        #w, h = 4, 3
        w, h = view_angle_w, view_angle_h  # fov_x, fov_y
        self._long_side = long_side  # 1920

        if w >= h:
            self._imgW = long_side
            self._imgH = int(h / 360 * np.pi * self._imgW / float(w / 360 * np.pi))
        else:
            self._imgH = long_side
            self._imgW = int(w / 360 * np.pi * self._imgH / float(h / 360 * np.pi))

        #self._imgH = int(h * self._imgW / float(w))
        #self._Y = np.arange(self._imgH) + (self._imgW - self._imgH)/2

        TX, TY = self._meshgrid()
        R, ANGy = self._compute_radius(view_angle_w, TY)

        self._R = R
        self._ANGy = ANGy
        self._Z = TX

    def _meshgrid(self):
        """Construct mesh point
        :returns: TX, TY

        """
        if self._imgW >= self._imgH:
            offset = int((self._imgW - self._imgH)/2)
            TX, TY = np.meshgrid(range(self._imgW), range(offset, self._imgH + offset))
        else:
            offset = int((self._imgH - self._imgW)/2)
            TX, TY = np.meshgrid(range(offset, self._imgW + offset), range(self._imgH))

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
        angle_x, angle_y = self._direct_camera(x, y, border_only)
        Px = (angle_x + np.pi) / (2*np.pi) * self.sphereW + 0.5
        Py = (np.pi/2 - angle_y) / np.pi * self.sphereH + 0.5
        INDx = Px < 1
        # pdb.set_trace()
        Px[INDx] += self.sphereW
        return Px, Py  # array

    def _direct_camera(self, rotate_x, rotate_y, border_only=False):
        if border_only:
            angle_y = np.hstack([self._ANGy[0,:], self._ANGy[-1,:], self._ANGy[:,0], self._ANGy[:,-1]]) + rotate_y
            Z = np.hstack([self._Z[0,:], self._Z[-1,:], self._Z[:,0], self._Z[:,-1]])
            R = np.hstack([self._R[0,:], self._R[-1,:], self._R[:,0], self._R[:,-1]])
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
        angle_y[INDy] = -np.pi -angle_y[INDy]
        angle_x[INDy] = angle_x[INDy] + np.pi

        INDx = angle_x <= -np.pi
        angle_x[INDx] += 2*np.pi
        INDx = angle_x > np.pi
        angle_x[INDx] -= 2*np.pi
        return angle_x, angle_y  # array

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
        for c in xrange(3):
            full_image = interp2d((y_grid, x_grid), im[:,:,c],
                                   bounds_error=False,
                                   method='linear',
                                   fill_value=None)
            warped_image = full_image(samples).reshape(Px.shape)
            warped_images.append(warped_image)
        warped_image = np.stack(warped_images, axis=2)
        return warped_image

class ZoomCameraImage(object):   # mine
    def __init__(self, image_in, image_out, distance="5s", imgW=640):
        print('ZoomCameraImage imread image: ' + image_in)
        image = cv2.imread(image_in)
        sphereH, sphereW, channel = image.shape
        self.__sphereW = sphereW
        self.__sphereH = sphereH
        self.__image = image
        self.__imgW = imgW

        unit = distance[-1:]
        try:
            value = int(distance[:-1])
        except:
            raise ValueError("Invalid distance format")
        else:
            if unit == "s":
                self.__distance = int(np.floor(self.__fps * value))
            elif unit == "f":
                self.__distance = value
            else:
                raise ValueError("Invalid distance unit {}".format(distance))

        recorder = ImageRecorder(self.__sphereW,
                                 self.__sphereH,
                                 view_angle=65.5,
                                 imgW=self.__imgW)
        self.__recorder = {65.5: recorder}
        self.__recording = False

        self.__x = 0.
        self.__y = 0.
        self.__z = 65.5
      
        output_path = '{}.jpg'.format(image_out)
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (recorder._imgW, recorder._imgH)
        #self.__output = cv2.VideoWriter(output_path, fourcc, self.__fps, size)
        #self.__output = cv2.imwrite(output_path, image)

    def start_record(self, x, y, z=65.5):
        self.__recording = True
        self.__x = x
        self.__y = y
        self.__z = z

    def end_record(self):
        self.__recording = False

    def forward(self, diff_x=0, diff_y=0, diff_z=0):
        finished = False
        dx = float(diff_x) / self.__distance
        dy = float(diff_y) / self.__distance
        dz = float(diff_z) / self.__distance
        for step in xrange(self.__distance):
            self.__displace(dx, dy, dz)

            warped_image = self.__capture_next()
            if warped_image is None:
                finished = True
                break
            #if self.__recording:
            #    self.__output.write(warped_image)
        return finished

    def __displace(self, dx, dy, dz):
        self.__x += dx
        if self.__x >= np.pi:
            self.__x -= 2*np.pi
        elif self.__x < -np.pi:
            self.__x += 2*np.pi
        self.__y += dy
        if self.__y > np.pi or self.__y < -np.pi:
            raise ValueError("Invalid latitude")
        self.__z += dz
        if self.__z <= 0 or self.__z > 180:
            raise ValueError("Invalid angle of view")

    def __capture_next(self):
        ret = True 
        frame = self.__image
        if not ret:
            warped_image = None
        else:
            if self.__recorder.has_key(self.__z):
                recorder = self.__recorder[self.__z]
            else:
                recorder = ImageRecorder(self.__sphereW, self.__sphereH,
                                         view_angle=self.__z,
                                         imgW=self.__imgW)
                self.__recorder[self.__z] = recorder
            warped_image = recorder.catch(self.__x, self.__y, frame)
            warped_image[warped_image>255.] = 255.
            warped_image[warped_image<0.] = 0.
            warped_image = warped_image.astype(np.uint8)
        return warped_image

