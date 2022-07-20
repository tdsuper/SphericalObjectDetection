import cv2

def qtpixmap_to_cvimg(qtpixmap):

    qimg = qtpixmap.toImage()
    temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    temp_shape += (4,)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    result = result[..., :3]

    return result


def cvimg_to_qtimg(cvimg):

    height, width, depth = cvimg.shape
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    cvimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
    qmap = QPixmap.fromImage(cvimg)
    return qmap

import numpy as np
import math
EPSILON = 1e-8

class Vector:
    def __init__(self, lst):
        self._values = list(lst)

    @classmethod
    def zero(cls, dim):
        return cls([0] * dim)

    def __add__(self, another):
        assert len(self) == len(another), \
            "Error in adding. Length of vectors must be same."

        return Vector([a + b for a, b in zip(self, another)])

    def __sub__(self, another):
        assert len(self) == len(another), \
            "Error in subtracting. Length of vectors must be same."

        return Vector([a - b for a, b in zip(self, another)])

    def norm(self):
        return math.sqrt(sum(e**2 for e in self))

    def normalize(self):
        if self.norm() < EPSILON:
            raise ZeroDivisionError("Normalize error! norm is zero.")
        return Vector(self._values) / self.norm()
        # return 1 / self.norm() * Vector(self._values)
        # return Vector([e / self.norm() for e in self])

    def __mul__(self, k):
        return Vector([k * e for e in self])

    def __rmul__(self, k):
        return self * k

    def __truediv__(self, k):
        return (1 / k) * self

    def __pos__(self):
        return 1 * self

    def __neg__(self):
        return -1 * self

    def __iter__(self):
        return self._values.__iter__()

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return "Vector({})".format(self._values)

    def __str__(self):
        return "({})".format(", ".join(str(e) for e in self._values))

class tools(object):
    def __init__(self, erp_w=1024, erp_h=512):
        self.erp_w = erp_w
        self.erp_h = erp_h

    def pxpy_to_xyz(self, p):
        theta, phi = self.pxpy_to_theta_phi(p[0],p[1])
        xyz = self.theta_phi_to_xyz(theta, phi)
        return xyz

    def xyz_to_pxpy(self, xyz):
        theta, phi = self.xyz_to_theta_phi(xyz)
        px, py = self.theta_phi_to_px_py(theta, phi)
        return [px, py]

    def pxpy_to_theta_phi(self, px, py):
        theta = px / self.erp_w * (2 * np.pi) - np.pi
        phi = -py / self.erp_h * np.pi + np.pi / 2
        return theta, phi

    def theta_phi_to_px_py(self, theta, phi):
        px = (theta + np.pi) / (2 * np.pi) * self.erp_w
        py = -((phi + np.pi / 2) / np.pi * self.erp_h) + self.erp_h
        return px, py

    def theta_phi_to_xyz(self, theta, phi):
        sph_r = 1
        x_3d = sph_r * np.cos(phi) * np.sin(theta)
        y_3d = sph_r * np.sin(phi)
        z_3d = sph_r * np.cos(phi) * np.cos(theta)

        return np.array([x_3d, y_3d, z_3d])

    def xyz_to_theta_phi(self, xyz):
        theta = np.arctan2(xyz[0], xyz[2])
        phi = np.arctan2(xyz[1], np.sqrt(xyz[0] ** 2 + xyz[2] ** 2))
        return theta, phi

    def norm(self, v):
        val = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        v = v / val
        return v

    def get_angle(self, v1, v2):
        Lx = np.sqrt(v1.dot(v1))
        Ly = np.sqrt(v2.dot(v2))
        cos_angle = v1.dot(v2) / (Lx * Ly)
        angle = np.arccos(cos_angle)
        return angle

    def get_v1_v2_angle(self, v1_theta, v1_phi, v2_theta, v2_phi):
        v1_xyz = self.theta_phi_to_xyz(v1_theta, v1_phi)
        v2_xyz = self.theta_phi_to_xyz(v2_theta, v2_phi)

        angle = self.get_angle(v1_xyz, v2_xyz)
        return angle

    def get_n(self, v1, v2):
        n = np.cross(v1, v2)
        v = Vector(n)
        n = v.normalize()
        return n

    def v1xv2(self, v1, v2):
        val = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        return val

    def roll_T(self, n, xyz, gamma=0):
        gamma = gamma / 180 * np.pi
        n11 = (n[0] ** 2) * (1 - np.cos(gamma)) + np.cos(gamma)
        n12 = n[0] * n[1] * (1 - np.cos(gamma)) - n[2] * np.sin(gamma)
        n13 = n[0] * n[2] * (1 - np.cos(gamma)) + n[1] * np.sin(gamma)

        n21 = n[0] * n[1] * (1 - np.cos(gamma)) + n[2] * np.sin(gamma)
        n22 = (n[1] ** 2) * (1 - np.cos(gamma)) + np.cos(gamma)
        n23 = n[1] * n[2] * (1 - np.cos(gamma)) - n[0] * np.sin(gamma)

        n31 = n[0] * n[2] * (1 - np.cos(gamma)) - n[1] * np.sin(gamma)
        n32 = n[1] * n[2] * (1 - np.cos(gamma)) + n[0] * np.sin(gamma)
        n33 = (n[2] ** 2) * (1 - np.cos(gamma)) + np.cos(gamma)

        x, y, z = xyz[0], xyz[1], xyz[2]
        xx = n11 * x + n12 * y + n13 * z
        yy = n21 * x + n22 * y + n23 * z
        zz = n31 * x + n32 * y + n33 * z

        return [xx, yy, zz]

    def cors2NFOV(self, cors):
        A_left, A_right, B_left, B_right = cors[0], cors[1], cors[2], cors[3]
        A_left_theta, A_left_phi = self.pxpy_to_theta_phi(A_left[0], A_left[1])
        A_right_theta, A_right_phi = self.pxpy_to_theta_phi(A_right[0], A_right[1])
        B_left_theta, B_left_phi = self.pxpy_to_theta_phi(B_left[0], B_left[1])
        B_right_theta, B_right_phi = self.pxpy_to_theta_phi(B_right[0], B_right[1])

        A_left_xyz = self.theta_phi_to_xyz(A_left_theta, A_left_phi)
        A_right_xyz = self.theta_phi_to_xyz(A_right_theta, A_right_phi)
        n_up = np.cross(A_left_xyz, A_right_xyz)

        B_left_xyz = self.theta_phi_to_xyz(B_left_theta, B_left_phi)
        B_right_xyz = self.theta_phi_to_xyz(B_right_theta, B_right_phi)
        n_down = np.cross(B_left_xyz, B_right_xyz)

        fov_h = self.get_angle(n_up, n_down)
        #
        # 求左右两面的夹角
        n_left = np.cross(A_left_xyz, B_left_xyz)
        n_right = np.cross(A_right_xyz, B_right_xyz)

        fov_w = self.get_angle(n_left, n_right)
        fov_w = fov_w / np.pi * 180
        fov_h = fov_h / np.pi * 180
        return [fov_w, fov_h]


def roBbox(center, p, ang, erp_w, erp_h):
    t = tools(erp_w, erp_h)
    cx, cy = t.theta_phi_to_px_py(center[0], center[1])
    c_xyz = t.pxpy_to_xyz([cx, cy])
    p_xyz = t.pxpy_to_xyz(p)
    pp_xyz = t.roll_T(c_xyz, p_xyz, ang)
    pp = t.xyz_to_pxpy(pp_xyz)
    return pp

def ro_Shpbbox(theta, phi, Px, Py, ang, erp_w=1920, erp_h=960):
    px = Px.copy()
    py = Py.copy()
    for i in range(len(Px)):
        p = roBbox([theta, phi], [Px[i], Py[i]], ang, erp_w, erp_h)
        px[i], py[i] = p[0], p[1]
    return px, py
