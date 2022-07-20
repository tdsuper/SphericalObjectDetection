import math
import functools
import numpy as np
from scipy.ndimage import map_coordinates
EPSILON = 1e-8

class Vector:
    def __init__(self, lst):
        self._values = list(lst)

    @classmethod
    def zero(cls, dim):
        """返回一个dim维的零向量"""
        return cls([0] * dim)

    def __add__(self, another):
        """向量加法，返回结果向量"""
        assert len(self) == len(another), \
            "Error in adding. Length of vectors must be same."

        return Vector([a + b for a, b in zip(self, another)])

    def __sub__(self, another):
        """向量减法，返回结果向量"""
        assert len(self) == len(another), \
            "Error in subtracting. Length of vectors must be same."

        return Vector([a - b for a, b in zip(self, another)])

    def norm(self):
        """返回向量的模"""
        return math.sqrt(sum(e**2 for e in self))

    def normalize(self):
        """返回向量的单位向量"""
        if self.norm() < EPSILON:
            raise ZeroDivisionError("Normalize error! norm is zero.")
        return Vector(self._values) / self.norm()
        # return 1 / self.norm() * Vector(self._values)
        # return Vector([e / self.norm() for e in self])

    def __mul__(self, k):
        """返回数量乘法的结果向量：self * k"""
        return Vector([k * e for e in self])

    def __rmul__(self, k):
        """返回数量乘法的结果向量：k * self"""
        return self * k

    def __truediv__(self, k):
        """返回数量除法的结果向量：self / k"""
        return (1 / k) * self

    def __pos__(self):
        """返回向量取正的结果向量"""
        return 1 * self

    def __neg__(self):
        """返回向量取负的结果向量"""
        return -1 * self

    def __iter__(self):
        """返回向量的迭代器"""
        return self._values.__iter__()

    def __getitem__(self, index):
        """取向量的第index个元素"""
        return self._values[index]

    def __len__(self):
        """返回向量长度（有多少个元素）"""
        return len(self._values)

    def __repr__(self):
        return "Vector({})".format(self._values)

    def __str__(self):
        return "({})".format(", ".join(str(e) for e in self._values))

class tools(object):
    def __init__(self, erp_w=1920, erp_h=960):
        self.erp_w = erp_w
        self.erp_h = erp_h

    def pxpy2xyz(self, p):
        theta, phi = self.pxpy2uv(p[0],p[1])
        xyz = self.uv2xyz(theta, phi)
        return xyz

    def xyz2pxpy(self, xyz):
        theta, phi = self.xyz2uv(xyz)
        px, py = self.uv2pxpy(theta, phi)
        return [px, py]

    def pxpy2uv(self, px, py):
        theta = px / self.erp_w * (2 * np.pi) - np.pi
        phi = -py / self.erp_h * np.pi + np.pi / 2
        return theta, phi

    def uv2pxpy(self, theta, phi):
        px = (theta + np.pi) / (2 * np.pi) * self.erp_w
        py = -((phi + np.pi / 2) / np.pi * self.erp_h) + self.erp_h
        return px, py

    def uv2xyz(self, theta, phi):
        sph_r = 1
        x_3d = sph_r * np.cos(phi) * np.sin(theta)
        y_3d = sph_r * np.sin(phi)
        z_3d = sph_r * np.cos(phi) * np.cos(theta)

        return np.array([x_3d, y_3d, z_3d])

    def xyz2uv(self, xyz):
        theta = np.arctan2(xyz[0], xyz[2])
        phi = np.arctan2(xyz[1], np.sqrt(xyz[0] ** 2 + xyz[2] ** 2))
        return theta, phi


    def roll_T(self, n, xyz, gamma=0):
        '''
            rotation matrix
        '''
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

def roBbox(center, p, ang, erp_w, erp_h):
    t = tools(erp_w, erp_h)
    cx, cy = t.uv2pxpy(center[0], center[1])
    c_xyz = t.pxpy2xyz([cx, cy])
    p_xyz = t.pxpy2xyz(p)
    pp_xyz = t.roll_T(c_xyz, p_xyz, ang)
    pp = t.xyz2pxpy(pp_xyz)
    return pp

def ro_Shpbbox(new_gt, Px, Py, erp_w, erp_h):
    theta, phi, angle = new_gt[...,0], new_gt[...,1], new_gt[...,4]
    for i in range(len(Px)):
        p = roBbox([theta, phi], [Px[i], Py[i]], angle, erp_w, erp_h)
        Px[i], Py[i] = p[0], p[1]
    return Px, Py


def uv_meshgrid(erp_w, erp_h):
    uv = np.stack(np.meshgrid(range(erp_w), range(erp_h)), axis=-1)
    uv = uv.astype(np.float64)
    uv[..., 0] = ((uv[..., 0] + 0.5) / erp_w - 0.5) * 2 * np.pi
    uv[..., 1] = ((uv[..., 1] + 0.5) / erp_h - 0.5) * np.pi

    return uv


@functools.lru_cache()
def _uv_tri(erp_w, erp_h):
    uv = uv_meshgrid(erp_w, erp_h)
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    sin_v = np.sin(uv[..., 1])
    cos_v = np.cos(uv[..., 1])
    return sin_u, cos_u, sin_v, cos_v


def uv_tri(erp_w, erp_h):
    sin_u, cos_u, sin_v, cos_v = _uv_tri(erp_w, erp_h)
    return sin_u.copy(), cos_u.copy(), sin_v.copy(), cos_v.copy()

def uv2xyz(erp_w, erp_h):
    sin_u, cos_u, sin_v, cos_v = uv_tri(erp_w, erp_h)
    x = cos_v * sin_u
    y = sin_v
    z = cos_v * cos_u
    return x, y, z


def xyz2uv(x, y, z):
    u = np.arctan2(x, z)
    v = np.arctan2(y, np.sqrt(x ** 2 + z ** 2))

    return u, v

def roll(gamma, n, erp_w, erp_h,):
    t = tools(erp_w, erp_h)
    x, y, z = uv2xyz(erp_w, erp_h)
    xyz = t.roll_T(n, [x, y, z], gamma)
    u, v = xyz2uv(xyz[0], xyz[1], xyz[2])
    return u, v

def rotate_image(img, gamma, n):
    u0, v0 = roll(gamma, n, img.shape[1], img.shape[0],)

    refx = (u0 / (2 * np.pi) + 0.5) * img.shape[1] - 0.5
    refy = (v0 / np.pi + 0.5) * img.shape[0] - 0.5

    # [TODO]: using opencv remap could probably speedup the process a little
    stretched_img = np.stack([
        map_coordinates(img[..., i], [refy, refx], order=1, mode='wrap')
        for i in range(img.shape[-1])
    ], axis=-1)

    return stretched_img


def roPoint(point, gamma, n, erp_w, erp_h):
    t = tools(erp_w, erp_h)
    point[0] = ((point[0] + 0.5) / erp_w - 0.5) * 2 * np.pi
    point[1] = ((point[1] + 0.5) / erp_h - 0.5) * np.pi

    sin_u = np.sin(point[0])
    cos_u = np.cos(point[0])
    sin_v = np.sin(point[1])
    cos_v = np.cos(point[1])

    x = cos_v * sin_u
    y = sin_v
    z = cos_v * cos_u

    xyz = t.roll_T(n, [x, y, z], gamma)
    u0, v0 = xyz2uv(xyz[0], xyz[1], xyz[2])

    px = (u0 / (2 * np.pi) + 0.5) * erp_w - 0.5
    py = (v0 / np.pi + 0.5) * erp_h - 0.5

    return [px, py]

