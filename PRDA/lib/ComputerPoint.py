import numpy as np

def uv2xyz(theta, phi):
    xyz = np.concatenate((
        np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
    ), axis=1)
    return xyz

def roll_T(n, xyz, gamma=0):
    gamma = gamma / 180 * np.pi
    n11 = (n[...,0] ** 2) * (1 - np.cos(gamma)) + np.cos(gamma)
    n12 = n[...,0] * n[...,1] * (1 - np.cos(gamma)) - n[...,2] * np.sin(gamma)
    n13 = n[...,0] * n[...,2] * (1 - np.cos(gamma)) + n[...,1] * np.sin(gamma)

    n21 = n[...,0] * n[...,1] * (1 - np.cos(gamma)) + n[...,2] * np.sin(gamma)
    n22 = (n[...,1] ** 2) * (1 - np.cos(gamma)) + np.cos(gamma)
    n23 = n[...,1] * n[...,2] * (1 - np.cos(gamma)) - n[...,0] * np.sin(gamma)

    n31 = n[...,0] * n[...,2] * (1 - np.cos(gamma)) - n[...,1] * np.sin(gamma)
    n32 = n[...,1] * n[...,2] * (1 - np.cos(gamma)) + n[...,0] * np.sin(gamma)
    n33 = (n[...,2] ** 2) * (1 - np.cos(gamma)) + np.cos(gamma)

    x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]
    xx = np.array([np.diagonal(n11 * x + n12 * y + n13 * z)]).T
    yy = np.array([np.diagonal(n21 * x + n22 * y + n23 * z)]).T
    zz = np.array([np.diagonal(n31 * x + n32 * y + n33 * z)]).T

    return np.concatenate((xx,yy, zz), axis=1)



def roArrayVector(theta, phi, v, ang):
    c_xyz = uv2xyz(theta, phi)
    p_xyz = v
    pp_xyz = roll_T(c_xyz, p_xyz, ang)
    return pp_xyz


def transFormat(gt):
    import copy
    ann = copy.copy(gt)
    ann[..., 2] = ann[..., 2] / 180 * np.pi
    ann[..., 3] = ann[..., 3] / 180 * np.pi
    ann[..., 0] += np.pi
    ann[..., 1] = np.pi / 2 - ann[..., 1]
    return ann

class ComputerPoint:
    def theta_phi_to_px_py(self, uv, erp_w=1920, erp_h=960):
        px = uv[:,[0]] / (2 * np.pi) * erp_w
        py = uv[:,[1]] / np.pi * erp_h
        return np.concatenate((px, py), axis=1)

    def xyz_to_theta_phi(self, xyz):
        theta = np.arctan2(xyz[:,[1]], xyz[:,[0]])
        theta[theta<0] += 2 * np.pi
        phi = np.arccos(xyz[:,[2]])
        return np.concatenate((theta, phi), axis=1)

    def pxpy_to_theta_phi(self, px, py, erp_w, erp_h):
        theta = px / erp_w * (2 * np.pi) - np.pi
        phi = -py / erp_h * np.pi + np.pi / 2
        return theta, phi

    def getNormal(self, bbox):
        theta, phi, fov_x_half, fov_y_half, angle = bbox[:, [0]], bbox[:, [1]], bbox[:, [2]] / 2, bbox[:, [3]] / 2, bbox[:,[4]]
        V_lookat = np.concatenate((
            np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
        ), axis=1)
        V_right = np.concatenate((-np.sin(theta), np.cos(theta), np.zeros(theta.shape)), axis=1)
        V_up = np.concatenate((
            -np.cos(phi) * np.cos(theta), -np.cos(phi) * np.sin(theta), np.sin(phi)
        ), axis=1)
        N_left = -np.cos(fov_x_half) * V_right + np.sin(fov_x_half) * V_lookat
        N_right = np.cos(fov_x_half) * V_right + np.sin(fov_x_half) * V_lookat
        N_up = -np.cos(fov_y_half) * V_up + np.sin(fov_y_half) * V_lookat
        N_down = np.cos(fov_y_half) * V_up + np.sin(fov_y_half) * V_lookat

        N_left = roArrayVector(theta, phi, N_left, angle)
        N_right = roArrayVector(theta, phi, N_right, angle)
        N_up = roArrayVector(theta, phi, N_up, angle)
        N_down = roArrayVector(theta, phi, N_down, angle)

        V = np.array([
            np.cross(N_up, N_left), np.cross(N_left, N_down),
            np.cross(N_right, N_up), np.cross(N_down, N_right)
        ])
        norm = np.linalg.norm(V, axis=2)[:, :, np.newaxis].repeat(V.shape[2], axis=2)
        V = np.true_divide(V, norm)

        return V


    def get_angular_point(self, gt, erp_w=1920, erp_h=512):
        V = -self.getNormal(gt)
        p0_uv = self.xyz_to_theta_phi(V[0])
        p1_uv = self.xyz_to_theta_phi(V[1])
        p2_uv = self.xyz_to_theta_phi(V[2])
        p3_uv = self.xyz_to_theta_phi(V[3])
        p0_xy = self.theta_phi_to_px_py(p0_uv, erp_w, erp_h)
        p1_xy = self.theta_phi_to_px_py(p1_uv, erp_w, erp_h)
        p2_xy = self.theta_phi_to_px_py(p2_uv, erp_w, erp_h)
        p3_xy = self.theta_phi_to_px_py(p3_uv, erp_w, erp_h)
        np.concatenate((p0_xy, p1_xy, p2_xy, p3_xy), axis=1)

        return np.array([p2_xy, p0_xy, p3_xy, p1_xy])

    def get_angular_V0(self, gt):
        V = -self.getNormal(gt)
        return V[0]

def get_corner(gt):
    import copy
    bbox = copy.deepcopy(gt)
    _gt = transFormat(bbox)
    p = ComputerPoint()
    points = p.get_angular_point(_gt, erp_w=1024, erp_h=512)
    return [points[0][0][0], points[0][0][1]]


if __name__ == '__main__':
    pass
