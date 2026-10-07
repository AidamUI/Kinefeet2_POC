"""A tiny stand-in for smplx.SMPLX so the fitting logic can be tested without
the licensed model weights. 22 body joints in a forward-kinematics chain with
SMPL-X's parent layout; the first shape coefficient scales every bone."""
from types import SimpleNamespace

import numpy as np
import torch

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
OFFSETS = np.array([
    [0, 0, 0], [0.09, -0.07, 0], [-0.09, -0.07, 0], [0, 0.11, 0], [0, -0.4, 0], [0, -0.4, 0],
    [0, 0.13, 0], [0, -0.4, 0], [0, -0.4, 0], [0, 0.05, 0], [0, -0.05, 0.12], [0, -0.05, 0.12],
    [0, 0.2, 0], [0.08, 0.1, 0], [-0.08, 0.1, 0], [0, 0.15, 0], [0.1, 0, 0], [-0.1, 0, 0],
    [0.26, 0, 0], [-0.26, 0, 0], [0.25, 0, 0], [-0.25, 0, 0]], dtype=np.float32)


def rodrigues(v):
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices."""
    theta = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = v / theta
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    eye = torch.eye(3, dtype=v.dtype).expand_as(K)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


class FakeSMPLX:
    faces = np.array([[0, 1, 2]])

    def __call__(self, betas, global_orient, body_pose, transl, return_verts=True):
        B = betas.shape[0]
        scale = (1.0 + 0.1 * betas[:, :1])[:, :, None]            # (B,1,1)
        offsets = torch.tensor(OFFSETS)[None] * scale              # (B,22,3)
        rots = rodrigues(torch.cat([global_orient[:, None], body_pose.reshape(B, 21, 3)], 1))
        glob_R, glob_p = [None] * 22, [None] * 22
        for j, p in enumerate(PARENTS):
            if p < 0:
                glob_R[j], glob_p[j] = rots[:, 0], torch.zeros(B, 3)
            else:
                glob_R[j] = glob_R[p] @ rots[:, j]
                glob_p[j] = glob_p[p] + (glob_R[p] @ offsets[:, j, :, None])[..., 0]
        joints = torch.stack(glob_p, 1) + transl[:, None]
        out = SimpleNamespace(joints=joints)
        if return_verts:
            # sole points 3 cm below every joint in output-frame Z
            out.vertices = torch.cat([joints, joints - torch.tensor([0, 0, 0.03])], 1)
        return out
