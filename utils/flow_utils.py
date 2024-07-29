import math
from functools import lru_cache

import cv2
import numpy as np
import torch
import math
import torch.nn.functional as F

from scene import GaussianModel
from scene.cameras import Camera


def unproject(depth_map: torch.Tensor, extrinsic_matrix: torch.Tensor, intrinsic_matrix: torch.Tensor):
    """
    Args:
        depth_map: (1, H, W)
        extrinsic_matrix: (4, 4) world-to-camera
        intrinsic_matrix: (3, 3) camera-to-pixel
    Returns:
        (3, H, W)
    """
    # assert len(depth_map.shape) == 3 and depth_map.shape[0] == 1
    H, W = depth_map.shape[1:]
    depth_map = depth_map.transpose(1, 2)

    if not isinstance(extrinsic_matrix, torch.Tensor):
        extrinsic_matrix = torch.tensor(extrinsic_matrix, dtype=depth_map.dtype, device=depth_map.device)
    if not isinstance(intrinsic_matrix, torch.Tensor):
        intrinsic_matrix = torch.tensor(intrinsic_matrix, dtype=depth_map.dtype, device=depth_map.device)

    u, v = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing="ij")
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=0).float().to(depth_map.device)
    uv1 = uv1.view(3, -1)
    xyz_cam = torch.inverse(intrinsic_matrix) @ uv1 * depth_map.reshape(1, -1)
    xyz1_cam = torch.cat([xyz_cam, torch.ones_like(xyz_cam[:1])], dim=0)

    xyz1_world = torch.inverse(extrinsic_matrix) @ xyz1_cam
    xyz1_world = xyz1_world.view(4, W, H).transpose(1, 2)
    xyz_world = xyz1_world[:3, :, :]

    return xyz_world


def project(xyz_world: torch.Tensor, extrinsic_matrix: torch.Tensor, intrinsic_matrix: torch.Tensor):
    """
    Args:
        xyz_world: (3, N)
        extrinsic_matrix: (4, 4) world-to-camera
        intrinsic_matrix: (3, 3) camera-to-pixel
    Returns:
        [(2, N), (1, N)]
    """
    if not isinstance(extrinsic_matrix, torch.Tensor):
        extrinsic_matrix = torch.tensor(extrinsic_matrix, dtype=xyz_world.dtype, device=xyz_world.device)
    if not isinstance(intrinsic_matrix, torch.Tensor):
        intrinsic_matrix = torch.tensor(intrinsic_matrix, dtype=xyz_world.dtype, device=xyz_world.device)
    xyz1_world = torch.cat([xyz_world, torch.ones_like(xyz_world[:1])], dim=0)
    xyz1_cam = extrinsic_matrix @ xyz1_world
    uvz = intrinsic_matrix @ xyz1_cam[:3, :]
    z = uvz[2:]
    uv = uvz[:2] / z
    return uv, z


@lru_cache(maxsize=1)
def get_faiss_index():
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="faiss.contrib.torch_utils")

    import faiss
    from faiss.contrib import torch_utils

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(3)
    # index_flat = faiss.IndexIVFFlat(index_flat, 3, 100, faiss.METRIC_L2)
    # index_flat.train(xyz2)
    gpu_index_flat: faiss.IndexFlatL2 = faiss.index_cpu_to_gpu(res, 0, index_flat)

    return gpu_index_flat


def knn(xyz1: torch.Tensor, xyz2: torch.Tensor, K: int, backend="mmcv") -> torch.Tensor:
    """
    Args:
        xyz1: (N, 3)
        xyz2: (M, 3)
        K: int <= M
    Returns:
        (N, K)
    """
    # assert K > 0
    # assert xyz2.shape[0] > K, "K is too large"
    N = xyz1.shape[0]

    if backend in ("o3d", "open3d"):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz2.cpu().numpy(), np.float64))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        top_k_nearest_idx = np.zeros([N, K], dtype=np.int64)
        for i, p in enumerate(xyz1.cpu().numpy()):
            k, idx, d = pcd_tree.search_knn_vector_3d(p, K)
            top_k_nearest_idx[i, :] = idx
        top_k_nearest_idx = torch.Tensor(top_k_nearest_idx).to(xyz1.device).long()
    elif backend == "faiss":
        faiss_index = get_faiss_index()
        faiss_index.add(xyz2)
        d, top_k_nearest_idx = faiss_index.search(xyz1.contiguous(), K)
        faiss_index.reset()
    elif backend == "mmcv":
        from mmcv.ops.knn import knn

        top_k_nearest_idx = knn(K, xyz2.unsqueeze(0).contiguous(), xyz1.unsqueeze(0).contiguous()).squeeze(0).T
        # top_k_nearest_idx = top_k_nearest_idx.long()
    else:
        CHUNK = 2048  # prevent OOM
        top_k_nearest_idx = torch.zeros(N, K, dtype=torch.int64, device=xyz1.device)  # N,K
        for i in range(0, N, CHUNK):
            dist_matrix = torch.cdist(xyz1[i : i + CHUNK, :], xyz2)  # N(CHUNK),M
            _, top_k_nearest_idx[i : i + CHUNK, :] = torch.topk(dist_matrix, k=K, dim=1, largest=False)

    return top_k_nearest_idx.long()


# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.compile_fx")
# @torch.compile()
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices. (Copy from PyTorch3D)

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# @torch.compile()
def calculate_gaussian_3d(
    pts: torch.Tensor, means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, opacities: torch.Tensor
):
    """
    G(x) = opacity * exp(-0.5 * (x-means).T @ Sigma^(-1) @ (x-means))

    Args:
        pts: (..., 3)
        means: (..., 3)
        scales: (..., 3)
        rotations: (..., 4)
        opacities: (..., 1)
    Returns:
        (..., 1)
    """
    # assert pts.shape[:-1] == means.shape[:-1] == scales.shape[:-1] == rotations.shape[:-1] == opacities.shape[:-1]
    batch_shape = pts.shape[:-1]
    pts = pts.reshape(-1, 3)
    means = means.reshape(-1, 3)
    scales = scales.reshape(-1, 3)
    rotations = rotations.reshape(-1, 4)
    opacities = opacities.reshape(-1, 1)

    S = torch.diag_embed(scales)
    R = quaternion_to_matrix(rotations)
    M = S @ R
    Sigma = M.transpose(1, 2) @ M

    diff = (pts - means).unsqueeze(-1)
    try:
        # power = -0.5 * (diff.transpose(1, 2) @ torch.inverse(Sigma) @ diff)
        power = -0.5 * (diff.transpose(1, 2) @ torch.linalg.solve(Sigma, diff))
    except torch._C._LinAlgError:
        power = -0.5 * (diff.transpose(1, 2) @ torch.linalg.lstsq(Sigma, diff).solution)
    gaussian_values = torch.exp(torch.log(opacities) + power.squeeze(-1))
    return gaussian_values.reshape(*batch_shape, 1)


@torch.no_grad()
def pixel_to_gaussian_idx(
    depth_map: torch.Tensor,
    w2c: torch.Tensor,
    k: torch.Tensor,
    seg: torch.Tensor,
    gaussians: GaussianModel,
    is_fg: torch.Tensor,
    K=1,
    return_K=False,
):
    """
    Args:
        depth_map: (1, H, W)
        w2c: (4, 4) world-to-camera
        k: (3, 3) camera-to-pixel
        seg: (H, W), N = seg.sum() <= H*W
    Returns:
        (N,)
        if return_K: [(N, K), (N, K, 1)]
    """
    pts_unproject = unproject(depth_map, w2c, k)[:, seg].reshape(3, -1).T
    gaussian_fg_means = gaussians.get_xyz[is_fg]
    knn_idx = knn(pts_unproject, gaussian_fg_means, K=K)
    if K == 1:
        return knn_idx.squeeze(-1)
    nearby_means = gaussian_fg_means[knn_idx]
    nearby_rotations = gaussians.get_rotation[is_fg][knn_idx]
    nearby_scales = gaussians.get_scaling[is_fg][knn_idx]
    nearby_opacities = gaussians.get_opacity[is_fg][knn_idx]
    contributions = calculate_gaussian_3d(
        pts_unproject.unsqueeze(-2).repeat_interleave(knn_idx.shape[-1], dim=-2),
        nearby_means,
        nearby_scales,
        nearby_rotations,
        nearby_opacities,
    )
    if return_K:
        return knn_idx, contributions
    else:
        knn_idx_idx = torch.argmax(contributions, dim=1)
        target_gaussian_idx = torch.gather(knn_idx, dim=1, index=knn_idx_idx).squeeze(-1)
        return target_gaussian_idx


def readFlow_flo(fn: str):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError("Magic number incorrect. Invalid .flo file")
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def readFlow(fn: str):
    if fn.endswith(".flo"):
        return readFlow_flo(fn)
    else:
        try:
            return np.load(fn)
        except:
            print(f"Error reading {fn}")
            raise


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col : col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col : col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col : col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col : col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += +BM

    # MR
    colorwheel[col : col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col : col + MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    UNKNOWN_FLOW_THRESH = 1e7
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.0
    maxv = -999.0
    minu = 999.0
    minv = 999.0

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def resize_flow(flow: np.ndarray, des_w: int, des_h: int, method="bilinear") -> np.ndarray:
    """
    Args:
        flow: (H, W, 2), 2 represents uv
    """
    src_height = flow.shape[0]
    src_width = flow.shape[1]
    if src_width == des_w and src_height == des_h:
        return flow
    ratio_height = float(des_h) / float(src_height)
    ratio_width = float(des_w) / float(src_width)
    if method in ("bilinear", "linear"):
        flow = cv2.resize(flow, (des_w, des_h), interpolation=cv2.INTER_LINEAR)
    elif method == "nearest":
        flow = cv2.resize(flow, (des_w, des_h), interpolation=cv2.INTER_NEAREST)
    else:
        raise Exception("Invalid resize flow method!")
    flow[:, :, 0] = flow[:, :, 0] * ratio_width
    flow[:, :, 1] = flow[:, :, 1] * ratio_height
    return flow


def render_flow(flow_project: torch.Tensor, render_fn) -> torch.Tensor:
    """
    Args:
        flow_project: (N, 2), 2 represents uv
    Returns:
        (2, H, W)
    """
    if flow_project.shape[-1] == 3:
        flow2render = flow_project
    else:
        flow2render = torch.cat(
            [flow_project, torch.zeros_like(flow_project[:, :1]).repeat(1, 3 - flow_project.shape[-1])], dim=-1
        )
        assert flow2render.shape[-1] == 3
    render_pkg_flow = render_fn(flow2render)
    flow_render = render_pkg_flow["render"][:2]
    return flow_render


def query_feat_map(query_coords: torch.Tensor, feat_map: torch.Tensor, mode="bilinear", padding_mode="zeros"):
    """
    Args:
        query_coords: (N, 2) in uv coordinates (uv <--> WH)
        feat_map: (H, W, C)
    Returns:
        (N, C)
    """
    H, W = feat_map.shape[:2]
    # uv_int = torch.clamp(
    #     query_coords.clone(),
    #     max=torch.tensor([W - 1, H - 1]).to(query_coords),
    # ).round()
    # uv_int = torch.clamp(uv_int, min=0)
    # uv_int = uv_int.long()
    # feat_uv = feat_map[uv_int[:, 1], uv_int[:, 0]]
    uv_grid = query_coords.clone()
    uv_grid[:, 0] = 2.0 * uv_grid[:, 0].clone() / max(W - 1, 1) - 1.0
    uv_grid[:, 1] = 2.0 * uv_grid[:, 1].clone() / max(H - 1, 1) - 1.0
    uv_grid = uv_grid.unsqueeze(0).unsqueeze(0)
    feat_map = feat_map.permute(2, 0, 1).unsqueeze(0)
    feat_uv = F.grid_sample(feat_map, uv_grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    feat_uv = feat_uv.permute(0, 2, 3, 1).squeeze(0).squeeze(0)
    return feat_uv


class CosineAnnealing:
    def __init__(self, warmup_step: int, total_step: int, max_value=1.0, min_value=0.0):
        self.warmup_step = warmup_step
        self.total_step = total_step
        self.max_value = max_value
        self.min_value = min_value
        self.enable = True

    def _linear(self, step: int):
        k = (self.max_value - self.min_value) / self.warmup_step
        return k * step + self.min_value

    def _cosine(self, step: int):
        return self.min_value + 0.5 * (self.max_value - self.min_value) * (
            1 + math.cos(math.pi * (step - self.warmup_step) / (self.total_step - self.warmup_step))
        )

    def get_value(self, step: int, decay=True):
        assert step >= 0, "step must be non-negative"
        if self.enable:
            if decay:
                if step <= self.warmup_step:
                    return self._linear(step)
                elif step <= self.total_step:
                    return self._cosine(step)
                else:
                    return self.min_value
            else:
                if step <= self.warmup_step:
                    return self._linear(step)
                else:
                    return self.max_value
        else:
            return self.max_value

    def __call__(self, *args, **kwargs):
        return self.get_value(*args, **kwargs)


def get_neighbor_cams(
    dataset: "list[Camera]", current_idx: int, current_cam: Camera = None, use_next=True, use_prev=True
):
    if current_cam is None:
        current_cam = dataset.__getitem__(current_idx, get_neighbors=False)
    next_cam = None
    prev_cam = None
    if use_next:
        # next_cam should be the right neighbor of viewpoint_cam in timeline (the same view for multiview dataset)
        try:
            next_cam = dataset.__getitem__(current_idx + 1, get_neighbors=False)
            if next_cam.time < current_cam.time:
                next_cam = None
        except IndexError:
            next_cam = None
    if use_prev:
        if current_idx - 1 >= 0:
            prev_cam = dataset.__getitem__(current_idx - 1, get_neighbors=False)
            if prev_cam.time > current_cam.time:
                prev_cam = None
    return next_cam, prev_cam


def gaussian_filter_2d(img: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.0) -> torch.Tensor:
        radius = math.ceil(num_sigmas * sigma)
        support = torch.arange(-radius, radius + 1, dtype=torch.float)
        kernel = torch.distributions.Normal(loc=0, scale=sigma).log_prob(support).exp_()
        # Ensure kernel weights sum to 1, so that image brightness is not altered
        return kernel.mul_(1 / kernel.sum())

    kernel_1d = gaussian_kernel_1d(sigma).to(img)  # Create 1D Gaussian kernel
    padding = len(kernel_1d) // 2  # Ensure that image size does not change
    img = img.unsqueeze(0).unsqueeze_(0)  # Need 4D data for ``conv2d()``
    # Convolve along columns and rows
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
    return img.squeeze_(0).squeeze_(0)  # Make 2D again
