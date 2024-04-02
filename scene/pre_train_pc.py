import torch
import numpy as np
import cv2


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    
    # tensor(0.9961, device='cuda:0')
    # torch.Size([3, 480, 640])
    # tensor(2.0256, device='cuda:0')
    # torch.Size([1, 480, 640])
    CX = intrinsics[2]
    CY = intrinsics[3]
    FX = intrinsics[0]
    FY = intrinsics[1]

    # cv2.imwrite('test_dep.png', ((depth*mask).transpose(1,2,0)/depth.max()*255).astype(np.uint8))
    # Compute indices of pixels
    x_grid, y_grid = np.meshgrid(np.arange(width).astype(np.float32), 
                                    np.arange(height).astype(np.float32),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)
    mask = mask[0].reshape(-1).astype(np.bool8)

    # Initialize point cloud
    pts_cam = np.stack((xx * depth_z, yy * depth_z, depth_z), axis=-1)
    if transform_pts:
        pix_ones = np.ones(height * width, np.float32)[:, None]
        pts4 = np.concatenate((pts_cam, pix_ones), axis=1)
        c2w = np.linalg.inv(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam
    
    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    # Colorize point cloud
    cols = np.transpose(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    # Select points based on mask
    if mask is not None:
        pts = pts[mask]
        cols = cols[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return pts, cols, mean3_sq_dist
    else:
        return pts, cols
    
def get_pc_only(depth, mask, intrinsics, w2c):
    '''
    depth: [1, h, w] in mm
    mask: [1, h, w]
    width, height: int
    '''
    width, height = depth.shape[2], depth.shape[1]
    CX = intrinsics[2]
    CY = intrinsics[3]
    FX = intrinsics[0]
    FY = intrinsics[1]

    # cv2.imwrite('test_dep.png', ((depth*mask).transpose(1,2,0)/depth.max()*255).astype(np.uint8))
    # Compute indices of pixels
    x_grid, y_grid = np.meshgrid(np.arange(width).astype(np.float32), 
                                    np.arange(height).astype(np.float32),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)
    mask = mask[0].reshape(-1).astype(np.bool8)

    # Initialize point cloud
    pts_cam = np.stack((xx * depth_z, yy * depth_z, depth_z), axis=-1)
    pix_ones = np.ones(height * width, np.float32)[:, None]
    pts4 = np.concatenate((pts_cam, pix_ones), axis=1)
    c2w = np.linalg.inv(w2c)
    pts = (c2w @ pts4.T).T[:, :3]
    # Select points based on mask
    if mask is not None:
        pts = pts[mask]
    
    return pts