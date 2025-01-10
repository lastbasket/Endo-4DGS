#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import cv2
import torch
import math
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh
from utils.loss_utils import get_smallest_axis
from time import time as get_time
import torch.nn.functional as F
from utils.sh_utils import RGB2SH


def render(viewpoint_camera, gs, pipe, bg_color: torch.Tensor, scaling_modifier = 1.0, \
        override_color = None, stage="fine", cam_type=None, multi_scale=False, iteration=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gs.get_xyz, dtype=gs.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    means3D = gs.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=gs.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
    
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    norm_rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # add deformation to each points
    # deformation = pc.get_deformation
    means2D = screenspace_points
    opacity = gs._opacity
    shs = gs.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = gs.get_covariance(scaling_modifier)
    else:
        scales = gs._scaling
        rotations = gs._rotation
        
    mask_deform = gs._deformation_table
    
    if stage == "coarse" :
        # means3D_deform, scales_deform, rotations_deform, opacity_deform, shs_deform = means3D, scales, rotations, opacity, shs
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    else:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = gs._deformation(means3D, scales, 
                                                                rotations, opacity, shs, time)
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    # print(viewpoint_camera.camera_center)
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = gs.get_features.transpose(1, 2).view(-1, 3, (gs.max_sh_degree+1)**2)
            dir_pp = (gs.get_xyz - viewpoint_camera.camera_center.cuda().repeat(gs.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gs.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color
        
    
    scales_final = gs.scaling_activation(scales_final)
    rotations_final = gs.rotation_activation(rotations_final)
    opacity_final = gs.opacity_activation(opacity_final)
    
    rendered_image, radii, depth, weight = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final)
    
    normal = get_smallest_axis(rotations_final, scales_final)
    normal_sh = RGB2SH((normal+1)/2)
    
    
    normal_shs = shs_final.clone()
    normal_shs[:, 0, :3] = normal_sh
    
    
    normal_map, rad_norm, dep_norm, weight_norm = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = normal_shs,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final)
    
    re_dict = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "radii": radii,
            "depth": depth.unsqueeze(0),
            'normal': normal_map,
            'confidence': weight.unsqueeze(0)}
    
    return re_dict

