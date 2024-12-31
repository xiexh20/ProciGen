"""
some helper functions
"""
"""
randomly select two latent code and try to interpolate and render them
"""
import argparse
import json

import cv2
import numpy as np
import os
import torch
from tqdm import tqdm

from pytorch3d.renderer import look_at_view_transform


def render_1view(verts, faces, R, T, device='cuda', resolution=(512, 512),
                 vc=None, light_center=[[0.0, 3.0, 0]]):
    """
    return one RGB image given vertices and faces
    :param verts: torch tensor
    :param faces:
    :param R:
    :param T:
    :return:
    """
    from pytorch3d.renderer import (
        look_at_view_transform,
        PerspectiveCameras,
        FoVPerspectiveCameras,
        PointLights,
        Materials,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        TexturesVertex,
        HardPhongShader
    )
    from pytorch3d.structures import Meshes

    raster_settings = RasterizationSettings(
        image_size=resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(device=device, location=light_center)
    if vc is None:
        text = TexturesVertex(torch.from_numpy(np.array([[0.65098039, 0.74117647, 0.85882353]]).repeat(len(verts), 0))[None].to(
                device).float())
    else:
        assert len(vc) == len(verts) or len(vc) == 3, f'given vertex shape={verts.shape}, vc shape={vc.shape}'
        text = TexturesVertex(torch.from_numpy(vc)[None].to(device).float())

    # print(type(verts), type(faces))
    mesh_cuda = Meshes(torch.from_numpy(verts.copy())[None].float().to(device),
                       torch.from_numpy(faces.copy().astype(int))[None].long().to(device),
                       text)
    cameras = PerspectiveCameras(device=device, R=R, T=T)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    images = renderer(mesh_cuda)
    return (images[0, ..., :3].cpu().numpy()*255).astype(np.uint8)


def render_360views(verts, faces, device='cuda', resolution=(512, 512), num_views=30, vc=None):
    rends = []
    for i in tqdm(range(num_views)):
        azim = 360 * i / num_views # no need to rotate by 180 degree
        R, T = look_at_view_transform(2.0, 15, azim)
        rend = render_1view(verts, faces, R, T, device, resolution, vc)
        rends.append(rend)
    return rends


