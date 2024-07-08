"""
simple script which uses blender to load glb file and export scene as obj/ply file
adapted from objaverse repo: https://github.com/allenai/objaverse-rendering/blob/main/scripts/blender_script.py
"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple

import bpy
import numpy as np
from mathutils import Vector


parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument('-of', '--output_format', default='ply', choices=['ply', 'obj', 'glb'])
parser.add_argument('-o', '--output_path', default='/BS/databases24/objaverse/')
parser.add_argument('--normalize', default=False, action='store_true')

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale # this normalization includes lighting?
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return scale, offset

def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


if __name__ == '__main__':
    reset_scene()
    load_object(args.object_path)
    if args.normalize:
        normalize_scene()
    object_uid = os.path.basename(args.object_path).split(".")[0]

    # save output
    if args.output_format in ['ply', 'obj']:
        outfile = os.path.join(args.output_path, args.output_format+'-orig', f'{object_uid}.{args.output_format}')
    else:
        # glb file, by default from objaverse, it should have a /glbs/ in its full path
        assert '/glbs/' in args.object_path, 'input glb file path is invalid!'
        outfile = str(args.object_path).replace('/glbs/', '/glbs-normalized/')

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    if args.output_format == 'ply':
        bpy.ops.export_mesh.ply(filepath=outfile)
    elif args.output_format == 'obj':
        bpy.ops.export_mesh.obj(filepath=outfile)
    else:
        bpy.ops.export_scene.gltf(filepath=outfile, export_format='GLB')
