"""
base class to render using blender
Author: Xianghui, July 08, 2024
Cite: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation
"""
import sys, os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # to allow reading exr image
import cv2

sys.path.append(os.getcwd())
import bpy
from mathutils import Vector
from os.path import join
import numpy as np
import os.path as osp
import math

from data.kinect_transform import CameraTransform
from lib_smpl.th_hand_prior import load_grab_prior
import paths as paths
bpy_version = bpy.app.version_string


class BaseRenderer:
    def __init__(self, camera_config, camera_count, ext='jpg',
                 loader=None, reso_x=2048, reso_y=1536, icap=False):
        self.loader = loader # HOI synthesizer, useful for synthesize new objects
        self.camera_config = camera_config
        self.cam_transform = CameraTransform(camera_config, camera_count)
        self.camera_count = camera_count
        self.ext = ext

        self.reso_x, self.reso_y = reso_x, reso_y
        # init blender scene and saving settings
        self.debug = False
        self.icap = icap
        self.init_render()
        self.mesh_names = []
        self.mean_hand_pose = np.concatenate([x['mean'] for x in load_grab_prior('assets')])

    def smpl2smplh(self, pose):
        """
        add mean hand pose to the smpl body pose parameters
        :param pose: (72, ) or (B, 72)
        :type pose: np.array
        :return: (156, ) or (B, 156)
        :rtype:
        """
        if len(pose.shape) == 1:
            pose_new = np.zeros((156, ))
            pose_new[:66] = pose[:66]
            pose_new[66:] = self.mean_hand_pose
        elif len(pose.shape) == 2:
            bs = pose.shape[0]
            pose_new = np.zeros((bs, 156))
            pose_new[:, :66] = pose[:, :66]
            pose_new[:, 66:] = self.mean_hand_pose
        else:
            raise ValueError(f"Invalid input pose shape: {pose.shape}!")
        return pose_new

    def init_render(self):
        # Empty the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # setup
        scene = bpy.data.scenes[0]
        scene.use_nodes = True
        scene.render.resolution_x = self.reso_x
        scene.render.resolution_y = self.reso_y
        scene.render.resolution_percentage = 100
        bpy.context.scene.world.use_nodes = False
        bpy.context.scene.render.engine = 'CYCLES' # use ray-tracing
        bpy.context.scene.render.image_settings.quality = 100

        # Build scene
        bpy.context.scene.world.color = (1.0, 1.0, 1.0) if bpy_version >= '3.0' else (0, 0, 0) # bpy 3.0+ seems to have different ambient color setting
        bpy.context.scene.render.film_transparent = True
        self.init_lights(depth=2.5)
        if bpy_version < '3.0':
            bpy.context.scene.world.light_settings.use_ambient_occlusion = True  # enable ambient light

        # Add camera
        self.add_camera()

        # Configure output
        if bpy_version >= '3.0':
            bpy.context.scene.view_layers['ViewLayer'].use_pass_combined = True
            bpy.context.scene.view_layers['ViewLayer'].use_pass_diffuse_color = True
            bpy.context.scene.view_layers['ViewLayer'].use_pass_z = True
        else:
            bpy.context.scene.view_layers['View Layer'].use_pass_combined = True
            bpy.context.scene.view_layers['View Layer'].use_pass_diffuse_color = True
            bpy.context.scene.view_layers['View Layer'].use_pass_z = True

        bpy.context.scene.use_nodes = True
        scene_node_tree = bpy.context.scene.node_tree
        for n in scene_node_tree.nodes:
            scene_node_tree.nodes.remove(n)

        render_layer = scene_node_tree.nodes.new(type="CompositorNodeRLayers")
        output_rgb = scene_node_tree.nodes.new(type="CompositorNodeOutputFile")
        # for bpy>3.0, empty means ./, hence we need / to use abs path. if set to empty, will be /tmp/
        # output_rgb.base_path = '/' if bpy_version > '3.0' and bpy_version < '4.0' else ''
        output_rgb.base_path = ''
        output_rgb.file_slots[0].use_node_format = True
        output_rgb.format.file_format = "JPEG" if self.ext == 'jpg' else "PNG"
        output_rgb.format.color_mode = 'RGB'
        scene_node_tree.links.new(render_layer.outputs['Image'], output_rgb.inputs['Image'])
        self.render_layer, self.output_rgb = render_layer, output_rgb  # save for later manipulation

        output_depth = scene_node_tree.nodes.new(type="CompositorNodeOutputFile")
        # output_depth.base_path = '/' if bpy_version > '3.0' and bpy_version < '4.0' else ''
        output_depth.base_path = ''
        output_depth.file_slots[0].use_node_format = True
        output_depth.format.file_format = "OPEN_EXR"
        output_depth.format.color_depth = '16'
        scene_node_tree.links.new(render_layer.outputs['Depth'], output_depth.inputs['Image'])
        self.output_depth = output_depth

        # rendering setting
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.device = 'GPU'
        # Enable CUDA
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

        self.deselect_all()

    def add_camera(self):
        "setup the camera"
        if not self.icap:
            K = np.array([[979.784, 0, 1018.952],
                          [0, 979.840, 779.486],
                          [0, 0, 1]])
            width, height = 2048, 1536
        else:
            ICAP_FOCALs = np.array([[918.457763671875, 918.4373779296875], [915.29962158203125, 915.1966552734375],
                                    [912.8626708984375, 912.67633056640625], [909.82025146484375, 909.62469482421875],
                                    [920.533447265625, 920.09722900390625], [909.17633056640625, 909.23529052734375]])
            ICAP_CENTERs = np.array([[956.9661865234375, 555.944580078125], [956.664306640625, 551.6165771484375],
                                     [956.72003173828125, 554.2166748046875], [957.6181640625, 554.60296630859375],
                                     [958.4615478515625, 550.42987060546875], [956.14801025390625, 555.01593017578125]])
            K = np.array([[ICAP_FOCALs[0, 0], 0, ICAP_CENTERs[0, 0]],
                          [0, ICAP_FOCALs[0, 1], ICAP_CENTERs[0, 1]],
                          [0, 0, 1]])
            width, height = 1920, 1080

        E = np.eye(4)  # world to camera transform for the object to be rendered

        R_bcam2cv = np.array([[1, 0, 0],
                              [0, -1., 0],
                              [0, 0, -1.]])
        bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(0, 0, 0))
        rotation = E[:3, :3] @ R_bcam2cv
        location = (E[0, 3], E[1, 3], E[2, 3])
        camera = bpy.data.objects['Camera']
        camera.location = location
        loc_mat = np.eye(4)
        loc_mat[:, 3] = E[:, 3]
        rot_mat = np.eye(4)
        rot_mat[:3, :3] = rotation
        camera.matrix_world = np.matmul(loc_mat, rot_mat)
        camera.data.type = 'PERSP'
        camera.data.lens_unit = "FOV"
        camera.data.angle = 2 * np.arctan2(K[0, 2], K[0, 0])  # ~=1.6187871139786363 for behave
        camera.data.clip_start = 0.01
        camera.data.clip_end = 3000
        camera.data.shift_x = -(K[0, 2] / width - 0.5)
        camera.data.shift_y = (K[1, 2] - 0.5 * height) / width
        bpy.context.scene.camera = camera

    def clear_meshes(self):
        self.deselect_all()
        for name in self.mesh_names:
            bpy.data.objects[name].select_set(True)  # Blender 2.8x
            bpy.ops.object.delete()
        self.mesh_names.clear()

    def deselect_all(self):
        bpy.ops.object.select_all(action='DESELECT')  # unselect all objects first

    def init_lights(self, depth=2.0):
        # top left
        bpy.ops.object.light_add(type='POINT', radius=1.0, location=(-2, -2, depth * 2))
        if self.debug:
            bpy.context.object.data.energy = 20
        else:  # random lighting
            bpy.context.object.data.energy = np.random.uniform(0.1, 100)
        # top middle
        bpy.ops.object.light_add(type='POINT', radius=1.0, location=(0, -2, depth * 2))
        if self.debug:
            bpy.context.object.data.energy = 20
        else:  # random lighting
            bpy.context.object.data.energy = np.random.uniform(0.1, 100)
        # top right
        bpy.ops.object.light_add(type='POINT', radius=1.0, location=(2, -2, depth * 2))
        if self.debug:
            bpy.context.object.data.energy = 20
        else:  # random lighting
            bpy.context.object.data.energy = np.random.uniform(0.1, 100)
        # bottom left
        bpy.ops.object.light_add(type='POINT', radius=1.0, location=(-2, -2, 0.))
        if self.debug:
            bpy.context.object.data.energy = 20
        else:  # random lighting
            bpy.context.object.data.energy = np.random.uniform(0.1, 100)
        # bottom middle
        bpy.ops.object.light_add(type='POINT', radius=1.0, location=(0., -2, 0.))
        if self.debug:
            bpy.context.object.data.energy = 20
        else:  # random lighting
            bpy.context.object.data.energy = np.random.uniform(0.1, 100)
        # bottom right
        bpy.ops.object.light_add(type='POINT', radius=1.0, location=(2, -2, 0.))
        if self.debug:
            bpy.context.object.data.energy = 20
        else:  # random lighting
            bpy.context.object.data.energy = np.random.uniform(0.1, 100)
        # middle left
        bpy.ops.object.light_add(type='POINT', radius=1.0, location=(-2, -2, depth))
        if self.debug:
            bpy.context.object.data.energy = 20
        else:  # random lighting
            bpy.context.object.data.energy = np.random.uniform(0.1, 100)
        # center
        bpy.ops.object.light_add(type='POINT', radius=1.0, location=(0, -2.5, depth))
        if self.debug:
            bpy.context.object.data.energy = 20
        else:  # random lighting
            bpy.context.object.data.energy = np.random.uniform(0.1, 100)
        # middle right
        bpy.ops.object.light_add(type='POINT', radius=1.0, location=(2, -2., depth))
        if self.debug:
            bpy.context.object.data.energy = 20
        else:  # random lighting
            bpy.context.object.data.energy = np.random.uniform(0.1, 100)
        bpy.context.object.data.cycles.cast_shadow = True

    def normalize_scene(self):
        "normalize the objects according to objaverse convention"
        bbox_min, bbox_max = self.scene_bbox()
        scale = 1.0 / max(bbox_max - bbox_min)
        for obj in self.scene_root_objects():
            if obj.name.lower() == 'camera':
                # do not change camera!
                continue
            obj.scale = obj.scale * scale  # this scale is always centered around object?
        # Apply scale to matrix_world.
        bpy.context.view_layer.update()
        bbox_min, bbox_max = self.scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        for obj in self.scene_root_objects():
            if obj.name.lower() == 'camera':
                continue
            obj.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")

    @staticmethod
    def reset_scene() -> None:
        """Resets the scene to a clean state."""
        # delete everything that isn't part of a camera or a light
        for obj in bpy.data.objects:
            if obj.type not in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)
        # delete all the materials
        for material in bpy.data.materials:
            material.user_clear() # remove fake users
            bpy.data.materials.remove(material, do_unlink=True)
        # delete all the textures
        for texture in bpy.data.textures:
            texture.user_clear() # remove fake users
            bpy.data.textures.remove(texture, do_unlink=True)
        # delete all the images
        for image in bpy.data.images:
            image.user_clear() # remove fake users
            bpy.data.images.remove(image, do_unlink=True)

    def reinit_light(self):
        "reinitialize some random lights"
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj, do_unlink=True)
        self.init_lights(depth=2.5)

    @staticmethod
    def scene_root_objects():
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj
    @staticmethod
    def scene_bbox(single_obj=None, ignore_matrix=False):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        for obj in BaseRenderer.scene_meshes() if single_obj is None else [single_obj]:
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

    def import_object2blender(self, dname, ins, obj_mat, synset_id):
        """
        import object mesh/scene into blender scene
        :param dname: dataset name of the object
        :param ins: instance id of the object
        :param obj_mat: 4x4 transformation applied to the object
        :param synset_id: object class/synset id
        :return:
        :rtype:
        """
        blender_name_hum, blender_name_obj = 'procigen-hum', 'procigen-obj'
        if dname == 'shapenet':
            # simply apply the parameter
            mesh_file = osp.join(paths.SHAPENET_ROOT, synset_id, ins, 'models/model_normalized.obj')
            bpy.ops.import_scene.obj(filepath=mesh_file, axis_forward='Y', axis_up='Z')
            obj_object = bpy.context.selected_objects[-1]
            obj_object.name = blender_name_obj  # specify the name for the imported one
            bpy.ops.object.shade_smooth()
            obj = bpy.data.objects[blender_name_obj]  # reselect
            obj.select_set(True)
            self.mesh_names.append(blender_name_obj)  # this allows deletion
            self.apply_transform(obj, obj_mat)  # now apply transformation to the object
        elif dname == 'objaverse':
            # use objaverse lib to get path
            import objaverse
            uids = [ins]
            # Get local glb file path, if non-existent, will download using uid.
            # Warning: objaverse downloads are saved to home directory by default, if that is not desired,
            # change the BASE_PATH in objaverse.__init__.py file.
            objs = objaverse.load_objects(uids, download_processes=1)
            obj_values = list(objs.values())
            assert len(obj_values) == 1, f'invalid object paths found: {obj_values}'
            glb_path = obj_values[0]
            # Note: the saved object transformation is always w.r.t normalized object meshes
            # Option 1: do normalization inside blender after loading
            self.add_glb_object(glb_path, obj_mat, normalize=True)

            # Option 2: export a normalized glb file, and reload to blender, should also work.
            # export a normalized glb file
            # glb_path_norm = str(glb_path).replace('/glbs/', '/glbs-normalized/')
            # if not osp.isfile(glb_path_norm):
            #     cmd = f'blender -b -P render/blender_export.py -- --object_path {glb_path} --normalize --output_format glb'
            #     os.system(cmd)
            # self.add_glb_object(glb_path_norm, obj_mat, normalize=False)
        else:
            # Note: for abo dataset, the saved object transformation is w.r.t original glb file, w/o further processing.
            glb_path = f'{paths.ABO_ROOT}/{ins}.glb'
            self.add_glb_object(glb_path, obj_mat, normalize=False)
        return blender_name_hum

    def add_glb_object(self, glb_path, obj_mat, normalize):
        """
        load object from a glb file
        :param glb_path: glb file path
        :param obj_mat: the transformation applied to the object after loading
        :param normalize: normalize the scene or not, for objaverse, it should be normalized
        :return:
        """
        bpy.ops.import_scene.gltf(filepath=glb_path, merge_vertices=True)
        if normalize:
            # this normalization
            self.normalize_scene()  # do not do normalization, since the glb file is already normalized!
        # apply transform to the object
        for obj in self.scene_root_objects():
            if obj.type in ["CAMERA", 'LIGHT']:
                continue  # do not change lighting or camera!
            original_mat = np.array(obj.matrix_world)
            obj.matrix_world = (obj_mat @ original_mat).T

    @staticmethod
    def scene_meshes():
        for obj in bpy.context.scene.objects.values():
            if isinstance(obj.data, (bpy.types.Mesh)):
                yield obj

    def apply_transform(self, obj, mat):
        """
        apply a transformation to the object
        make sure the object is selected!
        :param obj: bpy object
        :param mat: 4x4 numpy array
        :return:
        """
        assert mat.shape == (4, 4), f'the given matrix shape ({mat.shape}) is not 4x4!'
        # first make sure all existing transform are applispot =ed
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        # transform = Matrix(mat)
        original_mat = np.array(obj.matrix_world)
        assert np.allclose(original_mat, np.eye(4)), 'the original matrix is not identity!'
        obj.matrix_world = (mat @ original_mat).T # bpy matrix is transposed!

        # sanity check: make sure the transform is applied
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        newmat = np.array(obj.matrix_world)
        assert np.allclose(newmat, np.eye(4)), 'transformation was not applied successfully to the given object!'

    def add_textured_mesh_from_file(self, obj_file, text_file, name):
        """
        load obj file and append texture directly
        :param obj_file:
        :param text_file:
        :return:
        """
        bpy.ops.import_scene.obj(filepath=obj_file, axis_forward='Y', axis_up='Z')
        obj_object = bpy.context.selected_objects[-1]
        obj_object.name = name # specify the name for the imported one

        self.assign_texture_image(name, text_file)

    def assign_texture_image(self, name, text_file):
        "load texture image and assign it to the object specified by name"
        bpy.ops.object.select_all(action='DESELECT')
        obj = bpy.data.objects[name]
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        # Add material and texture, this changes the original material properties
        object_material = bpy.data.materials.new('object_material')
        object_material.use_nodes = True
        bsdf = object_material.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Specular'].default_value = 0
        object_texture = object_material.node_tree.nodes.new('ShaderNodeTexImage')
        object_texture.image = bpy.data.images.load(text_file)
        object_material.node_tree.links.new(bsdf.inputs['Base Color'], object_texture.outputs['Color'])
        obj.active_material = object_material
        self.mesh_names.append(name)

    def transforom_hum_obj_local(self, transform):
        for obj in self.scene_root_objects():
            if obj.type in ["CAMERA", 'LIGHT']:
                continue  # do not change lighting or camera!
            original_mat = np.array(obj.matrix_world)
            obj.matrix_world = (transform @ original_mat).T
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    def format_outfiles(self, depth_full, depth_obj, frame_folder, k):
        """
        save output depth and masks
        :param depth_full: depth with both human and object
        :param depth_obj: depth of only object
        :param frame_folder: output frame folder
        :param k: camera index
        :return:
        :rtype:
        """
        depth_obj[depth_obj == np.max(depth_obj)] = 0
        depth_full[depth_full == np.max(depth_full)] = 0
        mask_obj = (depth_obj <= depth_full) & (depth_obj > 0)
        mask_hum = (~mask_obj) & (depth_full > 0)
        mask_obj_full = depth_obj > 0

        mask_hum = mask_hum.astype(np.uint8) * 255
        mask_obj = mask_obj.astype(np.uint8) * 255
        mask_obj_full = mask_obj_full.astype(np.uint8) * 255

        rgb_file = join(frame_folder, f'k{k}.color.0001.{self.ext}')
        os.system(f'mv {rgb_file} {rgb_file.replace(".0001.jpg", ".jpg")}')

        # save results
        cv2.imwrite(join(frame_folder, f'k{k}.person_mask.png'),  mask_hum)
        cv2.imwrite(join(frame_folder, f'k{k}.obj_rend_mask.png'), mask_obj)
        cv2.imwrite(join(frame_folder, f'k{k}.obj_rend_full.png'), mask_obj_full)
        cv2.imwrite(join(frame_folder, f'k{k}.depth.png'), (depth_full * 1000).astype(np.uint16))
