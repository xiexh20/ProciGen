"""
utility class to handle SMPLD scan paths, and other dataset path constants
"""
from os.path import join, dirname, basename


# Dataset paths, modify before you run.
SHAPENET_ROOT = '/BS/databases19/ShapeNet/ShapeNetCore.v2' # root to shapenet, format: ROOT/synset_id/ins_name/model/model_normalized.obj
OBJAVERSE_ROOT = '/BS/databases24/objaverse' # ROOT path to objaverse, this is used to save exported ply file
ABO_ROOT = "/BS/databases23/abo-3dmodels/3dmodels" # root path to all abo glb files, format: ROOT/model_uid.glb
PROCIGEN_ROOT = '/BS/databases24/ProciGen/' # root path to procigen sequences
MGN_ROOT = '/BS/databases24/mgn-smpld' # root path to SMPLD of MGN scans, format of texture image: ROOT/scan_id/scan_id.png

PROCIGEN_ASSET_ROOT = '/BS/xxie-6/static00/ProciGen-assets' # root path to additional procigen assets

BEHAVE_OBJECTS_ROOT = '/BS/xxie-5/static00/behave_release/objects' # path to BEHAVE templates, from this file: https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/objects.zip
BEHAVE_CORR_ROOT = f'{PROCIGEN_ASSET_ROOT}/corr-behave-objs' # path to the correspondence points for behave objects

# paths for object shape datasets, these files can be downloaded, see
SHAPENET_SIMPLIFIED_ROOT = '/BS/databases24/ShapeNetV2-simplified'  # path to simplified mesh path
ABO_MESHES_ROOT = SHAPENET_SIMPLIFIED_ROOT # path to abo exported meshes
OBJAVERSE_MESHES_ROOT = SHAPENET_SIMPLIFIED_ROOT

NEWSHAPE_CORR_ROOT = '/BS/xxie-2/static00/shapenet/' # root path to autoencoder output files for new object shapes

class ScanPath:
    "handling scan paths"
    def __init__(self, scan_folder):
        self.folder = scan_folder
        scan_folder = scan_folder[:-1] if scan_folder.endswith('/') else scan_folder
        self.name = basename(scan_folder) # name of the scan
        self.dataset = basename(dirname(scan_folder)) # dataset name of the scan

    def smpld_reg_obj(self):
        'registered smpld mesh obj file'
        return join(self.folder, f'{self.name}_reg.obj')

    def smpld_texture(self):
        "texture image for the registered smpld"
        return join(self.folder, f'{self.name}.jpg')

    def smpld_params(self):
        "th_good_1/125611487366942/125611487366942_unposed.pkl"
        return join(self.folder, f'{self.name}_unposed.pkl')

