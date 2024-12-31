"""
some paths for behave objects
"""
import os.path as osp

import numpy as np

# path to the simplified mesh used for registration
_mesh_template = {
    "backpack": "backpack/backpack_f1000.ply",
    'basketball': "basketball/basketball_f1000.ply",
    'boxlarge': "boxlarge/boxlarge_f1000.ply",
    'boxtiny': "boxtiny/boxtiny_f1000.ply",
    'boxlong': "boxlong/boxlong_f1000.ply",
    'boxsmall': "boxsmall/boxsmall_f1000.ply",
    'boxmedium': "boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard': "keyboard/keyboard_f1000.ply",
    'plasticcontainer': "plasticcontainer/plasticcontainer_f1000.ply",
    'stool': "stool/stool_f1000.ply",
    'tablesquare': "tablesquare/tablesquare_f2000.ply",
    'toolbox': "toolbox/toolbox_f1000.ply",
    "suitcase": "suitcase/suitcase_f1000.ply",
    'tablesmall': "tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball': "yogaball/yogaball_f1000.ply",
    'trashbin': "trashbin/trashbin_f1000.ply",

    # ABO objects, simply copy them
    'abo-chairblack': "chairblack/chairblack_f2500.ply",
    'abo-chairwood': "chairwood/chairwood_f2500.ply",
    'abo-tablesquare': "tablesquare/tablesquare_f2000.ply",
    'abo-tablesmall': "tablesmall/tablesmall_f1000.ply",
    'obja-chairblack': "chairblack/chairblack_f2500.ply",
    'obja-chairwood': "chairwood/chairwood_f2500.ply",

}

_icap_template = {
    'obj01': '/BS/xxie-2/work/InterCap/obj_track/objs/01.ply', #'suitcase'
    'obj02': '/BS/xxie-2/work/InterCap/obj_track/objs/02.ply', #  'skateboard'
    'obj03': '/BS/xxie-2/work/InterCap/obj_track/objs/03.ply', # 'football'
    'obj04': '/BS/xxie-2/work/InterCap/obj_track/objs/04.ply', # 'umbrella'
    'obj05': '/BS/xxie-2/work/InterCap/obj_track/objs/05.ply', # 'tennis-racket'
    'obj06': '/BS/xxie-2/work/InterCap/obj_track/objs/06.ply', # toolbox
    'obj07': '/BS/xxie-2/work/InterCap/obj_track/objs/07.ply', #  chair01
    'obj08': '/BS/xxie-2/work/InterCap/obj_track/objs/08.ply', #'bottle'
    'obj09': '/BS/xxie-2/work/InterCap/obj_track/objs/09.ply', # 'cup'
    'obj10': '/BS/xxie-2/work/InterCap/obj_track/objs/10.ply', # 'chair02', stool
    'abo-obj07': '/BS/xxie-2/work/InterCap/obj_track/objs/07.ply',
    'obja-obj07': '/BS/xxie-2/work/InterCap/obj_track/objs/07.ply'
}

def get_template_path(obj_name, behave_path="/BS/xxie-5/static00/behave_release/objects"):
    if obj_name in _mesh_template:
        path = _mesh_template[obj_name]
        temp_path = osp.join(behave_path, path)
    else:
        # intercap
        temp_path = _icap_template[obj_name]
    assert osp.isfile(temp_path), f'{temp_path} does not exist!'
    return temp_path


_watertight_templates = {'backpack': 'backpack/backpack_f1000_cent_watertight_fused.ply',
                         'basketball': 'basketball/basketball_f1000_cent_watertight.ply',
                         'boxlarge': 'boxlarge/boxlarge_f1000_cent_watertight_fused.ply',
                         'boxtiny': 'boxtiny/boxtiny_f1000_cent_watertight.ply',
                         'boxlong': 'boxlong/boxlong_f1000_cent_watertight_fused.ply',
                         'boxsmall': 'boxsmall/boxsmall_f1000_cent_watertight.ply',
                         'boxmedium': 'boxmedium/boxmedium_f1000_cent_watertight.ply',
                         'chairblack': 'chairblack/chairblack_f2500_cent_watertight.ply',
                         'chairwood': 'chairwood/chairwood_f2500_cent_watertight.ply',
                         'monitor': 'monitor/monitor_closed_f1000_cent_watertight.ply',
                         'keyboard': 'keyboard/keyboard_f1000_cent_watertight_fused.ply',
                         'plasticcontainer': 'plasticcontainer/plasticcontainer_f1000_cent_watertight.ply',
                         'stool': 'stool/stool_f1000_cent_watertight_fused.ply',
                         'tablesquare': 'tablesquare/tablesquare_f2000_cent_watertight.ply',
                         'toolbox': 'toolbox/toolbox_f1000_cent_watertight.ply',
                         'suitcase': 'suitcase/suitcase_f1000_cent_watertight.ply',
                         'tablesmall': 'tablesmall/tablesmall_f1000_cent_watertight_fused.ply',
                         'yogamat': 'yogamat/yogamat_f1000_cent_watertight_fused.ply',
                         'yogaball': 'yogaball/yogaball_f1000_cent_watertight.ply',
                         'trashbin': 'trashbin/trashbin_f1000_cent_watertight.ply'}

TEMPLATE_AXIS_ALIGNED = {
    "backpack": "/BS/xxie-5/static00/behave_release/objects/backpack/backpack_f1000_aligned.ply",
    "boxlarge": "/BS/xxie-5/static00/behave_release/objects/boxlarge/boxlarge_f1000_aligned.ply",
    "boxlong": "/BS/xxie-5/static00/behave_release/objects/boxlong/boxlong_f1000_aligned.ply",
    "boxsmall": "/BS/xxie-5/static00/behave_release/objects/boxsmall/boxsmall_f1000_aligned.ply",
    "boxtiny": "/BS/xxie-5/static00/behave_release/objects/boxtiny/boxtiny_f1000_aligned.ply",
    "boxmedium": "/BS/xxie-5/static00/behave_release/objects/boxmedium/boxmedium_f1000_aligned.ply",
    "yogamat": "/BS/xxie-5/static00/behave_release/objects/yogamat/yogamat_f1000_aligned.ply",
    "plasticcontainer": "/BS/xxie-5/static00/behave_release/objects/plasticcontainer/plasticcontainer_f1000_aligned.ply",
    "suitcase": "/BS/xxie-5/static00/behave_release/objects/suitcase/suitcase_f1000_aligned.ply",
    "stool": "/BS/xxie-5/static00/behave_release/objects/stool/stool_f1000_aligned.ply",
    "basketball": "/BS/xxie-5/static00/behave_release/objects/basketball/basketball_f1000_aligned.ply",
    "yogaball": "/BS/xxie-5/static00/behave_release/objects/yogaball/yogaball_f1000_aligned.ply"
}





CORR_FILES = {
    "chairwood": 'chairwood_f5000_corr.ply',
    "chairblack": "chairblack_f2500_corr.ply",

    "tablesquare": "tablesquare_corr.ply",
    "tablesmall": "tablesmall_corr.ply",
    "keyboard": "keyboard_corr.ply",
    "trashbin": "trashbin_corr.ply",
    "monitor": "monitor_corr.ply",
    "toolbox": "toolbox_corr.ply",
    "plasticcontainer": "plasticcontainer-plasticcontainer_corr.ply",

    # objects from objaverse
    "stool": "stool_corr.ply",
    "suitcase": "suitcase_corr.ply",
    "backpack": "backpack_corr.ply",
    "boxlarge": "boxlarge_corr.ply",
    "boxmedium": "boxmedium_corr.ply",
    "boxtiny": "boxtiny_corr.ply",
    "boxlong": "boxlong_corr.ply",
    "boxsmall": "boxsmall_corr.ply",
    "basketball": "basketball_corr.ply",
    "yogaball": "yogaball_corr.ply",

    # Objaverse chairs
    'obja-chairwood':"chairwood-obja-chair_corr.ply",
    "obja-chairblack":"chairblack-obja-chair_corr.ply",
    "obja-obj07":"obj07-obja-chair_corr.ply",

    # ABO objects
    "abo-chairwood": 'chairwood-abo-chair_corr.ply',
    "abo-chairblack": "chairblack-abo-chair_corr.ply",
    "abo-tablesquare": "tablesquare-abo-table_corr.ply",
    "abo-tablesmall": "tablesmall-abo-table_corr.ply",
    "abo-obj07": "obj07-abo-chair_corr.ply"

}


def get_corr_mesh_file(obj_name):
    if obj_name in CORR_FILES:
        return CORR_FILES[obj_name]
    return f'{obj_name}_corr.ply'


def load_template_watertight(obj_name, cent=True, high_reso=False):
    "load watertight object template mesh given object name"
    temp_path = osp.join("/BS/xxie-5/static00/behave_release/objects", _watertight_templates[obj_name])
    from psbody.mesh import Mesh
    return Mesh(filename=temp_path)


def load_axis_aligned_template(obj_name):
    if obj_name not in TEMPLATE_AXIS_ALIGNED.keys():
        return None
    file = TEMPLATE_AXIS_ALIGNED[obj_name]
    from psbody.mesh import Mesh
    m = Mesh(filename=file)
    # avg = np.mean(m.v, 0) # we don't need the template to be centered
    # assert np.allclose(np.zeros(3), avg), f'the given template is not centered! mean={avg}'
    return m
