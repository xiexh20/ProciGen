"""
util functions for rendering
"""

_icap_template = {
    'obj01': '/objs/01.ply', #'suitcase'
    'obj02': '/objs/02.ply', #  'skateboard'
    'obj03': '/objs/03.ply', # 'football'
    'obj04': '/objs/04.ply', # 'umbrella'
    'obj05': '/objs/05.ply', # 'tennis-racket'
    'obj06': '/objs/06.ply', # toolbox
    'obj07': '/objs/07.ply', #  chair01
    'obj08': '/objs/08.ply', #'bottle'
    'obj09': '/objs/09.ply', # 'cup'
    'obj10': '/objs/10.ply' # 'chair02', stool
}


def get_shape_datasetname(seq_name:str):
    "given ProciGen sequence name, return the name of the object shape dataset used to synthesize this seq"
    obj_name = seq_name.split('_')[2]
    if '_abo' in seq_name:
        return 'abo'
    elif obj_name in ['backpack', 'basketball', 'boxlarge',
                      'boxlong', 'boxmedium', 'boxsmall', 'boxtiny',
                      'stool', 'stuicase', 'yogaball',
                      'obj10', 'obj01', 'obj03']: # the last row are seqs with hoi from InterCap
        return 'objaverse'
    else:
        return 'shapenet'