import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import shutil

from objfile import ObjFile


def vis_mesh(file_path,color=(0,0,0)):
    normalized_color = tuple(x / 255.0 for x in color)
    tmp_dir, basename = osp.split(file_path)
    basename, _ = osp.splitext(basename)
    base = basename.split('_')[0]
    tmp_dir = osp.join(tmp_dir, f'{base}_tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    ob = ObjFile(file_path,color=normalized_color)
    ob.Plot(osp.join(tmp_dir, f'{basename}.png'), elevation=90, azim=-90, dpi=None, scale=None, animate=None)
    return cv2.imread(osp.join(tmp_dir, f'{basename}.png'))
    
def vis_3_meshes(filename, framework_base, color, color_p, color_f):
    file_path = 'results/'+ filename + '_' +  framework_base + '.obj'
    file_path_p = 'results/'+ filename + '_p' + framework_base + '.obj'
    file_path_f = 'results/'+ filename + '_f' + framework_base + '.obj'

    vis_mesh(file_path,color=color)
    vis_mesh(file_path_p,color=color_p)
    vis_mesh(file_path_f,color=color_f)



filename = 'image18634'
vis_3_meshes(filename, '', (87, 208, 203), (141,212,208), (58,141,150))
vis_3_meshes(filename, '_gt',  (87, 208, 203), (141,212,208), (58,141,150)) # (255, 148, 138), (255,212,172), (225,70,88)
vis_3_meshes(filename, '_baseline', (87, 208, 203), (141,212,208), (58,141,150))

# filename = 'image23206'
# vis_3_meshes(filename, '', (87, 208, 203),  (141,212,208), (87, 208, 203))
# vis_3_meshes(filename, '_gt',  (87, 208, 203), (141,212,208), (58,141,150))
# vis_3_meshes(filename, '_baseline', (87, 208, 203), (141,212,208), (87, 208, 203))