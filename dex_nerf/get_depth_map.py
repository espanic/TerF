import os, sys
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import imageio
import json
import random
import time
import pprint

import matplotlib.pyplot as plt

import run_nerf

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data


basedir = './logs'
expname = 'flask_test3'

config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())
parser = run_nerf.config_parser()

args = parser.parse_args('--config {} --ft_path {}'.format(config, os.path.join(basedir, expname, 'model_390000.npy')))
print('loaded args')

# imgs, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res)
# H, W, focal = hwf

H, W, focal = 540, 960, 679.25


H = int(H)
W = int(W)
hwf = [H, W, focal]

# images = imgs.astype(np.float32)
# poses = poses.astype(np.float32)

near = 0
far = 1


trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w
camera_angle = 1.2303659303290422





# Create nerf model
_, render_kwargs_test, start, grad_vars, models = run_nerf.create_nerf(args)

bds_dict = {
    'near' : tf.cast(near, tf.float32),
    'far' : tf.cast(far, tf.float32),
}
render_kwargs_test.update(bds_dict)

print('Render kwargs:')
pprint.pprint(render_kwargs_test)


down = 4
render_kwargs_fast = {k : render_kwargs_test[k] for k in render_kwargs_test}
render_kwargs_fast['N_importance'] = 64
render_kwargs_fast['dex'] = True
render_kwargs_fast['sigma'] = 20


render_poses = tf.stack([pose_spherical(angle, -50.0, 0.5) for angle in np.linspace(0,180,40+1)[:-1]],0)

c2w = np.eye(4)[:3,:4].astype(np.float32) # identity pose matrix
test = run_nerf.render(H//down, W//down, focal/down, 1024*32, c2w=render_poses[5, :3, :4],**render_kwargs_fast)
img = ( np.clip(test[0],0,1))
plt.figure()
plt.subplot(1, 2, 1)
plt.title('RGB Map')
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(-test[1], cmap = 'gray')
plt.title('Depth Map')
plt.show()
