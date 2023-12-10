import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as cl_random
import pyopencl.clmath as cl_math

# ======== GLOBAL PARAMS =========

N_PARTICLES = 80000
TIME_STEP = 0.00001 # yr
CLAMP_VEL = 120.
PRESSURE_SCALE = 0.0001
IMG_X = IMG_Y = 2000

G = 1 # kpc^3 yr^-2 M_sun^-1
m = 1 # solar mass

box_size = 3
initial_v = 0 # kpc/yr

# ======== LOAD OPENCL ===============

def render(string):
    import re
    for kw in set(re.findall(r'{{(.[^\}]*)}}', string)):
        string = string.replace('{{' + kw + '}}', str(eval(kw)))
    return string

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

rendered_code = render(open('gravity.cl').read())
with open('gravity_rendered.cl', 'w') as f:
    f.write(rendered_code)
prg = cl.Program(ctx, rendered_code).build()

# ============= INIT MEMORY ========================

xx = cl_array.empty(queue, shape=N_PARTICLES, dtype=np.double)
xy = cl_array.empty(queue, shape=N_PARTICLES, dtype=np.double)
a = cl_array.empty(queue, shape=N_PARTICLES, dtype=np.double)

cl_random.fill_rand(xx, queue, a=0, b=box_size)
cl_random.fill_rand(xy, queue, a=0, b=box_size)

if initial_v:
    vrand = cl_array.empty(queue, shape=N_PARTICLES, dtype=np.double)
    cl_random.fill_rand(vrand, queue, a=0, b=initial_v)
    vtheta = cl_math.atan2((xy - box_size / 2) ,( xx - box_size / 2), queue=queue)
    vx = -vrand*(0.9*cl_math.sin(vtheta)+0.1*cl_math.cos(vtheta))
    vy = vrand*(0.9*cl_math.cos(vtheta)+0.1*cl_math.sin(vtheta))
    print(vtheta[0], vx[0], vy[0])
else:
    vx = cl_array.empty(queue, shape=N_PARTICLES, dtype=np.double)
    vy = cl_array.zeros(queue, shape=N_PARTICLES, dtype=np.double)

img_buffer_R = cl_array.zeros(queue, shape=(IMG_X, IMG_Y), dtype=np.uint8)
img_buffer_G = cl_array.zeros(queue, shape=(IMG_X, IMG_Y), dtype=np.uint8)
img_buffer_B = cl_array.zeros(queue, shape=(IMG_X, IMG_Y), dtype=np.uint8)
img_buffer_local_R = np.empty(shape=(IMG_X, IMG_Y), dtype=np.uint8)
img_buffer_local_G = np.empty(shape=(IMG_X, IMG_Y), dtype=np.uint8)
img_buffer_local_B = np.empty(shape=(IMG_X, IMG_Y), dtype=np.uint8)

#prg.to_image(queue, (N_PARTICLES,), None, x.data, img_buffer.data).wait()

# ============ DISPLAY ===============

#import matplotlib.pyplot as plot
#from PIL import Image
import pygame
from pygame.locals import *

pygame.display.init()
screen = pygame.display.set_mode((IMG_X, IMG_Y))
pygame.font.init()
FPS = 60
clock = pygame.time.Clock()
pygame.display.set_caption('Galaxy')

img_surf = pygame.surface.Surface((IMG_X, IMG_Y), depth=24)

SUBSTEPS = 1

std = 2
x, y = np.meshgrid(np.linspace(-3, 3, 7), np.linspace(-1, 1, 7))
gaussian_filter = np.exp(- (x/std)**2 - (y/std)**2)

#from scipy.ndimage import convolve

#import cv2
#video = cv2.VideoCapture(0)
#result = cv2.VideoWriter('video.mp4',
                         #cv2.VideoWriter_fourcc(*'mp4v'),
                         #30, (IMG_X, IMG_Y))

def main():
    global a
    while 1:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

        img_buffer_R.fill(0)
        img_buffer_G.fill(0)
        img_buffer_B.fill(0)

        for _ in range(SUBSTEPS):
            prg.step_gravity(queue, (N_PARTICLES,), None, xx.data, xy.data, vx.data, vy.data, a.data).wait()
        #print(xx[0], vx[0])
        a /= cl_array.max(a)
        prg.to_image(queue, (N_PARTICLES,), None, xx.data, xy.data,a.data, img_buffer_R.data, img_buffer_G.data, img_buffer_B.data).wait()
        img_buffer_R.get(ary=img_buffer_local_R)
        img_buffer_G.get(ary=img_buffer_local_G)
        img_buffer_B.get(ary=img_buffer_local_B)
        #print('hello')

        #print(gaussian_filter.shape)
        #print(img_buffer.shape)
        #convolve(img_buffer_local_B, gaussian_filter, output=img_buffer_local_B, mode='constant')
        #convolve(img_buffer_local_G, gaussian_filter, output=img_buffer_local_G, mode='constant')
        #print(img_buffer_local.shape)

        frame = np.transpose(np.stack([img_buffer_local_R,img_buffer_local_G,img_buffer_local_B]), (1,2,0))
        #cv2.imshow('stuff', frame)
        #result.write(frame)

        pygame.surfarray.blit_array(img_surf, frame)
        screen.blit(img_surf, (0,0))
        pygame.display.update()
        pygame.display.flip()
        clock.tick(FPS)
        print(clock.get_fps())


main()