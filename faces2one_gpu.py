'''
Face Swaping Demo  
Used for generating a new video, given the source video (e.g. avatar actor) and reference video (real actor).

On basis of Yao Feng's work on texture changing, some improvements were made to inhance the performance.
In condition GPU is on, applying rendering in C++, and without much remarkable time consuming method in cv, 
the process speed on single frame varies, depending mainly on face detection (~70%, dlib utilized currently), 
cropping (~15%), filtering (~5%), etc. 1.0 FPS on average, with GeForce GTX 1060.

When face dection fails, it may due to environment light, head orientation, face occlusion or camera distance.
In failing cases, try turning on the exposure adjustment. It would take extra ~400ms on every frame though. 
If the exposure doesn't settle it down, it should go to the worst situation, needing for further polishing.

Still great potential for acceleration.

Usage:
python videos2one_gpu.py -s videos/source_video.avi -r videos/reference_video.avi -o videos/new_video.avi

Author: Chao
Build: 2019-10-18

Update:
2019-10-22
Add "softmax" smooth in API. Filter the vertices data. Set WINDOW_SIZE to 1 to unable the smooth function.
Extra ~1.9s per frame.

2019-11-6
Modify the consistence of 3 channels of new_mask1. Merge the images by mask_sum.

2019-11-7
Compatible with smooth in "storyboard"(different scenes) by setting the frame numbers of which the scene changes. 

2019-11-12
Improve the texture deformation from different shapes of mouth.
'''

import os
import argparse
import cv2
import numpy as np

from time import time
from skimage import data, exposure, img_as_float
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from scipy import signal

from api import PRN
#from utils.render import render_texture
from utils.rotate_vertices import frontalize
from mesh.render import render_colors as render_texture

OUTPUT_FPS = 30
FOURCC = cv2.VideoWriter_fourcc(*'MPEG')

WINDOW_SIZE = 3
IS_SMOOTHEN = False
STEEP = 0.94
v_list = []


def face_exchanging(prn, src, ref, h, w, prev_s, prev_r, i):
    # 3d reconstruction
    prev_s = src
    pos, prev_s, s = prn.process(src, prev_s, 's')
    if s == 0:
        return None, prev_s, prev_r, s
    vertices = prn.get_vertices(pos)

    # smooth vertices
    if WINDOW_SIZE > 1:
        if  i == 1:
            global v_list
            v_list.append(vertices)
        if len(v_list) <= WINDOW_SIZE:           
            v_list = np.concatenate(([vertices], v_list), axis=0)
        else:
            v_list[-1] = vertices                      
            v_list = np.roll(v_list, 1, axis=0)                      
            vertices = prn.softmax_smooth2(v_list, STEEP)
            v_list[0] = vertices
    else:
        if IS_SMOOTHEN:
            v_list[-1] = vertices                      
            v_list = np.roll(v_list, 1, axis=0) 
                                
    image = src/255.
    #texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

    # texture extracting
    uv_face_sp = cv2.imread('Data/uv-data/uv_face_new.png') 
    uv_face_sp = cv2.cvtColor(uv_face_sp, cv2.COLOR_BGR2GRAY)
    uv_face = cv2.imread('Data/uv-data/uv_face.png')
    uv_face = cv2.cvtColor(uv_face, cv2.COLOR_BGR2GRAY)
    #mask = (abs(cv2.cvtColor(cv2.imread('Data/uv-data/uv_face_mask.png'), cv2.COLOR_BGR2GRAY)) > 0).astype(np.float32)
    mask = (abs(uv_face_sp - uv_face) > 0).astype(np.float32)

    ref_pos, prev_r, _ = prn.process(ref, prev_r, 'r')
    ref_image = ref/255.
    ref_texture = cv2.remap(ref_image, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    new_texture = ref_texture*mask[:,:,np.newaxis]
    #cv2.imwrite("videos/frames_select/" + str(i) + "_newtex.jpg", 255*new_texture)

    # remaping (abandoned now, use new mask instead.)
    #vis_colors = np.ones((vertices.shape[0], 1))
    #face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c = 1)
    #face_mask = render_texture(vertices, prn.triangles, vis_colors, h, w, c = 1) # crender
    #face_mask = np.squeeze(face_mask > 0).astype(np.float32)
  
    new_colors = prn.get_colors_from_texture(new_texture)
    #new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c = 3)
    '''
    # RETARGET MODE 1
    vertices_ref = prn.get_vertices(ref_pos)
    vertices_ref_front = frontalize(vertices_ref)    
    vertices_src_front = frontalize(vertices)

    vertices_src_front_homo = np.hstack((vertices_src_front, np.ones([vertices_src_front.shape[0],1])))
    P_inv = np.linalg.lstsq(vertices_src_front_homo, vertices)[0].T

    vertices_src_front[:,:] = vertices_ref_front[:,:]
    vertices_src_front_homo = np.hstack((vertices_src_front, np.ones([vertices_src_front.shape[0],1])))
    vertices = vertices_src_front_homo.dot(P_inv.T)
    '''
    # RETARGET MODE 2
    vertices_ref = prn.get_vertices(ref_pos)
    vertices_ref_homo = np.hstack((vertices_ref, np.ones([vertices_ref.shape[0], 1])))
    P_inv = np.linalg.lstsq(vertices_ref_homo, vertices)[0].T
    vertices = vertices_ref_homo.dot(P_inv.T)

    new_image = render_texture(vertices, prn.triangles, new_colors, h, w, c = 3) # crender
    #cv2.imwrite("videos/frames_select/" + str(i) + "_newimage.jpg", 255*new_image)
    new_mask = np.squeeze(new_image > 0).astype(np.float32)

    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    #new_mask1 = signal.convolve2d(new_mask, kernel, mode='same',boundary='symm',fillvalue=0)
    new_mask1 = cv2.filter2D(new_mask, -1, kernel)
    new_mask1 = np.squeeze(new_mask1 > 8).astype(np.float32)
    new_mask1 = cv2.filter2D(new_mask1, -1, kernel)
    new_mask1 = np.squeeze(new_mask1 > 8).astype(np.float32)

    mask_sum = new_mask1[:,:,0] + new_mask1[:,:,1] + new_mask1[:,:,2]
    mask_sum = np.squeeze(mask_sum>0).astype(np.float32)    

    new_image = image*(1 - mask_sum[:,:,np.newaxis]) + new_image*mask_sum[:,:,np.newaxis]
    #new_image = image*(1 - new_mask1) + new_image*new_mask1
   
    # blending
    vis_ind = np.argwhere(mask_sum>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))

    output = cv2.seamlessClone((new_image*255).astype(np.uint8), (image*255).astype(np.uint8), (mask_sum*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    #output = cv2.seamlessClone((new_image*255).astype(np.uint8), (new_image_for_clone*255).astype(np.uint8), (new_mask1*255).astype(np.uint8), center, cv2.NORMAL_CLONE)
    #cv2.imwrite("videos/frames_select/output_" + str(i) + ".jpg", output)
    
    return output, prev_s, prev_r, 1
    

def main(args):

    # deploy GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    prn = PRN(is_dlib = True)    
    #print("source video: ", args.src, " reference: ", args.ref)

    # sampling from source video
    cap_s = cv2.VideoCapture(args.src)
    ret_s, frame_s = cap_s.read()
    if ret_s:
        [h, w, _] = frame_s.shape

    # sampling from reference video
    cap_r = cv2.VideoCapture(args.ref)
    ret_r, frame_r = cap_r.read()

    videoWriter = cv2.VideoWriter(args.output, FOURCC, OUTPUT_FPS, (w, h))
    
    #frame_test = cv2.imread("./test.jpg")
    frame_no = 0
    prev_valid_src = frame_s
    prev_valid_ref = frame_r

    if WINDOW_SIZE > 1:
        global IS_SMOOTHEN
        IS_SMOOTHEN = True

    while (ret_s and ret_r):
        frame_no += 1
        '''
        if frame_no in [397, 398, 399, 481, 482, 483, 579, 580, 581]:
            global WINDOW_SIZE
            WINDOW_SIZE = 1
            print('change window_size to:', WINDOW_SIZE)
        if frame_no in [400, 484, 582]:
            global WINDOW_SIZE
            WINDOW_SIZE = 3
            print('change window_size to:', WINDOW_SIZE)
        '''
        ret_s, frame_s = cap_s.read()
        ret_r, frame_r = cap_r.read()

        #frame_r = frame_test
        frame_s = exposure.adjust_gamma(frame_s, 0.8).astype(np.uint8)
        #frame_r = exposure.adjust_gamma(frame_r, 1.3).astype(np.uint8)        
        print(frame_no, "processing")

        st = time()
        # main body
        merged_frame, prev_valid_src, prev_valid_ref, s = face_exchanging(prn, frame_s, frame_r, h, w, prev_valid_src, prev_valid_ref, frame_no)
        print("		", time() - st, "seconds passed.")

        if s == 0:
            print("skip one frame.")

            frame_s = exposure.adjust_gamma(frame_s, 1.5)
            videoWriter.write(frame_s)
            #cv2.imwrite("./videos/detect_fail/"+str(frame_no)+".jpg", frame_s)
            continue
        merged_frame = exposure.adjust_gamma(merged_frame, 1.5)
        videoWriter.write(merged_frame)

    videoWriter.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face exChanging by two videos')

    parser.add_argument('-s', '--src', type=str,
                                   help='specify the directory of source video.')
    parser.add_argument('-r', '--ref', type=str,
                                   help='specify the directory of reference video.')
    parser.add_argument('-o', '--output', default='./videos/output_gpu.avi' ,type=str,
                                   help='specify the directory of output video.')

    main(parser.parse_args())
