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
from mesh.render import render_colors as render_texture

OUTPUT_FPS = 30
FOURCC = cv2.VideoWriter_fourcc(*'MPEG')

WINDOW_SIZE = 3
STEEP = 0.95
v_list = []

def face_exchanging(prn, src, ref, h, w, prev_s, prev_r, i):
    # 3d reconstruction
    prev_s = src
    pos, prev_s, s = prn.process(src, prev_s)
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
    
    image = src/255.
    texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

    # texture extracting
    mask = cv2.imread('Data/uv-data/new_mask.png').astype(np.float32)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    ref_pos, prev_r, _ = prn.process(ref, prev_r)
    ref_image = ref/255.
    ref_texture = cv2.remap(ref_image, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    new_texture = ref_texture*mask[:,:,np.newaxis]

    # remaping (abandoned now, use new mask instead.)
    #vis_colors = np.ones((vertices.shape[0], 1))
    #face_mask = render_texture(vertices, prn.triangles, vis_colors, h, w, c = 1)
    #face_mask = np.squeeze(face_mask > 0).astype(np.float32)
  
    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices, prn.triangles, new_colors, h, w, c = 3)
    new_mask = np.squeeze(new_image > 0).astype(np.float32)

    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    new_mask1 = cv2.filter2D(new_mask, -1, kernel)
    new_mask1 = np.squeeze(new_mask1 > 8).astype(np.float32)
    new_mask1 = cv2.filter2D(new_mask1, -1, kernel)
    new_mask1 = np.squeeze(new_mask1 > 8).astype(np.float32)

    new_image = image*(1 - new_mask1) + new_image*new_mask1

    # blending
    vis_ind = np.argwhere(new_mask1>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((new_image*255).astype(np.uint8), (image*255).astype(np.uint8), (new_mask1*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    return output, prev_s, prev_r, 1
    

def main(args):

    # deploy GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    prn = PRN(is_dlib = True)    
    print("source video: ", args.src, " reference: ", args.ref)

    # sampling from source video
    cap_s = cv2.VideoCapture(args.src)
    ret_s, frame_s = cap_s.read()
    if ret_s:
        [h, w, _] = frame_s.shape

    # sampling from reference video
    cap_r = cv2.VideoCapture(args.ref)
    ret_r, frame_r = cap_r.read()

    videoWriter = cv2.VideoWriter(args.output, FOURCC, OUTPUT_FPS, (w, h))
    
    frame_no = 0
    prev_valid_src = frame_s
    prev_valid_ref = frame_r
    while (ret_s and ret_r):
        frame_no += 1        

        ret_s, frame_s = cap_s.read()
        ret_r, frame_r = cap_r.read()
        #frame_s = exposure.adjust_gamma(frame_s, 0.67).astype(np.uint8)
        #frame_r = exposure.adjust_gamma(frame_r, 0.5).astype(np.uint8)
        
        print(frame_no, "processing")

        # main body
        merged_frame, prev_valid_src, prev_valid_ref, s = face_exchanging(prn, frame_s, frame_r, h, w, prev_valid_src, prev_valid_ref, frame_no)

        if s == 0:
            print("skip one frame.")
            #frame_s = exposure.adjust_gamma(frame_s, 1.5)
            videoWriter.write(frame_s)
            cv2.imwrite("./videos/detect_fail/"+str(frame_no)+".jpg", frame_s)
            continue
            
        #merged_frame = exposure.adjust_gamma(merged_frame, 1.5)
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
