import re
import cv2
import numpy as np
import random

#return a list of classes and array of rgb values for a given class
def read_classes(arg):
    pattern=r"^.*\.txt"
    if isinstance(arg,str):
        if re.match(pattern,arg):
            return read_classes_txt(arg)
    elif isinstance(arg, list):
        return read_classes_list(arg)

def read_classes_txt(path_to_classes_txt):
    with open(path_to_classes_txt,"r") as f:
        lines=f.readlines()
    f.close()
    classes=[]
    colors=np.zeros((0,3))
    for i in range(len(lines)):
        classes.append(lines[i].split("\n")[0])
        r,g,b= (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        colors=np.vstack((colors,(r,g,b)))
    return classes,colors

def read_classes_list(c_list):
    colors=np.zeros((0,3))
    for i in range(len(c_list)):
        r,g,b= (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        colors=np.vstack((colors,(r,g,b)))
    return colors

def get_h_w(img):
    h=img.shape[0]
    w=img.shape[1]
    return h,w

def unnormalise_img(rh, rw, x,y,w,h):
    ux=x*rw
    uw=w*rw
    uy=y*rh
    uh=h*rh
    return ux,uy,uw,uh

def center_to_vertex(x,y,w,h):
    dx= int(w/2)
    dy=int(h/2)
    v1=((x-dx),(y-dy))
    v2=((x+dx),(y+dy))
    return v1, v2

def proper_array(img, arr):
    arr[1],arr[2],arr[3],arr[4]=unnormalise_img(img,arr[1],arr[2],arr[3],arr[4])
    (arr[1],arr[2]), (arr[3],arr[4]) = center_to_vertex(arr[1],arr[2],arr[3],arr[4])
    return arr

#returns a 2d array with each file line as a row and entry as a column
#array is of format cat_id x_top_left y_top_left x_bottom_right y_bottom_right
def get_file_details(file_path,img):
    with open(file_path,"r") as f:
        lines=f.readlines()
        arr=np.zeros((0,5))
        f.close()
    for i in range(len(lines)):
        spl=lines[i].strip().split(" ")
        splA=np.asarray(spl)
        splA=splA.astype(float)
        splA=proper_array(img, splA)
        arr=np.vstack([arr,splA])
    return arr

#returns array appended with rgb data
#format: cat_id x_top_left y_top_left x_bottom_right y_bottom_right r g b
def append_arr_rgb(arr,colors):
    rgb_arr=np.zeros((0,3))
    for i in range(arr.shape[0]):
        tmp_arr=np.zeros((0,3))
        idx=int(arr[i][0])
        color=colors[idx]
        rgb_arr=np.vstack([rgb_arr,color])
    new_arr=np.hstack((arr,rgb_arr))
    return new_arr

#color is 3 width tuple in the format (r,g,b)
def plot_images(two_d_array, img, classes,img_title="image"):
    for i in range(two_d_array.shape[0]):
        color=(two_d_array[i][5],two_d_array[i][6],two_d_array[i][7])
        p1, p2 = (int(two_d_array[i][1]),int(two_d_array[i][2])),(int(two_d_array[i][3]),int(two_d_array[i][4]))
        img=cv2.rectangle(img,p1,p2,color)
        img=cv2.putText(img,classes[int(two_d_array[i][0])],(p1[0],p1[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color)
    newimg=img
    cv2.imshow(img_title,img)
    return newimg

'''
WH - GIVEN TOP AND BOTTOM LEFT CORNER
'''
def calc_W_H(x1,y1,x2,y2):
    x1=int(x1)
    x2=int(x2)
    y1=int(y1)
    y2=int(y2)
    w=x2-x1
    h=y2-y1
    return w,h

def calc_center(x1,y1,x2,y2):
    x1=int(x1)
    x2=int(x2)
    y1=int(y1)
    y2=int(y2)
    cx=(x2-x1)/2+x1
    cy=(y2-y1)/2+y1
    return cx,cy

def normalise_center(cx,cy,img):
    w=img.shape[1]
    h=img.shape[0]
    ncx=cx/w
    ncy=cy/h
    return ncx,ncy

def normalise_wh(w,h,img):
    iw=img.shape[1]
    ih=img.shape[0]
    nw=w/iw
    nh=h/ih
    return nw,nh

