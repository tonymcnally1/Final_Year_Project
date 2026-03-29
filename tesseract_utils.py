import pytesseract as pts
import skimage as ski
import numpy as np
import cv2

def update_img(thres,img):
    newimg=np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= thres:
                newimg[i][j]=1
            else:
                newimg[i][j]=0
    newimg=newimg.astype(np.uint8)
    return newimg

def mode(array):
    vals,counts=np.unique(array, return_counts=True)
    index=np.argmax(counts)
    return vals[index]


def swap_binarisation(binImg):
    m=mode(binImg)
    if m ==0:
        binImg=swap_ones_zeros(binImg)

    return binImg

def swap_ones_zeros(img):
    newimg=np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 0:
                newimg[i][j]=1
            else:
                newimg[i][j]=0
    newimg=newimg.astype(np.uint8)
    return newimg

def big_image_infr(xyxy,bwimg):
    a=np.ones(bwimg.shape,dtype=np.uint8)
    for i in range(xyxy.shape[0]):
        xtl,ytl,xbr,ybr= xyxy[i,0],xyxy[i,1],xyxy[i,2],xyxy[i,3]
        pict=bwimg[ytl:ybr,xtl:xbr]
        thres=ski.filters.threshold_otsu(pict)
        binImg=update_img(thres,pict)
        binImg=swap_binarisation(binImg)
        a[ytl:ybr,xtl:xbr]=binImg
    return a

def  tesseract_on_image(frame,resultsOBJ):
    sep=" "
    img_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).squeeze()
    xyxy=resultsOBJ.boxes.xyxy.numpy().astype(int)
    a=big_image_infr(xyxy,img_gray)
    binstring=sep.join(pts.image_to_string(a).split("\n"))
    return binstring
