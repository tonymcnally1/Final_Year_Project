import numpy as np
import math
import plotting_functions as pf
import cv2
import pdb
import random
import os
#pdb.set_trace()

def calculate_distance(c1,c2):
    (cx1,cy1)=c1
    (cx2,cy2)=c2
    return math.sqrt(math.pow((cx1-cx2),2)+math.pow((cy1-cy2),2))

'''
Function to match Ground Truth bounding box with Inferred bounding box by nearest distance.
Assumes ncx,ncw,nw,nh format.
'''
def match_bboxes(inf_arr,gt_arr):
    cxyI=inf_arr[:,1:3]
    cxyA=gt_arr[:,1:3]
    match_list=[]
    for i in range(cxyA.shape[0]):
        minD=100
        nearest=inf_arr[0,:]
        cBB=gt_arr[i,:]
        jtag=0
        for j in range(cxyI.shape[0]):
            if inf_arr[j,0]==0:
                dist=calculate_distance((cxyA[i,0],cxyA[i,1]),(cxyI[j,0],cxyI[j,1]))
                if dist<minD:
                    minD=dist
                    nearest=inf_arr[j,:]
                    jtag=j
        match_list.append((cBB,nearest,minD,i,jtag))
    drop_list=[]
    for i in range(len(match_list)):
        for j in range(len(match_list)):
            if (i!=j):
                if np.array_equal(match_list[i][1],match_list[j][1]):
                    if match_list[i][2]>=match_list[j][2]:
                        drop_list.append(i)
                    else:
                        drop_list.append(j)
    drop_list = list(set(drop_list))
    if drop_list!=None:
        for i in range(len(drop_list)):
            idx=drop_list[-i]
            match_list[idx]=[match_list[idx][0],np.zeros((5)),"inf",match_list[idx][3],"-"]
        
    return match_list

def is_overlap(tl1,tl2,br1,br2):
    t1D=calculate_distance(tl1,(0,0))
    t2D=calculate_distance(tl2,(0,0))
    b1D=calculate_distance(br1,(0,0))
    b2D=calculate_distance(br2,(0,0))
    flag=False
    if(t1D<b2D):
        if(t2D<b1D):
            flag=True
    return flag

def overlap_bounds(tl1,tl2,br1,br2):
    tlx=max(tl1[0],tl2[0])
    tly=max(tl1[1],tl2[1])
    brx=min(br1[0],br2[0])
    bry=min(br1[1],br2[1])
    return (tlx,tly),(brx,bry)

'''
input is 2 unnormalised bounding box corner coordinates
a1=[xtl1,ytl1,xbr1,ybr2][xtl2,ytl2,xbr2,ybr2]
'''
def iou(a1,a2):
    area1=(a1[2]-a1[0])*(a1[3]-a1[1])
    area2=(a2[2]-a2[0])*(a2[3]-a2[1])
    if is_overlap((a1[0],a1[1]),(a2[0],a2[1]),(a1[2],a1[3]),(a2[2],a2[3])):
        oLT, oLB=overlap_bounds((a1[0],a1[1]),(a2[0],a2[1]),(a1[2],a1[3]),(a2[2],a2[3]))
        overlap=(oLB[0]-oLT[0])*(oLB[1]-oLT[1])
        union=area1+area2-overlap
        iou=overlap/union
        return iou
    else:
        return 0
          
def file_to_array(path):
    with open(path,'r') as f:
        lines=f.readlines()
        i=0
            
        for i in range(len(lines)):
            lines[i]=lines[i].split("\n")[0]
            lines[i]=lines[i].split(" ")
        arr=np.zeros((len(lines),5))
        for i in range(len(lines)):
            for j in range(5):
                arr[i,j]=float(lines[i][j])    
        return arr

'''
prec_array must be flat array along axis
'''
def generate_recall_array(N,prec_array):
    rARR=np.zeros(prec_array.shape)
    delta=1/N
    cval=delta
    for i in range(rARR.shape[0]):
        rARR[i]=cval
        cval+=delta
    return rARR
        
if __name__=="__main__":
    results_dir="Inference_results\ICDAR_2013_Text\model25thfeb\labels\\"
    values_dir="Modified_Datasets\ICDAR_2013_text_reading\Val\labels\\"
    img_dir="Modified_Datasets\ICDAR_2013_text_reading\Val\images\\"
    targ_img_list=["img_190","img_23","img_80","img_126","img_155"]
    rdL=os.listdir(results_dir)
    vdL=os.listdir(values_dir)
    idL=os.listdir(img_dir)
    for i in range(5):
        r=random.randint(0,len(rdL)-1)
        targ_img=rdL[r].split(".")[0]
        targ_img=targ_img_list[i]
        infr_res_arr=file_to_array(results_dir+targ_img+".txt")
        actual_results=file_to_array(values_dir+targ_img+".txt")
        mLIST=match_bboxes(infr_res_arr,actual_results)
        iou_list=np.array([])
        img=cv2.imread(img_dir+targ_img+".jpg")
        h,w=pf.get_h_w(img)
        for i in range(len(mLIST)):
            a1=mLIST[i][0]
            a2=mLIST[i][1]
            ux1,uy1,uw1,uh1=pf.unnormalise_img(h,w,a1[1],a1[2],a1[3],a1[4])
            v1a1,v2a1=pf.center_to_vertex(ux1,uy1,uw1,uh1)
            a1=[v1a1[0],v1a1[1],v2a1[0],v2a1[1]]
            ux2,uy2,uw2,uh2=pf.unnormalise_img(h,w,a2[1],a2[2],a2[3],a2[4])
            v1a2,v2a2=pf.center_to_vertex(ux2,uy2,uw2,uh2)
            a2=[v1a2[0],v1a2[1],v2a2[0],v2a2[1]]
            IOU=iou(a1,a2)
            iou_list=np.append(iou_list,IOU)
        print("============================================================")
        print(f"{'Target Image: ':>30}{targ_img:<30}")
        print("============================================================")
        print(f"{'GT idx':<8} {'Inf idx':<9} {'IoU':>8}")
        print("------------------------------------------------------------")
        for i in range(len(iou_list)):
            print(f"{mLIST[i][3]:<8} {mLIST[i][4]:<9} {iou_list[i]:>8.4f}")
        print("============================================================")
        
        
        
        
        


        
        
    
    
    

