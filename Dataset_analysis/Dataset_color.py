

import cv2
import os
import glob
import numpy as np
import shutil


def get_all_image_names(img_dir):
    all_files = []
    for root, dirs, files in os.walk(img_dir):
        file_names = [root + "/" + dir for dir in files]
        all_files += file_names
    return (all_files)


def get_all_image_names(img_dir):
    all_files = []
    for root, dirs, files in os.walk(img_dir):
        file_names = [root + "/" + dir for dir in files]
        all_files.append(file_names)
    return (all_files)



def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

def RGB_to_OPP(imag):
    R=imag[:,0]/255
    G=imag[:,1]/255
    B=imag[:,2]/255

    O1=(R+G+B-1.5)/1.5
    O2=(R-G)
    O3=(R+G-2*B)/2
    return [O1,O2,O3]

def OPP_hue(opp_im):
    hue=np.arctan2(opp_im[2],opp_im[1])

    intensity=np.sqrt(opp_im[2]*opp_im[2] + opp_im[1]*opp_im[1])
    return hue.flatten(), intensity.flatten()

def hue_OPP(hue_im):
    O2=np.cos(hue_im)
    O3=np.sin(hue_im)

    return O2,O3

def OPP_to_RGB(O2,O3):
    G=0.5+O3/3-O2/2
    R=O2+G
    B=1.5-R-G

    return [R,G,B]



def color_bars(rangee):

    hue=np.arange(0,2*np.pi,np.pi*2/rangee)
    O2,O3=hue_OPP(hue)
    rgb=OPP_to_RGB(O2,O3)
    RGB=[(rgb[0][x].clip(min=0).clip(max=1),rgb[1][x].clip(min=0).clip(max=1),rgb[2][x].clip(min=0).clip(max=1)) for x in range(rangee)]

    return(RGB)







if __name__ == '__main__':


    img_dir = 'D:/Dataset/modifyed/0'  # Directory containing all images. The program acces all the files insed this folder recursively



    image_names=get_all_image_names(img_dir)

    chunk_list= image_names[0]

    sum_hist = np.zeros(100)
    sum_weight = np.zeros(100)

    for j,image_batch in enumerate(image_names):

        print(len(image_batch))

        for i ,image_name in enumerate(image_batch):







            # print(i)
            img = cv2.imread(image_name)[:,:,::-1]
            img = cv2.resize(img,(100,100))
            img= np.reshape(img,(-1,img.shape[2]))

            opp=RGB_to_OPP(img) #transform the image into oponent colorspace
            hue,weigh=OPP_hue(opp) #calculate the hue
            hue=np.array([x if x > 0 else (2*np.pi+x) for x in hue])




            positions=[x for x in range(len(hue)) if  weigh[x]< 0.1] # threshold for low intensity values
            positions2=[x for x in range(len(hue)) if opp[0][x]> 0.95 or opp[0][x]< -0.95] # threshold for very white or black values
            positions=positions + list(set(positions2) - set(positions))
            good_pos=range(len(hue))
            thresholded= hue[list(set(good_pos)-set(positions))]




            weighted_histo=np.histogram(thresholded,100,[0,2*np.pi])

            sum_weight+=weighted_histo[0]/len(image_batch)

            print(str(i)+" out of "+str(len(image_names[0])))
    np.save("modifyed_0",sum_weight)





