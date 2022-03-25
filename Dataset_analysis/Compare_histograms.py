import cv2
import os
import glob
import numpy as np
import colorsys
import PIL


import matplotlib.pyplot as plt



def hue_OPP(hue_im):
    O2=np.cos(hue_im)
    O3=np.sin(hue_im)

    return O2,O3

def OPP_to_RGB(O2,O3):
    G=0.5+O3/3-O2/2
    R=O2+G
    B=1.5-R-G

    return [R,G,B]


def color_bars(range):

    hue=np.arange(0,1,1/range)
    print(hue)
    rgb=[ colorsys.hsv_to_rgb(h,1,1) for h in hue]
    return(rgb)


def color_bars_OPP(rangee):

    hue=np.arange(0,2*np.pi,np.pi*2/rangee)
    O2,O3=hue_OPP(hue)
    rgb=OPP_to_RGB(O2,O3)
    RGB=[(rgb[0][x].clip(min=0).clip(max=1),rgb[1][x].clip(min=0).clip(max=1),rgb[2][x].clip(min=0).clip(max=1)) for x in range(rangee)]

    return(RGB)


def main():
    target=np.load("modifyed.npy")
    original=np.load("OG.npy")




    color_bar=color_bars_OPP(100)

    X=np.arange(100)

    plt.subplot(2,1,1)
    axes=plt.gca()
    # axes.set_ylim([0, 350000])

    plt.bar(X, target/max(target), color=color_bar)
    plt.title('modifyed')
    plt.subplot(2, 1, 2)
    axes=plt.gca()

    # axes.set_ylim([0, 350000])
    plt.bar(X, original/max(original), color=color_bar)
    plt.title('Original')



    plt.show()



if __name__ == '__main__':
    main()