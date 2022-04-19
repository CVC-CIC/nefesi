import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colortrans

from skimage.transform import  resize
from skimage import io
import random

def RGB_to_OPP(imag):
    R=imag[:,0]
    G=imag[:,1]
    B=imag[:,2]

    O1=(R+G+B-1.5)/1.5
    O2=(R-G)
    O3=(R+G-2*B)/2

    return np.einsum('kl->lk', np.array([O1,O2,O3]))

def OPP_hue(opp_im):
    hue=np.arctan2(opp_im[:,2],opp_im[:,1])

    intensity=np.sqrt(opp_im[:,2]*opp_im[:,2] + opp_im[:,1]*opp_im[:,1])
    return hue.flatten(), intensity.flatten()


def main():
    filename3 = 'D:/Dataset/modifyed/1/a1016-050716_115658__I2E4159 (1).tif'

    rgb3 = io.imread(filename3)
    rgb3= resize(rgb3,(100,100))
    rgb3 =  rgb3.reshape((100*100,3))
    lab3 = RGB_to_OPP(rgb3)
    hue3=OPP_hue(lab3)
    hue3=colortrans.rgb_to_hsv(rgb3)


    filename4 = 'D:/Dataset/modifyed/2/a1016-050716_115658__I2E4159 (2).tif'

    rgb4 = io.imread(filename4)
    rgb4 = resize(rgb4, (100, 100))
    rgb4 = rgb4.reshape((100 * 100, 3))
    lab4 = RGB_to_OPP(rgb4)
    hue4 = OPP_hue(lab4)
    hue4 = colortrans.rgb_to_hsv(rgb4)


    filename5 = 'D:/Dataset/modifyed/3/a1016-050716_115658__I2E4159 (3).tif'

    rgb5 = io.imread(filename5)
    rgb5 = resize(rgb5, (100, 100))
    rgb5 = rgb5.reshape((100 * 100, 3))
    lab5 = RGB_to_OPP(rgb5)
    hue5 = OPP_hue(lab5)
    hue5 = colortrans.rgb_to_hsv(rgb5)





    filename2 = 'D:/Dataset/modifyed/0/a1016-050716_115658__I2E4159.tif'

    rgb2 = io.imread(filename2)
    rgb2 = resize(rgb2, (100, 100))
    rgb2 = rgb2.reshape((100 * 100, 3))
    lab2 = RGB_to_OPP(rgb2)
    hue2 = OPP_hue(lab2)
    hue2 = colortrans.rgb_to_hsv(rgb2)


    filename ='D:/Dataset/OG/a1016-050716_115658__I2E4159.dng'
    rgb = io.imread(filename)
    rgb = resize(rgb, (100, 100))
    og=rgb
    rgb = rgb.reshape((100 * 100, 3))

    lab = RGB_to_OPP(rgb)
    hue = OPP_hue(lab)
    hue = colortrans.rgb_to_hsv(rgb)




    # plt.subplot(2, 1, 1)
    rgb = rgb.reshape((100*100,3))
    rand = random.sample(range(1000), 1000)
    for j in range(len(hue)):

        plt.subplot(2,2,1)
        if(hue2[j][1]>0.2 and hue2[j][2]>0.5 ):
            plt.plot(hue[j][1],hue2[j][1],"o", color= rgb[j])

        plt.title('Expert 1')
        plt.subplot(2, 2, 2)

        plt.title('Expert 2')
        if (hue3[j][1] > 0.2 and hue3[j][2]>0.5 ):
            plt.plot(hue[j][1],hue3[j][1],"o", color= rgb[j])

        plt.subplot(2, 2, 3)

        plt.title('Expert 3')
        if (hue4[j][1] > 0.2 and hue4[j][2]>0.5 ):
            plt.plot(hue[j][1],hue4[j][1],"o", color= rgb[j])

    plt.subplot(2, 2, 4)

    plt.title('Original Image')
    plt.imshow(og)

    plt.subplot(2, 2, 1)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.subplot(2, 2, 2)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.subplot(2, 2, 3)
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.show()











if __name__ == '__main__':
    main()
