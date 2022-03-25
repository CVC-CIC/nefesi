import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from skimage.transform import rescale, resize
from skimage import io, color
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
    filename3 = 'D:/Dataset/modifyed/0/a0106-NKIM_120 (1).tif'

    rgb3 = io.imread(filename3)
    rgb3= resize(rgb3,(64,64))
    rgb3 =  rgb3.reshape((64*64,3))
    lab3 = RGB_to_OPP(rgb3)
    hue3=OPP_hue(lab3)

    x3 = [x[1] for x in lab3]
    y3 = [y[2] for y in lab3]

    filename4 = 'D:/Dataset/modifyed/1/a0106-NKIM_120 (2).tif'

    rgb4 = io.imread(filename4)
    rgb4 = resize(rgb4, (64, 64))
    rgb4 = rgb4.reshape((64 * 64, 3))
    lab4 = RGB_to_OPP(rgb4)
    hue4 = OPP_hue(lab4)

    x4 = [x[1] for x in lab4]
    y4 = [y[2] for y in lab4]

    filename5 = 'D:/Dataset/modifyed/2/a0106-NKIM_120 (3).tif'

    rgb5 = io.imread(filename5)
    rgb5 = resize(rgb5, (64, 64))
    rgb5 = rgb5.reshape((64 * 64, 3))
    lab5 = RGB_to_OPP(rgb5)
    hue5 = OPP_hue(lab5)

    x5 = [x[1] for x in lab5]
    y5 = [y[2] for y in lab5]



    filename2 = 'D:/Dataset/modifyed/1/a0106-NKIM_120 (2).tif'

    rgb2 = io.imread(filename2)
    rgb2 = resize(rgb2, (64, 64))
    rgb2 = rgb2.reshape((64 * 64, 3))
    lab2 = RGB_to_OPP(rgb2)
    hue2 = OPP_hue(lab2)

    x2 = [x[1] for x in lab2]
    y2 = [y[2] for y in lab2]

    filename ='D:/Dataset/OG/a0106-NKIM_120.dng'
    rgb = io.imread(filename)
    rgb = resize(rgb, (64, 64))
    rgb = rgb.reshape((64 * 64, 3))

    lab = RGB_to_OPP(rgb)
    hue = OPP_hue(lab)

    x= [x[1] for x in lab]
    y = [y[2] for y in lab]

    dx2 = [x2[i]-x[i] for i in range(len(x))]
    dx3 = [x3[i]-x[i] for i in range(len(x))]
    dy2 =[y2[i]-y[i] for i in range(len(x))]
    dy3 =[ y3[i]-y[i] for i in range(len(x))]

    dx4 = [x4[i] - x[i] for i in range(len(x))]
    dy4 = [y4[i] - y[i] for i in range(len(x))]
    dx5 = [x5[i] - x[i] for i in range(len(x))]
    dy5 = [y5[i] - y[i] for i in range(len(x))]

    # plt.subplot(2, 1, 1)
    rgb = rgb.reshape((64*64,3))
    rand = random.sample(range(1000), 1000)
    for j in range(1000):
        i=rand[j]
        plt.subplot(2,2,1)
        plt.arrow(x[i],y[i],dx2[i],dy2[i],ec= rgb[i],facecolor=rgb[i],width=0.01)
        plt.xlabel('green/red')
        plt.ylabel('blue/yellow')
        plt.title('Expert 1')
        plt.subplot(2, 2, 2)
        plt.xlabel('green/red')
        plt.ylabel('blue/yellow')
        plt.title('Expert 2')
        plt.arrow(x[i], y[i], dx3[i], dy3[i],ec= rgb[i],facecolor=rgb[i],width=0.01)

        plt.subplot(2, 2, 3)
        plt.xlabel('green/red')
        plt.ylabel('blue/yellow')
        plt.title('Expert 3')
        plt.arrow(x[i], y[i], dx4[i], dy4[i], ec=rgb[i], facecolor=rgb[i], width=0.01)

        plt.subplot(2, 2, 4)
        plt.xlabel('green/red')
        plt.ylabel('blue/yellow')
        plt.title('Expert 4')
        plt.arrow(x[i], y[i], dx5[i], dy5[i], ec=rgb[i], facecolor=rgb[i], width=0.01)





    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    # plt.subplot(1, 2, 1)
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])





    plt.show()


if __name__ == '__main__':
    main()
