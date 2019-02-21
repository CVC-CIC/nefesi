# colorNaming.py

from numpy import *
import numpy as np


# *************************
# parameters
# *************************
colors=['Red','Orange','Brown','Yellow','Green','Blue','Purple','Pink','Black','Grey','White']

parameters = np.array([[[  4.24199636e-01,   2.29563220e-01,  -1.17335858e-01,  -4.43169080e-01,   0.00000000e+00,   0.00000000e+00],\
                     [  2.49359126e-01,   6.63190063e-01,   5.18007668e-01,   1.07591884e+00,   0.00000000e+00,   0.00000000e+00],\
                     [ -3.90823298e-02,   3.85324254e-02,   2.36880030e-01,   4.62878817e-01,   0.00000000e+00,   0.00000000e+00],\
                     [ -9.86970116e-01,  -8.51969023e-01,  -7.94921771e-01,  -9.95040273e-01,   0.00000000e+00,   0.00000000e+00],\
                     [  9.00991049e-01,   5.24362727e-01,   9.99376086e-01,   9.39107161e-01,   0.00000000e+00,   0.00000000e+00],\
                     [  1.72499209e+00,   5.00000000e+00,   5.65279390e-01,   7.54842803e-01,   0.00000000e+00,   0.00000000e+00],\
                     [  9.84405931e+00,   6.03260303e+00,   6.81218149e+00,   7.31858270e+00,  -0.00000000e+00,  -0.00000000e+00],\
                     [  5.88939523e+00,   6.45705135e+00,   5.38109004e+00,   6.06365916e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  7.47432124e+00,   7.87254713e+00,   6.98237582e+00,   7.50204671e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  4.05071773e-02,   3.07016320e-01,   3.41770788e-01,   4.28449446e-01,   0.00000000e+00,   0.00000000e+00]],\
                    [[  0.00000000e+00,   0.00000000e+00,  -1.17335858e-01,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   5.18007668e-01,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   7.75874556e-01,   5.75756054e-01,   4.49437961e-01,   4.49316962e-01],\
                     [  0.00000000e+00,   0.00000000e+00,  -5.02014629e-01,  -1.72492737e-01,  -2.76553982e-01,  -3.06421502e-01],\
                     [  0.00000000e+00,   0.00000000e+00,   5.65279390e-01,   7.54842803e-01,   1.99511862e+00,   1.02801414e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   5.17227023e-01,   4.78353741e-01,   8.39910145e-01,   7.90983657e-01],\
                     [ -0.00000000e+00,  -0.00000000e+00,   6.81218149e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  0.00000000e+00,   0.00000000e+00,   5.38109004e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   6.98237582e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   3.41770788e-01,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]],\
                    [[  4.24199636e-01,   2.29563220e-01,  -1.17335858e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  2.49359126e-01,   6.63190063e-01,   5.18007668e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  5.83826210e-01,   7.18827304e-01,   1.06878170e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  2.54038810e-01,   1.19930098e-01,   1.16144803e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  1.72499209e+00,   5.00000000e+00,   5.17227023e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  8.44028374e-01,   6.92404886e-01,   8.44786041e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  9.84405931e+00,   6.03260303e+00,   6.81218149e+00,  -0.00000000e+00,  -0.00000000e+00,  -0.00000000e+00],\
                     [  5.88939523e+00,   6.45705135e+00,   5.38109004e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  7.47432124e+00,   7.87254713e+00,   6.98237582e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],\
                     [  4.05071773e-02,   3.07016320e-01,   3.41770788e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00]],\
                    [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.39830359e+00,   1.29424234e+00,   1.26437482e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   9.90182597e-02,   2.14168287e-01,   2.83461782e-01],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   4.78353741e-01,   8.39910145e-01,   7.90983657e-01],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   7.28741639e-01,   8.56004138e-01,   9.57414544e-01],\
                     [ -0.00000000e+00,  -0.00000000e+00,  -0.00000000e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]],\
                    [[  4.24199636e-01,   2.29563220e-01,  -1.17335858e-01,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  2.49359126e-01,   6.63190063e-01,   5.18007668e-01,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [  1.82483514e+00,   1.69072643e+00,   1.68694113e+00,   1.66981459e+00,   1.78496461e+00,   1.85425811e+00],\
                     [  2.34898971e+00,   2.10250398e+00,   1.90897812e+00,   1.89047978e+00,   1.72030985e+00,   1.74612759e+00],\
                     [  8.44028374e-01,   6.92404886e-01,   8.44786041e-01,   7.28741639e-01,   8.56004138e-01,   9.57414544e-01],\
                     [  1.95218383e+00,   9.58110786e-01,   5.99760697e-01,   6.43832855e-01,   7.36644526e-01,   9.00009936e-01],\
                     [  9.84405931e+00,   6.03260303e+00,   6.81218149e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  5.88939523e+00,   6.45705135e+00,   5.38109004e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  7.47432124e+00,   7.87254713e+00,   6.98237582e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  4.05071773e-02,   3.07016320e-01,   3.41770788e-01,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]],\
                    [[  4.24199636e-01,   2.29563220e-01,  -1.17335858e-01,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  2.49359126e-01,   6.63190063e-01,   5.18007668e-01,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [  3.91978603e+00,   3.67330031e+00,   3.47977445e+00,   3.46127611e+00,   3.29110617e+00,   3.31692392e+00],\
                     [ -2.56831073e+00,  -2.59153227e+00,  -2.58725114e+00,  -2.59660745e+00,  -2.63248600e+00,  -2.60808914e+00],\
                     [  1.95218383e+00,   9.58110786e-01,   5.99760697e-01,   6.43832855e-01,   7.36644526e-01,   9.00009936e-01],\
                     [  1.01425447e+00,   9.16607781e-01,   8.00824361e-01,   7.55213599e-01,   4.74809095e-01,   5.98984911e-01],\
                     [  9.84405931e+00,   6.03260303e+00,   6.81218149e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  5.88939523e+00,   6.45705135e+00,   5.38109004e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  7.47432124e+00,   7.87254713e+00,   6.98237582e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  4.05071773e-02,   3.07016320e-01,   3.41770788e-01,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]],\
                    [[  4.24199636e-01,   2.29563220e-01,  -1.17335858e-01,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  2.49359126e-01,   6.63190063e-01,   5.18007668e-01,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [ -9.97514408e-01,  -1.02073594e+00,  -1.01645481e+00,  -1.02581112e+00,  -1.06168967e+00,  -1.03729281e+00],\
                     [ -1.60987866e+00,  -1.84515592e+00,  -1.96583055e+00,  -2.16285442e+00,  -2.13886579e+00,  -2.13569227e+00],\
                     [  1.01425447e+00,   9.16607781e-01,   8.00824361e-01,   7.55213599e-01,   4.74809095e-01,   5.98984911e-01],\
                     [  9.00991049e-01,   1.10031784e+00,   6.23673475e-01,   5.00000000e+00,   1.73558321e+00,   1.93131386e+00],\
                     [  9.84405931e+00,   6.03260303e+00,   6.81218149e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  5.88939523e+00,   6.45705135e+00,   5.38109004e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  7.47432124e+00,   7.87254713e+00,   6.98237582e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  4.05071773e-02,   3.07016320e-01,   3.41770788e-01,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]],\
                    [[  0.00000000e+00,   2.29563220e-01,  -1.17335858e-01,  -4.43169080e-01,  -5.65458039e-01,  -1.25643027e+00],\
                     [  0.00000000e+00,   6.63190063e-01,   5.18007668e-01,   1.07591884e+00,   1.16327519e+00,   1.81438090e+00],\
                     [  0.00000000e+00,  -2.74359589e-01,  -3.95034220e-01,  -5.92058089e-01,  -5.68069463e-01,  -5.64895945e-01],\
                     [  0.00000000e+00,  -1.53226390e+00,  -1.33391630e+00,  -1.10791751e+00,  -1.12135837e+00,  -1.12147936e+00],\
                     [  0.00000000e+00,   1.10031784e+00,   6.23673475e-01,   5.00000000e+00,   1.73558321e+00,   1.93131386e+00],\
                     [  0.00000000e+00,   5.24362727e-01,   9.99376086e-01,   9.39107161e-01,   1.99511862e+00,   1.02801414e+00],\
                     [ -0.00000000e+00,   6.03260303e+00,   6.81218149e+00,   7.31858270e+00,   1.00000000e+02,   1.00000000e+02],\
                     [  0.00000000e+00,   6.45705135e+00,   5.38109004e+00,   6.06365916e+00,   5.36820773e+00,   6.03989989e+00],\
                     [  0.00000000e+00,   7.87254713e+00,   6.98237582e+00,   7.50204671e+00,   6.90048122e+00,   7.39318578e+00],\
                     [  0.00000000e+00,   3.07016320e-01,   3.41770788e-01,   4.28449446e-01,   4.32023987e-01,  -2.08201846e-02]]])

paramsAchro = np.array([[ 28.28252201,  -0.71423449],\
                    [ 28.28252201,   0.71423449],\
                     [ 79.64930057,  -0.30674052],\
                     [ 79.64930057,   0.30674052]])
thrL = np.array([  0,  31,  42,  51,  66,  76, 150], dtype=np.uint8)


# ***********************    
#    RGB2LAB conversion
# ***********************
def RGB2Lab(Ima):

    # RGB > XYZ transformation matrix (sRGB with D65)
    M = np.vstack(([0.412424, 0.357579, 0.180464],[0.212656, 0.715158, 0.0721856],[0.0193324, 0.119193, 0.950444]))
    Xn = 0.9505; Yn = 1.0000; Zn = 1.0891;

    Ima = Ima/255.0
    S = np.shape(Ima)
    #NF = S[0]; NC = S[1]; NCh = S[2]

    lRGB = np.zeros((3,1))
    XYZ = np.zeros((3,1))
    #ImaLab = zeros((NF, NC, NCh))
    ImaLab = np.zeros_like(Ima,dtype=float)

    #fRGB = vstack((reshape(Ima[:,:,0].T,(1,NF*NC)),reshape(Ima[:,:,1].T,(1,NF*NC)),reshape(Ima[:,:,2].T,(1,NF*NC))))
    fRGB = np.reshape(Ima,(-1,3)).T
    lRGB = (fRGB<=0.04045)*(fRGB/12.92)+(fRGB>0.04045)*(((fRGB+0.055)/1.055)**2.4)
    XYZ = np.dot(M,lRGB)

    f_X2 = (XYZ[0]/Xn > 0.008856)*((XYZ[0]/Xn)**(1.0/3.0))+(XYZ[0]/Xn <= 0.008856)*(7.787*(XYZ[0]/Xn)+(16.0/116.0))
    f_Y2 = (XYZ[1]/Yn > 0.008856)*((XYZ[1]/Yn)**(1.0/3.0))+(XYZ[1]/Yn <= 0.008856)*(7.787*(XYZ[1]/Yn)+(16.0/116.0))
    f_Z2 = (XYZ[2]/Zn > 0.008856)*((XYZ[2]/Zn)**(1.0/3.0))+(XYZ[2]/Zn <= 0.008856)*(7.787*(XYZ[2]/Zn)+(16.0/116.0))

    L2  = (XYZ[1]/Yn > 0.008856)*((116.0*((XYZ[1]/Yn)**(1.0/3.0)))-16.0)+(XYZ[1]/Yn <= 0.008856)*(903.3*(XYZ[1]/Yn))
    a2 = 500.0*(f_X2-f_Y2)
    b2 = 200.0*(f_Y2-f_Z2)

    ImaLab = np.reshape(np.vstack((L2.flatten(), a2.flatten(), b2.flatten())).T,ImaLab.shape)

    #ImaLab[:,:,0] = reshape(L2,(NC,NF)).T
    #ImaLab[:,:,1] = reshape(a2,(NC,NF)).T
    #ImaLab[:,:,2] = reshape(b2,(NC,NF)).T

    return ImaLab




# ***********************    
#    ColorName2rgb
# ***********************
def ColorName2rgb(colorIdx, Names):

    nr = colorIdx.shape[0]
    n = colorIdx.shape[1]*nr
    r = range(np.size(colorIdx))
    colorIdx = colorIdx.flatten()

    Red_idx    = [x for x in r if colorIdx[x]==Names.index('Red')]
    Orange_idx = [x for x in r if colorIdx[x]==Names.index('Orange')]
    Brown_idx  = [x for x in r if colorIdx[x]==Names.index('Brown')]
    Yellow_idx = [x for x in r if colorIdx[x]==Names.index('Yellow')]
    Green_idx  = [x for x in r if colorIdx[x]==Names.index('Green')]
    Blue_idx   = [x for x in r if colorIdx[x]==Names.index('Blue')]
    Purple_idx = [x for x in r if colorIdx[x]==Names.index('Purple')]
    Pink_idx   = [x for x in r if colorIdx[x]==Names.index('Pink')]
    Black_idx  = [x for x in r if colorIdx[x]==Names.index('Black')]
    Grey_idx   = [x for x in r if colorIdx[x]==Names.index('Grey')]
    White_idx  = [x for x in r if colorIdx[x]==Names.index('White')]

    Red = np.zeros((n,1)); Red[Red_idx] = 1;
    Orange = np.zeros((n,1)); Orange[Orange_idx] = 1;
    Brown = np.zeros((n,1)); Brown[Brown_idx] = 1;
    Yellow = np.zeros((n,1)); Yellow[Yellow_idx] = 1;
    Green = np.zeros((n,1)); Green[Green_idx] = 1;
    Blue = np.zeros((n,1)); Blue[Blue_idx] = 1;
    Purple = np.zeros((n,1)); Purple[Purple_idx] = 1;
    Pink = np.zeros((n,1)); Pink[Pink_idx] = 1;
    Black = np.zeros((n,1)); Black[Black_idx] = 1;
    Grey = np.zeros((n,1)); Grey[Grey_idx] = 1;
    White = np.zeros((n,1)); White[White_idx] = 1;

    RGB = np.tile(Red,(1,3))*np.tile([1.0,0.0,0.0],(n,1))    + \
          np.tile(Orange,(1,3))*np.tile([1.0,0.6,0.0],(n,1)) + \
          np.tile(Brown,(1,3))*tile([0.4, 0.2, 0.0],(n,1))   + \
          np.tile(Yellow,(1,3))*tile([1.0,1.0,0.0],(n,1))    + \
          np.tile(Green,(1,3))*tile([0.0,1.0,0.0],(n,1))     + \
          np.tile(Blue,(1,3))*tile([0.0,0.0,1.0],(n,1))      + \
          np.tile(Purple,(1,3))*tile([0.7,0.0,0.7],(n,1))    + \
          np.tile(Pink,(1,3))*tile([0.8,0.6,0.7],(n,1))      + \
          np.tile(Black,(1,3))*tile([0.0,0.0,0.0],(n,1))     + \
          np.tile(Grey,(1,3))*tile([0.5,0.5,0.5],(n,1))      + \
          np.tile(White,(1,3))*tile([1.0,1.0,1.0],(n,1))

    return RGB.reshape((nr,-1,3))


# ***********************    
#    Sigmoid
# ***********************
def Sigmoid(s,t,b):

    y = 1.0/(1.0+np.exp(-np.double(b)*(np.double(s)-np.double(t))))

    return y



# ***********************    
#    TripleSigmoid_E
# ***********************
def TripleSigmoid_E(s,tx,ty,alfa_x,alfa_y,bx,by,be,ex,ey,angle_e):

    sT = np.double(s.T) - np.hstack([tx, ty])
    sR = np.hstack([sT[:,0].reshape((-1,1))*np.cos(alfa_y)+sT[:,1].reshape((-1,1))*np.sin(alfa_y),\
                -sT[:,0].reshape((-1,1))*np.sin(alfa_x)+sT[:,1].reshape((-1,1))*np.cos(alfa_x)])
    sRE= np.hstack([sT[:,0].reshape((-1,1))*np.cos(angle_e)+sT[:,1].reshape((-1,1))*np.sin(angle_e),\
                -sT[:,0].reshape((-1,1))*np.sin(angle_e)+sT[:,1].reshape((-1,1))*np.cos(angle_e)])
    ex = (ex==0.0) + ex
    ey = (ey==0.0) + ey

    y = 1.0/np.hstack([1.0+np.exp(-sR*np.hstack([by, bx])),\
                    1.0+np.exp(-be*(np.sum((sRE/np.hstack([ex, ey]))**2.0,axis=1).reshape((-1,1))-1.0))])

    return np.prod(y,axis=1).reshape((-1,1))




# ***********************    
#    SampleColorNaming
# ***********************
def SampleColorNaming(s):

    if np.size(s)!=3:
        np.error('Error: s must be a 1 x 3 vector [R G B]');

    # Constants
    #colors=['Red','Orange','Brown','Yellow','Green','Blue','Purple','Pink','Black','Grey','White']
    numColors=11                           # Number of colors
    numAchromatics=3                       # Number of achromatic colors
    numChromatics=numColors-numAchromatics # Number of chromatic colors

    # Initializations
    numLevels = np.size(thrL)-1                   # Number of Lightness levels in the model
    CD = np.zeros((1,numColors))                 # Color descriptor to store results

    Lab = RGB2Lab(np.double(np.reshape(s,(1,1,3))))
    L=Lab[:,:,0].flatten();
    a=Lab[:,:,1].flatten();
    b=Lab[:,:,2].flatten();

    # Assignment of the sample to its corresponding level
    m = np.zeros(np.shape(L))
    m[np.where(L==0)[0]] = 1.0                    # Pixels with L=0 assigned to level 1
    k=0
    for k in range(numLevels):
        m = m + np.double(thrL[:,k]<L) * np.double(L<=thrL[:,k+1]) * k

    m = int(np.squeeze(m))

    # Computing membership values to chromatic categories
    for k in range(numChromatics):
        tx=parameters[k,0,m-1]
        ty=parameters[k,1,m-1]
        alfa_x=parameters[k,2,m-1]
        alfa_y=parameters[k,3,m-1]
        beta_x=parameters[k,4,m-1]
        beta_y=parameters[k,5,m-1]
        beta_e=parameters[k,6,m-1]
        ex=parameters[k,7,m-1]
        ey=parameters[k,8,m-1]
        angle_e=parameters[k,9,m-1]
        CD[:,k] = np.double(beta_e!=0.0) * TripleSigmoid_E(np.array([a,b]),tx,ty,alfa_x,alfa_y,beta_x,beta_y,beta_e,ex,ey,angle_e)

    # Computing membership values to achromatic categories
    valueAchro = max(1.0-sum(CD,1),0)
    CD[:,numChromatics+0] = valueAchro * Sigmoid(L,paramsAchro[0,0],paramsAchro[0,1])
    CD[:,numChromatics+1] = valueAchro * Sigmoid(L,paramsAchro[1,0],paramsAchro[1,1]) * Sigmoid(L,paramsAchro[2,0],paramsAchro[2,1])
    CD[:,numChromatics+2] = valueAchro * Sigmoid(L,paramsAchro[3,0],paramsAchro[3,1])

    # Returning the color name corresponding to the maximum membership value
    index = np.argmax(CD,axis=1)
    res = colors[index]

    return CD, res


# ***********************    
#    ImColorNamingTSELabDescriptor
#    positions: 2 column vectors indicating for which points (y,x) colour naming should be calculated
#    return: CD 
#            when calling the function without 'positions' parameter, the return is a 
#            11 channel matrix where each (x,y) position contains the set of 11 probabilities 
#            to belong to the 11 basic colors. CD.spahe=(n_rows, n_columns, 11)
#
#            when calling the function with 'positions' parameter, the return is a 
#            11 column matrix where each (y) position contains the set of 11 probabilities 
#            to belong to the 11 basic colors. CD.spahe=(n_points, 11)
# ***********************
def ImColorNamingTSELabDescriptor(ima, positions=None, patchSize=1):

    # Constants
    numColors=11                               # Number of colors
    numAchromatics=3                           # Number of achromatic colors
    numChromatics=numColors-numAchromatics     # Number of chromatic colors


    # Initializations
    numLevels = np.size(thrL)-1                   # Number of Lightness levels in the model


    # Image conversion: sRGB to CIELab
    Lab = RGB2Lab(ima)
    if positions!=None:
        if patchSize==1:
            Lab = Lab[positions[:,0],positions[:,1],:].reshape((1,-1,3))
        else:
            LabPatch = np.zeros((positions.shape[0],(2*np.trunc(patchSize/2)+1)**2,3))
            padSz = (int(np.trunc(patchSize/2)),int(np.trunc(patchSize/2)))
            Lab = np.pad(Lab,(padSz,padSz,(0,0)), 'symmetric')
            positions += padSz[0]
            c=0
            for x in range(-padSz[0],padSz[0]+1):
                for y in range(-padSz[0],padSz[0]+1):
                    LabPatch[:,c,:]=Lab[positions[:,0]+y,positions[:,1]+x,:]
                    c += 1
            Lab=LabPatch

    S = np.shape(Lab)
    if Lab.ndim==2:
        L=Lab[:,0].flatten()
        a=Lab[:,1].flatten()
        b=Lab[:,2].flatten()
        nr = S[0]; nc = 1;                      # Image dimensions: rows, columns, and channels
    else:
        L=Lab[:,:,0].flatten()
        a=Lab[:,:,1].flatten()
        b=Lab[:,:,2].flatten()
        nr = S[0]; nc = S[1];                      # Image dimensions: rows, columns, and channels

    npix = nr*nc                                  # Number of pixels
    CD = np.zeros((npix,numColors))                 # Color descriptor to store results

    # Assignment of the sample to its corresponding level
    m = np.zeros(np.shape(L))
    m[np.where(L==0)[0]] = 1    # Pixels with L=0 assigned to level 1
    for k in range(1,numLevels+1):
        m = m + np.double(thrL[k-1]<L) * np.double(L<=thrL[k]) * np.double(k)

    m = m.astype(int) - 1

    # Computing membership values to chromatic categories
    for k in range(numChromatics):
        tx=np.reshape(parameters[k,0,m],(npix,1))
        ty=np.reshape(parameters[k,1,m],(npix,1))
        alfa_x=np.reshape(parameters[k,2,m],(npix,1))
        alfa_y=np.reshape(parameters[k,3,m],(npix,1))
        beta_x=np.reshape(parameters[k,4,m],(npix,1))
        beta_y=np.reshape(parameters[k,5,m],(npix,1))
        beta_e=np.reshape(parameters[k,6,m],(npix,1))
        ex=np.reshape(parameters[k,7,m],(npix,1))
        ey=np.reshape(parameters[k,8,m],(npix,1))
        angle_e=np.reshape(parameters[k,9,m],(npix,1)); #figure;plot(angle_e); show()
        CD[:,k] = (np.double(beta_e!=0.0) * TripleSigmoid_E(np.vstack((a,b)),tx,ty,alfa_x,alfa_y,beta_x,beta_y,beta_e,ex,ey,angle_e)).T

    # Computing membership values to achromatic categories
    valueAchro = np.squeeze(np.maximum(1.0-np.reshape(np.sum(CD,axis=1),(npix,1)),np.zeros((npix,1))))
    CD[:,numChromatics+0] = valueAchro * Sigmoid(L,paramsAchro[0,0],paramsAchro[0,1])
    CD[:,numChromatics+1] = valueAchro * Sigmoid(L,paramsAchro[1,0],paramsAchro[1,1])*Sigmoid(L,paramsAchro[2,0],paramsAchro[2,1])
    CD[:,numChromatics+2] = valueAchro * Sigmoid(L,paramsAchro[3,0],paramsAchro[3,1])


    # Color descriptor with color memberships to all the categories (one color in each plane)
    if positions == None or patchSize>1:
        CD = np.reshape(CD,(nr,nc,numColors))
    if patchSize>1:
        CD=np.sum(CD,axis=1)
        CD=CD/np.tile(np.sum(CD,axis=1).reshape(-1,1),(1,numColors))

    if Lab.ndim==2:
        CD = np.reshape(CD, (-1,CD.shape[2]))

    CD = CD/np.expand_dims(np.sum(CD, axis=len(CD.shape)-1), axis=len(CD.shape)-1)
    return  CD
# ***********************    
#    ImColorNamingTSELab
#    positions: 2 column vectors indicating for which points (y,x) colour naming should be calculated
# ***********************
def ImColorNamingTSELab(ima):

    # Constants
    #colors=['Red','Orange','Brown','Yellow','Green','Blue','Purple','Pink','Black','Grey','White']

    CD = ImColorNamingTSELabDescriptor(ima)

    # Output image with each pixel labelled with the colour of maximum membership value
    imaIndex = np.argmax(CD,axis=2)
    counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for ii in np.nditer(imaIndex):
	    counter[ii] += 1


   
    #imaRes = reshape(ColorName2rgb(imaIndex, colors),(CD.shape[0], CD.shape[1],3))
    imaRes = ColorName2rgb(imaIndex, colors)

    return CD, imaRes, imaIndex, counter