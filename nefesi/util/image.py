import numpy as np

from keras.preprocessing import image
from scipy.ndimage.interpolation import rotate
# from ..neuron_feature import get_image_receptive_field
from nefesi.neuron_feature import get_image_receptive_field


class ImageDataset(object):

    def __init__(self, src_dataset, target_size=None, preprocessing_function=None):
        self.src_dataset = src_dataset
        self.target_size = target_size
        self.preprocessing_function = preprocessing_function

    def load_images(self, image_names, prep_function=True):
        images = []
        # if not isinstance(image_names, list):
        #     image_names = [image_names]

        for n in image_names:

            # print self.src_dataset, n
            i = image.load_img(self.src_dataset + n, target_size=self.target_size)
            i = image.img_to_array(i)

            # i -= avg_img
            images.append(i)

        if self.preprocessing_function is not None and prep_function is True:
            images = self.preprocessing_function(np.asarray(images))

        return images

    def load_image(self, img_name):
        return image.load_img(self.src_dataset + img_name,
                              target_size=self.target_size)

    def get_patch(self, img_name, crop_pos):
        img = image.load_img(self.src_dataset + img_name, target_size=self.target_size)
        ri, rf, ci, cf = crop_pos
        im_crop = img.crop((ci, ri, cf, rf))
        return im_crop


def rgb2opp(img):
    """Converts an image from RGB space to OPP (Opponent color space).

    :param img: Numpy array
    :return: Numpy array
    """
    opp = np.zeros(shape=img.shape)

    x = img / 255.
    R = x[:, :, 0]
    G = x[:, :, 1]
    B = x[:, :, 2]
    opp[:, :, 0] = (R + G + B - 1.5) / 1.5
    opp[:, :, 1] = (R - G)
    opp[:, :, 2] = (R + G - 2 * B) / 2

    return opp


def image2max_gray(img):
    """Converts an image to gray scale using the PCA in order to maximally
     preserve the shape pattern of the image.

    :param img: Numpy array
    :return: PIL instance (gray scale image in RGB space)
    """
    x = img.reshape(-1, 3)

    M = (x - np.mean(x, axis=0))

    latent, coeff = np.linalg.eig(np.cov(M.T))
    res = np.dot(M, coeff[:, 0])

    res = res[:, np.newaxis]
    res = np.tile(res, (1, 1, 3))
    res = np.reshape(res, img.shape)

    res = image.array_to_img(res, scale=True)
    return res


def crop_image(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def rotate_images(model, images, degrees, pos, layer):
    images_rotated = []
    for i in xrange(len(images)):
        init_image = np.copy(images[i])
        x, y = pos[i]
        print x, y
        row_ini, row_fin, col_ini, col_fin = get_image_receptive_field(x, y, model, layer)
        receptive_field = init_image[row_ini:row_fin+1, col_ini:col_fin+1]
        w, h, d = receptive_field.shape
        print w, h, d
        padding_w = int(round(w/2))
        padding_h = int(round(h/2))
        if padding_w % 2 != 0:
            padding_w += 1
        if padding_h % 2 != 0:
            padding_h += 1

        new_shape = np.zeros((w + padding_w, h + padding_h, d), dtype=receptive_field.dtype)
        print new_shape.shape

        for dim in xrange(d):
            new_shape[:,:,dim] = np.pad(receptive_field[:,:,dim], ((padding_w/2, padding_w/2),(padding_h/2, padding_h/2)), mode='edge')

        img = rotate(new_shape, degrees, reshape=False)
        img = crop_image(img, h, w)
        init_image[row_ini:row_fin + 1, col_ini:col_fin + 1] = img

        images_rotated.append(init_image)


    return images_rotated



def rotate_images_axis(images, rot_axis, model, layer, pos):

    rot_images = []
    for i in xrange(len(images)):
        init_image = np.copy(images[i])
        x, y = pos[i]
        row_ini, row_fin, col_ini, col_fin = get_image_receptive_field(x, y, model, layer)
        receptive_field = init_image[row_ini:row_fin+1, col_ini:col_fin+1]

        rf_shape = receptive_field.shape

        rotated_receptive_field = rotate_rf(receptive_field, rot_axis)

        rot_rf_shape = rotated_receptive_field.shape


        if rf_shape != rot_rf_shape:
            rotated_receptive_field = np.reshape(rotated_receptive_field, rf_shape)

        init_image[row_ini:row_fin + 1, col_ini:col_fin + 1] = rotated_receptive_field
        rot_images.append(init_image)

    return rot_images


def rotate_rf(img, rot_axis):
    if rot_axis == 0:
        return np.flipud(img)

    if rot_axis == 45:
        n_image = img.transpose(1, 0, 2)
        n_image = np.flipud(n_image)
        return np.fliplr(n_image)

    if rot_axis == 90:
        return np.fliplr(img)

    if rot_axis == 135:
        return img.transpose(1, 0, 2)

    return None
