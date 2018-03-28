import numpy as np

from keras.preprocessing import image
from scipy.ndimage.interpolation import rotate



class ImageDataset(object):

    def __init__(self, src_dataset, target_size=None,
                 preprocessing_function=None, color_mode='rgb'):
        self.src_dataset = src_dataset
        self.target_size = target_size
        self.preprocessing_function = preprocessing_function
        self.color_mode = color_mode

    def load_images(self, image_names, prep_function=True):

        images = []
        for n in image_names:
            i = self.load_image(n)
            i = image.img_to_array(i)
            images.append(i)

        if self.preprocessing_function is not None and prep_function is True:
            images = self.preprocessing_function(np.asarray(images))

        return images

    def get_patch(self, img_name, crop_pos):
        img = self.load_image(img_name)
        ri, rf, ci, cf = crop_pos
        im_crop = img.crop((ci, ri, cf, rf))
        return im_crop

    def load_image(self, img_name):
        grayscale = self.color_mode == 'grayscale'
        return image.load_img(self.src_dataset + img_name,
                              grayscale=grayscale,
                              target_size=self.target_size)

    def __str__(self):
        return str.format('Dataset dir: {}, target_size: {}, color_mode: {}, '
                          'preprocessing_function: {}.', self.src_dataset, self.target_size,
                          self.color_mode, self.preprocessing_function)




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


def rotate_images(images, degrees, pos, layer_data):
    images_rotated = []
    for i in xrange(len(images)):
        init_image = np.copy(images[i])
        x, y = pos[i]
        print x, y
        row_ini, row_fin, col_ini, col_fin = layer_data.receptive_field_map[x, y]
        receptive_field = init_image[row_ini:row_fin, col_ini:col_fin]
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
        init_image[row_ini:row_fin, col_ini:col_fin] = img

        images_rotated.append(init_image)

    return images_rotated


def rotate_images_axis(images, rot_axis, layer_data, pos):

    rot_images = []
    for i in xrange(len(images)):
        init_image = np.copy(images[i])
        x, y = pos[i]
        row_ini, row_fin, col_ini, col_fin = layer_data.receptive_field_map[x, y]
        receptive_field = init_image[row_ini:row_fin, col_ini:col_fin]

        rf_shape = receptive_field.shape

        rotated_receptive_field = rotate_rf(receptive_field, rot_axis)

        rot_rf_shape = rotated_receptive_field.shape


        if rf_shape != rot_rf_shape:
            rotated_receptive_field = np.reshape(rotated_receptive_field, rf_shape)

        init_image[row_ini:row_fin, col_ini:col_fin] = rotated_receptive_field
        rot_images.append(init_image)

    return rot_images


def rotate_rf(img, rot_axis):
    if rot_axis == 0:
        return np.flipud(img)

    elif rot_axis == 45:
        n_image = img.transpose(1, 0, 2)
        n_image = np.flipud(n_image)
        return np.fliplr(n_image)

    elif rot_axis == 90:
        return np.fliplr(img)

    elif rot_axis == 135:
        return img.transpose(1, 0, 2)
    else:
        return None
