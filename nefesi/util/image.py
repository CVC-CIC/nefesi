import numpy as np

from keras.preprocessing import image
from scipy.ndimage.interpolation import rotate


class ImageDataset(object):
    """This class stores the whole information about a dataset and provides
    some functions for load the images.

    Arguments:
        src_dataset: String, path of dataset.
        target_size: Tuple of integers, image height and width.
        preprocessing_function: Function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument: batch of images (Numpy tensor with rank 4),
             and should output a Numpy tensor with the same shape.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
    """

    def __init__(self, src_dataset, target_size=None,
                 preprocessing_function=None, color_mode='rgb'):
        self.src_dataset = src_dataset
        self.target_size = target_size
        self.preprocessing_function = preprocessing_function
        self.color_mode = color_mode

    def load_images(self, image_names, prep_function=True):
        """Returns a list of images after applying the
         corresponding transformations.

        :param image_names: List of strings, name of the images.
        :param prep_function: Boolean.

        :return: List of Numpy arrays.
        """
        images = []
        for n in image_names:
            i = self._load_image(n)
            i = image.img_to_array(i)
            images.append(i)

        if self.preprocessing_function is not None and prep_function is True:
            images = self.preprocessing_function(np.asarray(images))
        return images

    def get_patch(self, img_name, crop_pos):
        """Returns a region patch from an image.

        :param img_name: String, name of the image.
        :param crop_pos: Tuple of integers (left, upper, right, lower),
            pixel coordinates of the region to be cropped.

        :return: PIL image instance.
        """
        img = self._load_image(img_name)
        ri, rf, ci, cf = crop_pos
        im_crop = img.crop((ci, ri, cf, rf))
        return im_crop

    def _load_image(self, img_name):
        """Loads an image into PIL format.

        :param img_name: String, name of the image.

        :return: PIL image instance.
        """
        grayscale = self.color_mode == 'grayscale'
        return image.load_img(self.src_dataset + img_name,
                              grayscale=grayscale,
                              target_size=self.target_size)

    def __str__(self):
        return str.format("Dataset dir: {}, target_size: {}, color_mode: {}, "
                          "preprocessing_function: {}.", self.src_dataset,
                          self.target_size,
                          self.color_mode,
                          self.preprocessing_function)


def rgb2opp(img):
    """Converts an image from RGB space to OPP (Opponent color space).

    :param img: Numpy array of shape (height, width, channels).
        RGB image with values between [0, 255].

    :return: Numpy array, same shape as the input.

    :raise:
        ValueError: If invalid `img` is passed.
    """
    if img.shape[2] != 3:
        raise ValueError("Unsupported image shape: {}.", img.shape)

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
    """Converts an image to grayscale using PCA, in order to maximally
     preserve the shape pattern of the image.

    :param img: Numpy array of rank 3.

    :return: PIL image instance.
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


def crop_image(img, crop_x, crop_y):
    y, x, _ = img.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return img[start_y:start_y + crop_y, start_x:start_x + crop_x]


def rotate_images(images, degrees, pos, layer_data):
    """Rotates the receptive field for each image in `images`.

    :param images: List of numpy arrays.
    :param degrees: Float, the rotation angle in degrees.
    :param pos: List of receptive fields locations on `images`.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.

    :return: List of numpy arrays, same as the input `images` but rotated.
    """
    images_rotated = []
    for i in xrange(len(images)):
        init_image = np.copy(images[i])
        x, y = pos[i]

        # get the receptive field from the image
        row_ini, row_fin, col_ini, col_fin = layer_data.receptive_field_map[x, y]
        receptive_field = init_image[row_ini:row_fin, col_ini:col_fin]

        # adjusts the receptive field for not add black padding on it.
        w, h, d = receptive_field.shape
        padding_w = int(round(w / 2))
        padding_h = int(round(h / 2))
        if padding_w % 2 != 0:
            padding_w += 1
        if padding_h % 2 != 0:
            padding_h += 1
        new_shape = np.zeros((w + padding_w, h + padding_h, d),
                             dtype=receptive_field.dtype)
        for dim in xrange(d):
            new_shape[:, :, dim] = np.pad(receptive_field[:, :, dim],
                                          ((padding_w / 2, padding_w / 2),
                                           (padding_h / 2, padding_h / 2)),
                                          mode='edge')

        # apply the rotation function
        img = rotate(new_shape, degrees, reshape=False)
        # build back the origin image with the receptive field rotated
        img = crop_image(img, h, w)
        init_image[row_ini:row_fin, col_ini:col_fin] = img

        images_rotated.append(init_image)
    return images_rotated


def rotate_images_axis(images, rot_axis, layer_data, pos):
    """Rotates (flips) the receptive field for each image in `images`.

    :param images: List of numpy arrays.
    :param rot_axis: Integer, the rotation axis to flip the image.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param pos: List of receptive fields locations on `images`.

    :return: List of numpy arrays, same as the input `images` but flipped.
    """
    rot_images = []

    for i in xrange(len(images)):
        init_image = np.copy(images[i])
        x, y = pos[i]
        # get the receptive field from the image
        row_ini, row_fin, col_ini, col_fin = layer_data.receptive_field_map[x, y]
        receptive_field = init_image[row_ini:row_fin, col_ini:col_fin]

        rf_shape = receptive_field.shape
        rotated_receptive_field = rotate_rf(receptive_field, rot_axis)

        # if receptive field flipped has not same shape that before, resize it
        rot_rf_shape = rotated_receptive_field.shape
        if rf_shape != rot_rf_shape:
            rotated_receptive_field = np.reshape(rotated_receptive_field, rf_shape)

        # build back the origin image with receptive field flipped
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
