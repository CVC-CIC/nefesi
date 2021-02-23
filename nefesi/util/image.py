import numpy as np
import os
# from keras.preprocessing import image
from PIL import Image
from scipy.ndimage.interpolation import rotate
import warnings

ACCEPTED_COLOR_MODES = ['rgb','grayscale']

class ImageDataset():
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

    # ------------------------------------------- CONSTRUCTOR -----------------------------------------------


    def __init__(self, src_dataset, target_size=None,
                 preprocessing_function=None, color_mode='rgb', src_segmentation_dataset = None):

        self.src_dataset = src_dataset
        self.src_segmentation_dataset = src_segmentation_dataset
        self.target_size = target_size
        self.preprocessing_function = preprocessing_function
        self.color_mode = color_mode

    #---------------------------------------- SETTERS AND GETTERS -------------------------------------------

    @property
    def target_size(self):
        return self._target_size

    @target_size.setter
    def target_size(self, target_size):
        #Convert if is list or set to tuple to add flexibility
        if type(target_size) in (list,set):
            target_size = tuple(target_size)

        #Verify that is None or a valid formatted tuple
        if type(target_size) is tuple:
            if len(target_size) != 2:
                raise ValueError("target_size must be a (height, width) tuple (or None). "+str(target_size)+" not valid.")
            if target_size[0] is None or target_size[1] is None:
                target_size = (224, 224)
            if type(target_size[0]) != int or type(target_size[1]) != int:
                raise ValueError("target_size must be a (height, width) tuple (or None). "+str(target_size)+" not valid.")
        elif target_size is not None:
            raise ValueError("target_size must be a (height, width) tuple (or None). '"+str(type(target_size))+
                             "' type not admitted ")
        # Sets
        self._target_size = target_size

    @property
    def src_dataset(self):
        return self._src_dataset
    @src_dataset.setter
    def src_dataset(self,src_dataset):
        if src_dataset is not None:
            if type(src_dataset) is not str:
                raise ValueError("src_dataset attribute must be str")
            elif not os.path.isdir(src_dataset):
                raise FileNotFoundError(src_dataset+" not exists or is not a directory")
            elif os.listdir(src_dataset) == []:
                warnings.warn(src_dataset+" is an empty directory",FutureWarning)
            src_dataset = os.path.join(src_dataset, '')
        # Sets
        self._src_dataset = src_dataset

    @property
    def src_segmentation_dataset(self):
        return self._src_segmentation_dataset

    @src_segmentation_dataset.setter
    def src_segmentation_dataset(self,src_segmentation_dataset):
        if src_segmentation_dataset is not None:
            if type(src_segmentation_dataset) is not str:
                raise ValueError("src_dataset attribute must be str")
            elif not os.path.isdir(src_segmentation_dataset):
                warnings.warn(src_segmentation_dataset+" not exists or is not a directory."
                                                       " Remember to create it before analyzing object sel.",ResourceWarning)
            elif os.listdir(src_segmentation_dataset) == []:
                warnings.warn(src_segmentation_dataset+" is an empty directory",FutureWarning)
            src_segmentation_dataset = os.path.join(src_segmentation_dataset, '')
        # Sets
        self._src_segmentation_dataset = src_segmentation_dataset

    @property
    def preprocessing_function(self):
        return self._preprocessing_function
    @preprocessing_function.setter
    def preprocessing_function(self, preprocessing_function):
        #Verify if preprocessing function is None or a function that takes only 1 non-default argument
        if preprocessing_function is not None:
            #if function don't takes one non-default argument
            if callable(preprocessing_function):
                pass
                # if ((preprocessing_function.__code__.co_argcount - len(preprocessing_function.__defaults__)) != 1):
                #     raise ValueError("preprocessing_function argument must take a a numpy tensor 4D as argument"
                #                      " (any number of default arguments also admitted) and must return a numpy tensor of same"
                #                      "dimension.")
            #if not is None or a function
            else:
                raise ValueError("preprocessing_function must be None or a function (that takes a numpy tensor 4D"
                             "(with a batch of images) as argument and returns a numpy tensor of same dimension")
            # Sets
            self._preprocessing_function = preprocessing_function


    @property
    def color_mode(self):
        return self._color_mode
    @color_mode.setter
    def color_mode(self,color_mode):
        #Verify if is an admitted color_mode and put it in lower case before assign it
        if type(color_mode) is not str:
            raise ValueError("color_mode attribute must be str. With one of these values: "+str(ACCEPTED_COLOR_MODES))
        color_mode = color_mode.lower()
        if color_mode not in ACCEPTED_COLOR_MODES:
            raise ValueError("color_mode attribute must be one of these values: " + str(ACCEPTED_COLOR_MODES)+". '"+color_mode+
                             "' not accepted.")
        #Sets
        self._color_mode = color_mode

# ------------------------------------------- FUNCTIONS -----------------------------------------------


    #TO COMMENT.
    def load_images(self, image_names, prep_function=True):
        """Returns a list of images after applying the
         corresponding transformations.

        :param image_names: List of strings, name of the images.
        :param prep_function: Boolean.

        :return: Numpy array that contains the images (1+N dimension where N is the dimension of an image).
        """
        #Have the first to generalize channel shapes (in order to don't need to recode if new color_modes will be accepted)
        img = self._load_image(image_names[0])
        #Gets the output shape in order to assing a shape to images matrix
        outputShape = [len(image_names)]
        outputShape.extend(list(img.shape))
        #Declare the numpy where all images will be saved
        images = np.zeros(shape=tuple(outputShape),dtype=img.dtype)
        images[0] = img #assign de first
        for i in range(1,len(image_names)):
            img = self._load_image(image_names[i])
            images[i] = image.img_to_array(img)

        if self.preprocessing_function is not None and prep_function is True:
            #dtype = images.dtype
            #also for problems with the keras backend
            images = images.astype(np.float32)
            images = self.preprocessing_function(images) #np.asarray(images)) #Now are array right since the beginning
                                                            #NEEDS TO BE TESTED IF REALLY CONTINUE WORKING FINE
        return images

    def get_patch(self, img_name, crop_pos=None):
        """Returns a region patch from an image.

        :param img_name: String, name of the image.
        :param crop_pos: Tuple of integers (left, upper, right, lower),
            pixel coordinates of the region to be cropped.

        :return: PIL image instance.
        """
        img = self._load_image(img_name)
        if crop_pos is None:
            return img

        return img[crop_pos[0]:crop_pos[1], crop_pos[2]:crop_pos[3]]


    def _load_image(self, img_name, prep_function=False):
        """Loads an image into PIL format.

        :param img_name: String, name of the image.

        :return: PIL image instance
        """
        grayscale = self.color_mode == 'grayscale'

        # img = image.load_img(self.src_dataset + img_name,
        #                grayscale=grayscale,
        #                target_size=self.target_size)
        img = Image.open(img_name).convert('RGB')
        if grayscale:
            img = img.convert('L')
        img = img.resize(self.target_size, Image.ANTIALIAS)

        img = np.array(img)

        if self.preprocessing_function is not None and prep_function:
            img = self.preprocessing_function(img)
        return img


    def get_concepts_of_region(self, image_name, crop_pos,  normalized = True, dataset_name='ADE20K',
                               norm_activations=None):
        if dataset_name == 'ADE20K':
            tags_and_counts = []
            name = image_name[:image_name.index('.')]
            seg_name = name+'_seg.png'
            parts_name = name+'_parts_{}.png'
            atr_name = name+'_atr.txt'
            level = 0
            segmentation = self._load_image('../masks/'+seg_name)
            mask_segment = get_image_segmented(segmentation,crop_pos)
            tags, counts = np.unique(mask_segment,return_counts=True)
            if norm_activations is not None:
                counts = np.array(counts, dtype=np.float)
                for idx, id in enumerate(tags):
                    counts[idx] = np.sum(norm_activations[mask_segment==id])
            tags_and_counts.append([tags, counts])

            while True:
                level+=1
                mask_name = '../masks/'+parts_name.format(level)
                if os.path.exists(self.src_dataset+mask_name):
                    part = self._load_image(mask_name)
                else:
                    break
                mask_part = get_image_segmented(part, crop_pos)
                tags, counts = np.unique(mask_part, return_counts=True)
                if norm_activations is not None:
                    counts = np.array(counts, dtype=np.float)
                    for idx, id in enumerate(tags):
                        counts[idx] = np.sum(norm_activations[mask_part == id])
                if tags[0] == 0:
                    if len(tags)==1:
                        break
                    else:
                        tags,counts = tags[1:],counts[1:]
                tags_and_counts.append([tags, counts])

            concepts = [dict() for i in range(len(tags_and_counts))]
            if tags_and_counts[0][0][0] == 0:
                concepts[0]['unknown'] = tags_and_counts[0][1][0]
                if len(tags_and_counts[0][0]) == 1:
                    return concepts
                else:
                    tags_and_counts[0][0], tags_and_counts[0][1] = tags_and_counts[0][0][1:], tags_and_counts[0][1][1:]

            with open(self.src_dataset + '../texts/' + atr_name) as f:
                data = f.readlines()
                label = [np.zeros(len(data), dtype='U128') for i in range(level)]
                for i, line in enumerate(data):
                    splited_line = line.split(sep='#')
                    label_level = int(splited_line[1])
                    if label_level<len(label):
                        label[label_level][int(splited_line[0])-1] = splited_line[4]

            for actual_level in range(level):
                tags = tags_and_counts[actual_level][0]
                counts = tags_and_counts[actual_level][1]
                for tag, count in zip(tags, counts):
                    if label[actual_level][tag-1] in concepts[actual_level]:
                        concepts[actual_level][label[actual_level][tag - 1]] += count
                    else:
                        concepts[actual_level][label[actual_level][tag-1]] = count

        if normalized:
            if norm_activations is None:
                size = mask_segment.shape[0]*mask_segment.shape[1]
                for i in range(len(concepts)):
                    concepts[i] = np.array(list(concepts[i].items()),
                                           dtype=([('class', 'U64'), ('count', np.float)]))
                    concepts[i]['count'] /= size
            else:
                warnings.warn("normalization don't done because is already pondered by activations")


        return concepts






    def __str__(self):
        return str.format("Dataset dir: {}, target_size: {}, color_mode: {}, "
                          "preprocessing_function: {}.", self.src_dataset,
                          self.target_size,
                          self.color_mode,
                          self.preprocessing_function)

def get_correspondences_array_in_ADE20K(image_segmented):
    labels_idx = np.unique(image_segmented[:, :, 2])
    indexes_array = np.zeros(np.max(labels_idx)+1, dtype=np.uint8)
    indexes_array[labels_idx] = np.arange(0,len(labels_idx))
    return indexes_array

def get_image_segmented(segmented_image, crop_pos):
    correspondence_list = get_correspondences_array_in_ADE20K(segmented_image)
    ri, rf, ci, cf = crop_pos
    segmented_image = segmented_image[ri:rf, ci:cf, 2]
    uniques = np.unique(segmented_image)
    for i in uniques:
        segmented_image[segmented_image == i] = correspondence_list[i]
    return segmented_image

def rgb2opp(img):
    """Converts an image or imageSet from RGB space to OPP (Opponent color space).

    :param img: Numpy array of shape ([numOfImages], height, width, channels).
        RGB image with values between [0, 255].

    :return: Numpy array, same shape as the input.

    :raise:
        ValueError: If invalid `img` is passed.
    """
    if len(img.shape) not in [3,4]:
        raise ValueError("Unsupported image object shape: {}. Only 3 or 4 dimensions images accepted", img.shape)
    if img.shape[-1] != 3:
        raise ValueError("Unsupported image shape: {}.", img.shape)

    opp = np.zeros(shape=img.shape, dtype=np.float)
    x = img / 255.
    R = x[..., 0]
    G = x[..., 1]
    B = x[..., 2]
    opp[..., 0] = (R + G + B - 1.5) / 1.5
    opp[..., 1] = (R - G)
    opp[..., 2] = (R + G - 2 * B) / 2
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


def crop_center(img, crop):
    crop_y = crop[0]
    crop_x = crop[1]

    y = img.shape[0]
    x = img.shape[1]
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)

    return img[start_y:start_y + crop_y, start_x:start_x + crop_x]

def expand_im(im, margins): #margins (bl, bu, br, bd))
    sz = list(im.shape)
    sz[0] = sz[0] + margins[1] + margins[3]
    sz[1] = sz[1] + margins[0] + margins[2]
    out = np.zeros(sz, dtype=im.dtype)
    out[margins[1]:sz[0]-margins[3], margins[0]:sz[1]-margins[2]] = im
    return out


def rotate_images(images, degrees, pos, layer_data):
    """Rotates the receptive field for each image in `images`.

    :param images: List of numpy arrays, the images to rotate.
    :param degrees: list of Float, the rotation angles in degrees.
    :param pos: List of receptive fields locations on 'images'.
    :param layer_data: The 'nefesi.layer_data.LayerData' instance.

    :return: Numpy array that contains the images rotated (1+N dimension where N is the dimension of an image).
    Same as the input `images` but rotated.
    """
    #The images replicated, one for each rotation to do (at the end will contain the images rotated
    images_rotated = np.full((len(degrees),)+images.shape,images)

    for i in range(len(images)):
        x, y = pos[i]
        # get the receptive field from the image
        row_ini, row_fin, col_ini, col_fin = layer_data.receptive_field_map[x, y]
        receptive_field = images_rotated[0,i,row_ini:row_fin, col_ini:col_fin]

        # adjusts the receptive field for not add black padding on it.
        w, h, d = receptive_field.shape
        padding_w = round(w / 2)
        padding_h = round(h / 2)
        padding_w += (padding_w % 2)
        padding_h += (padding_h % 2)

        new_shape = np.pad(receptive_field,
                                          ((padding_w // 2, padding_w // 2),
                                           (padding_h // 2, padding_h // 2),
                                           (0,0)),
                                          mode='edge')
        for deg_pos, current_degrees in enumerate(degrees):
            # apply the rotation function
            img = rotate(new_shape, current_degrees, reshape=False)
            # build back the origin image with the receptive field rotated
            images_rotated[deg_pos, i, row_ini:row_fin, col_ini:col_fin] = crop_center(img, [w, h])

    return images_rotated


def rotate_images_axis(images, rot_axis, layer_data, pos):
    """Rotates (flips) the receptive field for each image in `images` (without the paddings).

    :param images: List of numpy arrays.
    :param rot_axis: Integer, the rotation axis to flip the image.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param pos: List of receptive fields locations on `images`.

    :return: 1+N-Dimensional numpy array where N is the dimension of the input images (axis 0 refers to an image (image_i
     will be img[i]), same as the input `images` but flipped.
    """
    rot_images = np.full((len(rot_axis),)+images.shape,images)

    for i in range(len(images)):
        x, y = pos[i]
        # get the receptive field from the image
        row_ini, row_fin, col_ini, col_fin = layer_data.receptive_field_map[x, y]
        for axis_pos, current_axis in enumerate(rot_axis):
            receptive_field = rot_images[axis_pos, i, row_ini:row_fin, col_ini:col_fin]
            rotated_receptive_field = rotate_rf(receptive_field, current_axis)
            # if receptive field flipped has not same shape that before, resize it
            if receptive_field.shape != rotated_receptive_field.shape:
                rotated_receptive_field = np.reshape(rotated_receptive_field, receptive_field.shape)
            # build back the origin image with receptive field flipped
            rot_images[axis_pos, i, row_ini:row_fin, col_ini:col_fin] = rotated_receptive_field
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

