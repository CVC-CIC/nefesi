from keras.preprocessing import image
import read_activations


def load_images(dataset_path, image_names):
    images = []
    for n in image_names:
        i = image.load_img(dataset_path + n, target_size=(224, 224))
        i = image.img_to_array(i)
        # i -= avg_img
        images.append(i)

    # i = image.array_to_img(images[0], scale=False)
    # i.save('origin.png')
    return images


def get_similarity_index(filter_a, filter_b, idx_a, idx_b, model, layer, dataset_path):
    act_a = filter_a.get_activations()
    act_b = filter_b.get_activations()


    # Falta normalizacion!!!!

    if act_a[0] != 0.0 and act_b[0] != 0.0:
        images_a = filter_a.get_images_id()
        images_b = filter_b.get_images_id()

        images_a = load_images(dataset_path, images_a)
        images_b = load_images(dataset_path, images_b)

        locations_a = filter_a.get_locations()
        locations_b = filter_b.get_locations()

        a_act_b = read_activations.get_activation_from_pos(images_a, model, layer, idx_b, locations_a)
        b_act_a = read_activations.get_activation_from_pos(images_b, model, layer, idx_a, locations_b)

        a_act_b = a_act_b/act_b[0]
        b_act_a = b_act_a/act_a[0]

        A = sum(b_act_a) / sum(filter_a.get_norm_activations())
        B = sum(a_act_b) / sum(filter_b.get_norm_activations())

        return (A + B) / 2

    else:
        return None

