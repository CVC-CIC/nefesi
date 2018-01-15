
import read_activations

def get_similarity_index(filter_a, filter_b, idx_a, idx_b, model, layer, dataset):
    act_a = filter_a.get_activations()
    act_b = filter_b.get_activations()

    if act_a[0] != 0.0 and act_b[0] != 0.0:
        # images_a = filter_a.get_images_id()
        images_b = filter_b.get_images_id()

        # images_a = dataset.load_images(images_a)
        images_b = dataset.load_images(images_b)

        # locations_a = filter_a.get_locations()
        locations_b = filter_b.get_locations()

        new_act = read_activations.get_activation_from_pos(images_b, model, layer, idx_a, locations_b)
        norm_new_act = new_act/act_a[0]
        # a_act_b = a_act_b/act_b[0]
        # b_act_a = b_act_a/act_a[0]

        # A = sum(b_act_a) / sum(filter_a.get_norm_activations())
        # B = sum(a_act_b) / sum(filter_b.get_norm_activations())
        #
        # return (A + B) / 2

        return sum(norm_new_act / sum(filter_a.get_norm_activations()))

    else:
        return None

