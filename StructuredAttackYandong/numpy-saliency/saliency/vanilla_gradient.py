import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import os
import copy
import pdb

RESULTS_FOLDER = "results"


def save_vanilla_gradient(network, data, labels):
    """ Implements gradient visualization with vanilla backprop. """

    # Create a saliency map for each data point
    for i, image in enumerate(data):
        # Forward pass on image
        output = image
        for l in range(len(network.layers)):
            output = network.layers[l].forward(output)

        # Backprop to get gradient
        label_one_hot = labels[i]
        dy = np.array(label_one_hot)
        for l in range(len(network.layers)-1, -1, -1):
            dout = network.layers[l].backward(dy)
            dy = dout
        raw_saliency_map = dout

        pdb.set_trace()
        # Process saliency map
        trimmed_saliency_map = trim_map(raw_saliency_map)
        saliency_map = normalize_array(trimmed_saliency_map)

        # Process data image for rendering
        image = normalize_array(image)
        image = format_np_output(image)
        image = Image.fromarray(image)

        # Export saliency map renderings
        filename = "index-{0}_class-{1}_vanilla".format(
            str(i), str(np.argmax(label_one_hot)))
        save_gradient_images(image, saliency_map, filename)

        print("Saved Vanilla Gradient image to results folder")


def trim_map(arr):
    """
        Remove zero padding & color channel
    """
    vertical_trimmed = arr[0][2:30]
    horizontal_trimmed = []
    for row in vertical_trimmed:
        horizontal_trimmed.append(row[2:30])
    trimmed = np.array(horizontal_trimmed)
    return trimmed


def normalize_array(arr):
    """
        Sets array values to span 0-1
    """
    arr = arr - arr.min()
    arr /= arr.max()
    return arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_gradient_images(org_img, saliency_map, file_name):
    """
        Saves saliency map and overlay on the original image

    Args:
        org_img (PIL img): Original image
        saliency_map (numpy arr): Saliency map (grayscale) 0-255
        file_name (str): File name of the exported image
    """

    # Grayscale saliency map
    heatmap, heatmap_on_image = apply_colormap_on_image(
        org_img, saliency_map, 'RdBu')

    # Save original
    path_to_file = os.path.join(RESULTS_FOLDER, file_name+'_base.png')
    save_image(org_img, path_to_file)

    # Save heatmap on image
    path_to_file = os.path.join(
        RESULTS_FOLDER, file_name+'_saliency_overlay.png')
    save_image(heatmap_on_image, path_to_file)

    # Save colored heatmap
    path_to_file = os.path.join(RESULTS_FOLDER, file_name+'_saliency.png')
    save_image(heatmap, path_to_file)

    # Save grayscale heatmap
    # path_to_file = os.path.join(RESULTS_FOLDER, file_name+'_grayscale.png')
    # save_image(saliency_map, path_to_file)


def apply_colormap_on_image(org_im, saliency_map, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        saliency_map (numpy arr): Saliency map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    import pdb
    pdb.set_trace()
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(saliency_map)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.5
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(
        heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image



