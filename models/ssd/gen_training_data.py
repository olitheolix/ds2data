import os
import glob
import random
import argparse
import textwrap
import stamp_images
import compile_bboxes

import numpy as np
from PIL import Image


def parseCmdline():
    """Parse the command line arguments."""
    description = textwrap.dedent('''\
        Stamp foreground shapes into background images and record their BBoxes.

        The background images will be loaded from "./data/background" and the
        output images, along with the pickled BBox data will be written to
        "./data/stamped".

        The foreground shapes will be loaded from "./data/shapes". This folder
        must contain specimen images for each shape in a separate sub-folder,
        eg "./data/shapes/dog". Use `-cls-specimen` to limit the number of
        images loaded for each label.

        Use the `--dummy-shapes` option to ignore the folder and stamp only
        discs and boxes into the image. This is mostly useful for debugging
        because nets are easy to train for those two shapes.

        Usage examples:
          gen_stamped.py -width 32 -height 32 -num-stamps 100
          gen_stamped.py -dummy-shapes
    ''')

    # Create a parser and program description.
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    padd = parser.add_argument

    # Add the command line options.
    padd('-N', metavar='', type=int, default=100,
         help='Number of output images')
    padd('-width', metavar='', type=int, default=64, help='Width (default 64)')
    padd('-height', metavar='', type=int, default=64, help='Height (default 64)')
    padd('-seed', metavar='', type=int, default=None,
         help='Seed value for reproducible results (default None)')
    padd('-num-stamps', metavar='', type=int, default=20,
         help='Number of objects to embed in each background image (default 20)')
    padd('-cls-specimen', metavar='', type=int, default=64,
         help='Limit number of specimen images to load for each label')
    padd('--dummy-shapes', action='store_true', default=False,
         help='Use dummy shapes instead of loading them from disk')

    # Parse the actual arguments.
    param = parser.parse_args()
    random.seed(param.seed)
    np.random.seed(param.seed)
    return param


def createForegroundShapes(param):
    """Return hard coded 'square' and 'disc' shape."""
    # The box and disk will fill the entire BBox minus this margin in pixels.
    margin = 2

    # Convenience.
    height, width = param.height, param.width

    # Box shape.
    box = np.zeros((height, width, 4), np.uint8)
    box[margin:-margin, margin:-margin, :] = 255

    # Disc shape.
    centre_x = width / 2
    centre_y = height / 2
    circle_radius = min(height, width) // 2 - margin
    disc = np.zeros((height, width, 4), np.uint8)
    for y in range(height):
        for x in range(width):
            dist = np.sqrt(((x - centre_x) ** 2 + (y - centre_y) ** 2))
            disc[y, x, :] = 255 if dist < circle_radius else 0

    # Put shapes in a dict for easier processing.
    shapes = dict(box=box, disc=disc)

    # Replicate each shape N times. Give each shape a random colour.
    out = {}
    for name, shape in shapes.items():
        img = np.array([shape] * param.cls_specimen)
        for i in range(len(img)):
            img[i, :, :, :3] = img[i, :, :, :3] * np.random.uniform(0.3, 1, 3)
        out[name] = img

    # Save shapes into NumPy array and return it with a human readable mapping.
    return out


def loadForegroundShapes(path, param):
    # Convenience.
    height, width = param.height, param.width

    # Makeshift threshold to remove images that are too dark. All images with
    # less cumulative intensity will be considered too dark.
    thresh = 100 * width * height

    # Load cubes for all 10 labels.
    out = {}
    for i in range(10):
        # Path to cubes with current label (ie number of side).
        name = f'{i:02d}'
        fnames = glob.glob(os.path.join(path, name, '*.jpg'))

        # Load all cubes and assign an alpha map. Skip cubes that are too dark
        # as they have no chance of being identified.
        out[name] = []
        for j, fname in enumerate(fnames):
            # Load the image into a NumPy array.
            img = Image.open(fname).convert('RGBA')
            if img.size != (width, height):
                img = img.resize((width, height))
            img = np.array(img, np.uint8)

            # Use makeshift threshold to remove images that are too dark.
            if np.sum(img[:, :, :3]) < thresh:
                continue

            # Compute alpha mask. Basically, we _hope_ that dark areas are
            # outside the cube.
            intensity = np.sum(np.array(img[:, :, :3], np.float32), axis=2)
            alpha = np.interp(intensity, [0, 20, 30, 255], [0, 25, 255, 255])
            img[:, :, 3] = alpha

            # Add the image the list and abort the loop once we have enough images.
            out[name].append(img)
            if len(out[name]) >= param.cls_specimen:
                break

        # Convert the list of NumPy arrays to a single NumPy array.
        out[name] = np.array(out[name], np.uint8)
        assert out[name].shape[1:] == (height, width, 4)
    return out


def findBackgroundImages(path):
    """Return list of background image file names"""
    # Abort if the data set does not exist.
    fnames = glob.glob(f'{path}/*.jpg')
    if len(fnames) == 0:
        # fixme: correct data path and download location.
        print(f'\nError: No files in {path}')
        print('\nPlease download '
              'https://github.com/olitheolix/ds2data/blob/master/ds2.tar.gz'
              '\nand unpack it to data/\n')
        raise FileNotFoundError
    return fnames


def main():
    param = parseCmdline()

    # Folders with background images and for output images.
    base = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(base, 'data')
    src_path = os.path.join(base, 'background')
    dst_path = os.path.join(base, 'stamped')
    shape_path = os.path.join(base, 'shapes')

    # Ensure output path exists.
    os.makedirs(dst_path, exist_ok=True)

    # Compile a list of all background image files we have.
    bg_fnames = findBackgroundImages(src_path)
    assert len(bg_fnames) > 0

    # Load the foreground objects.
    if param.dummy_shapes:
        shapes = createForegroundShapes(param)
    else:
        shapes = loadForegroundShapes(shape_path, param)

    # Compile a map to convert a label ID to its human readable name. Note that
    # this is 1-based because label 0 is reserved for "background" (ie
    # no-object) label in later stages.
    int2name = {idx + 1: name for idx, name in enumerate(shapes)}
    int2name[0] = 'background'

    # Stamp the foreground objects into background images.
    stamp_images.generate(dst_path, param, bg_fnames, shapes, int2name)

    stamped_path = os.path.dirname(os.path.abspath(__file__))
    stamped_path = os.path.join(stamped_path, 'data', 'stamped')
    compile_bboxes.generate(stamped_path)


if __name__ == '__main__':
    main()
