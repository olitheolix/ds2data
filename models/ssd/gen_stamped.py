""" Load background images, stamp shapes into it and save them as new images.

Each image will be saved as JPG alongside a JSON file with the same name. The
JSON file contains information about the BBox and label of each object that was
stamped into the image.
"""
import os
import glob
import json
import random
import numpy as np

from PIL import Image


def createForegroundShapes(width, height):
    """Return hard coded 'square' and 'disc' shape."""
    margin = 2

    # Box shape.
    box = np.zeros((height, width, 3), np.uint8)
    box[margin:-margin, margin:-margin, :] = 255

    # Disc shape.
    centre_x = width / 2
    centre_y = height / 2
    circle_radius = min(height, width) // 2 - margin
    disc = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        for x in range(width):
            dist = np.sqrt(((x - centre_x) ** 2 + (y - centre_y) ** 2))
            disc[y, x, :] = 255 if dist < circle_radius else 0

    # Save shapes into NumPy array and return it with a human readable mapping.
    return dict(box=box, disc=disc)


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


def stampImage(background, fg_shapes, N, xmin, xmax, ymin, ymax):
    """Return a new image with N shapes in it, and their BBox and labels.
    """
    # Background and shape images must be RGB.
    assert background.ndim == 3
    assert background.shape[2] == 3
    for fg in fg_shapes.values():
        assert fg.ndim == 3
        assert fg.shape[2] == 3

    # Convenience
    bg_height, bg_width = background.shape[:2]
    out = np.array(background, np.uint8)
    bboxes, labels = [], []
    stencil = np.zeros_like(out)

    # Stamp N non-overlapping images into the background. If this proves
    # difficult, abort after `max_attempts` and return what we have so far.
    attempts, max_attempts = 0, 10 * N
    while len(labels) < N and attempts < max_attempts:
        attempts += 1

        # Compute random region in foreground image to put the object.
        w = np.random.randint(xmin, xmax)
        h = np.random.randint(ymin, ymax)
        x0 = np.random.randint(0, bg_width - w)
        y0 = np.random.randint(0, bg_height - h)
        x1, y1 = x0 + w, y0 + h

        # Verify if the region is already occupied. Do nothing if it is.
        if np.sum(stencil[y0:y1, x0:x1]) > 0:
            continue
        stencil[y0:y1, x0:x1] = 1

        # Pick a random foreground image.
        label = random.choice(list(fg_shapes.keys()))
        labels.append(label)
        bboxes.append([x0, y0, x1, y1])
        fg = np.array(fg_shapes[label])

        # Make foreground shape a random colour.
        for chan in range(fg.shape[2]):
            fg[:, :, chan] = fg[:, :, chan] * np.random.uniform(0.3, 1)

        # Scale the foreground image.
        fg = Image.fromarray(fg.astype(np.uint8)).resize((w, h), Image.BILINEAR)
        fg = np.array(fg, np.uint8)

        # Compute a mask to only copy the image portion that contains the
        # object but not those that contain only the black background.
        idx = np.nonzero(fg > 30)
        mask = np.zeros_like(fg)
        mask[idx] = 1

        # Stamp the foreground object into the background image.
        out[y0:y1, x0:x1, :] = (1 - mask) * out[y0:y1, x0:x1, :] + mask * fg
    return out, bboxes, labels


def generateImages(dst_path, bg_fnames, fg_shapes, int2name, num_img, num_stamps):
    """Create N stamped background images and save them."""
    xmax = max([_.shape[1] for _ in fg_shapes.values()])
    ymax = max([_.shape[0] for _ in fg_shapes.values()])
    xmin, ymin = int(0.5 * xmax), int(0.5 * ymax)
    dims = (xmin, xmax, ymin, ymax)

    # Create N images.
    for i in range(num_img):
        # Load background image as NumPy array.
        img = Image.open(bg_fnames[i % len(bg_fnames)]).convert('RGB')
        img = np.array(img, np.uint8)

        # Stamp foreground shapes into background image.
        img, bboxes, labels = stampImage(img, fg_shapes, num_stamps, *dims)

        # File name prefix for image and meta data.
        fname = os.path.join(dst_path, f'{i:04d}')

        # Save meta data.
        meta = {'bboxes': bboxes, 'labels': labels, 'int2name': int2name}
        json.dump(meta, open(fname + '.json', 'w'))

        # Save the stamped image.
        Image.fromarray(img).save(fname + '.jpg')


def main():
    # Folders with background images and for output images.
    base = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(base, 'data')
    src_path = os.path.join(base, 'background')
    dst_path = os.path.join(base, 'stamped')

    # Ensure output path exists.
    os.makedirs(dst_path, exist_ok=True)

    # Compile a list of all background image files we have.
    bg_fnames = findBackgroundImages(src_path)
    assert len(bg_fnames) > 0

    # Load the foreground objects.
    shapes = createForegroundShapes(width=32, height=32)

    # Compile a map to convert a label ID to its human readable name. Note that
    # this is 1-based because label 0 will always be reserved for the
    # background (ie no-object) label in later stages.
    int2name = {idx + 1: name for idx, name in enumerate(shapes)}

    # Stamp the foreground objects into the background images.
    generateImages(dst_path, bg_fnames, shapes, int2name, num_img=2, num_stamps=20)


if __name__ == '__main__':
    main()
