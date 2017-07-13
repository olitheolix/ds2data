"""Load background images, stamp shapes into it and save them as new images.

Each image will be saved as JPG alongside a '*-meta.pickle' file. The meta file
contains information about the BBox and label of each object that was stamped
into the image.

"""
import os
import glob
import pickle
import tqdm
import random
import numpy as np

from PIL import Image


def createForegroundShapes(N, width, height):
    """Return hard coded 'square' and 'disc' shape."""
    margin = 2

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
        img = np.array([shape] * N)
        for i in range(len(img)):
            img[i, :, :, :3] = img[i, :, :, :3] * np.random.uniform(0.3, 1, 3)
        out[name] = img

    # Save shapes into NumPy array and return it with a human readable mapping.
    return out


def loadForegroundShapes(N, width, height):
    base = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(base, 'data', 'shapes')
    out = {}

    thresh = 50 * width * height

    # Load cubes for all 10 labels.
    for i in range(10):
        # Path to cubes with current label (ie number of side).
        name = f'{i:02d}'
        path = os.path.join(base, name)
        fnames = glob.glob(os.path.join(path, '*.jpg'))

        # Load all cubes and assign an alpha map. Skip cubes that are too dark
        # as they have no chance of being identified.
        out[name] = []
        for j, fname in enumerate(fnames[:N]):
            # Load the image and convert to NumPy array.
            img = Image.open(fname).convert('RGBA')
            if img.size != (width, height):
                img = img.resize((width, height))
            img = np.array(img, np.uint8)

            # Crappy threshold to remove images that are too dark.
            if np.sum(img[:, :, :3]) < thresh:
                continue

            # Compute alpha mask. Basically, we _hope_ that dark areas are
            # outside the cube.
            intensity = np.sum(np.array(img[:, :, :3], np.float32), axis=2)
            img[:, :, 3] = np.interp(intensity, [0, 20, 30, 255], [0, 25, 255, 255])

            # Store the image.
            out[name].append(img)

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


def stampImage(background, fg_shapes, N, xmin, xmax, ymin, ymax):
    """Return a new image with N shapes in it, and their BBox and labels.
    """
    # Background and shape images must be RGB.
    assert background.ndim == 3
    assert background.shape[2] == 3
    for fg in fg_shapes.values():
        assert fg.ndim == 4
        assert fg.shape[3] == 4

    # Convenience
    bg_height, bg_width = background.shape[:2]
    out = np.array(background, np.uint8)
    bboxes, labels = [], []
    stencil = np.zeros_like(out)

    # Stamp N non-overlapping shapes onto the background. If this proves
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
        bboxes.append([x0, y0, x1, y1])

        # Pick a random foreground label and specimen.
        label = random.choice(list(fg_shapes.keys()))
        labels.append(label)
        idx = np.random.randint(0, len(fg_shapes[label]) - 1)
        fg = np.array(fg_shapes[label][idx])

        # Scale the foreground image.
        fg = Image.fromarray(fg.astype(np.uint8)).resize((w, h), Image.BILINEAR)
        fg = np.array(fg, np.uint8)

        # Separate RGB from alpha and normalise alpha channel for blending.
        fg, alpha = fg[:, :, :3], fg[:, :, 3]
        alpha = np.array(alpha, np.float32) / 255
        alpha = np.stack([alpha, alpha, alpha], axis=2)

        # Stamp the foreground object into the background image.
        out[y0:y1, x0:x1] = (1 - alpha) * out[y0:y1, x0:x1] + alpha * fg
    return out, bboxes, labels


def generateImages(dst_path, bg_fnames, fg_shapes, int2name, num_img, num_stamps):
    """Create N stamped background images and save them."""
    xmax = max([_.shape[2] for _ in fg_shapes.values()])
    ymax = max([_.shape[1] for _ in fg_shapes.values()])
    xmin, ymin = int(0.95 * xmax), int(0.95 * ymax)
    dims = (xmin, xmax, ymin, ymax)

    # Create N images.
    for i in tqdm.tqdm(range(num_img)):
        # Load background image as NumPy array.
        img = Image.open(bg_fnames[i % len(bg_fnames)]).convert('RGB')
        img = np.array(img, np.uint8)

        # Stamp foreground shapes into background image.
        img, bboxes, labels = stampImage(img, fg_shapes, num_stamps, *dims)

        # File name prefix for image and meta data.
        fname = os.path.join(dst_path, f'{i:04d}')

        # Save meta data.
        meta = {'bboxes': bboxes, 'labels': labels, 'int2name': int2name}
        pickle.dump(meta, open(fname + '-meta.pickle', 'wb'))

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
    shapes = createForegroundShapes(N=10, width=32, height=32)
    shapes = loadForegroundShapes(N=10, width=32, height=32)

    # Compile a map to convert a label ID to its human readable name. Note that
    # this is 1-based because label 0 will always be reserved for the
    # background (ie no-object) label in later stages.
    int2name = {idx + 1: name for idx, name in enumerate(shapes)}
    int2name[0] = 'background'

    # Stamp the foreground objects into the background images.
    generateImages(dst_path, bg_fnames, shapes, int2name, num_img=10, num_stamps=20)


if __name__ == '__main__':
    main()
