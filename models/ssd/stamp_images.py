"""Load background images, stamp shapes into it and save them as new images.

Each image will be saved as JPG alongside a '*-meta.pickle' file. The meta file
contains information about the BBox and label of each object that was stamped
into the image.

"""
import os
import tqdm
import pickle
import random
import numpy as np

from PIL import Image


def stampImage(background, fg_shapes, N):
    """Return a new image with N shapes in it, and their BBox and labels.
    """
    # Background and shape images must be RGBA.
    assert background.ndim == 3
    assert background.shape[2] == 3
    for fg in fg_shapes.values():
        assert fg.ndim == 4
        assert fg.shape[3] == 4

    # Convenience
    bg_height, bg_width = background.shape[:2]
    out = np.array(background, np.uint8)
    bboxes, labels = [], []
    occupied = np.zeros_like(out)

    # Stamp N non-overlapping shapes onto the background. If this proves
    # difficult, abort after `max_attempts` and return what we have so far.
    attempts, max_attempts = 0, 10 * N
    while len(labels) < N and attempts < max_attempts:
        attempts += 1

        # Pick a random foreground label and specimen.
        label = random.choice(list(fg_shapes.keys()))
        idx = np.random.randint(0, len(fg_shapes[label]))
        fg = np.array(fg_shapes[label][idx])
        im_height, im_width = np.array(fg.shape[:2], np.float32)

        # Compute random region in foreground image to put the object.
        w, h = np.random.uniform(0.25, 1.0) * np.array([im_width, im_height])
        w, h = int(w), int(h)
        x0 = np.random.randint(0, 1 + bg_width - w)
        y0 = np.random.randint(0, 1 + bg_height - h)
        x1, y1 = x0 + w, y0 + h

        # Verify if the region is already occupied. Do nothing if it is.
        if np.sum(occupied[y0:y1, x0:x1]) > 0:
            continue
        labels.append(label)
        occupied[y0:y1, x0:x1] = 1
        bboxes.append([x0, y0, x1, y1])

        # Scale the foreground image.
        fg = Image.fromarray(fg.astype(np.uint8)).resize((w, h), Image.BILINEAR)
        fg = np.array(fg, np.uint8)

        # Separate RGB from alpha and normalise alpha channel for blending.
        fg, alpha = fg[:, :, :3], fg[:, :, 3]
        alpha = np.array(alpha, np.float32) / 255
        alpha = np.stack([alpha, alpha, alpha], axis=2)

        # Stamp the foreground object into the background image.
        out[y0:y1, x0:x1] = (1 - alpha) * out[y0:y1, x0:x1] + alpha * fg
    assert len(bboxes) == len(labels)
    return out, bboxes, labels


def generate(dst_path, param, bg_fnames, fg_shapes, int2name):
    """Create N stamped background images and save them."""
    # Create N images.
    print('Compiling training images with foreground shapes stamped into backgrounds')
    for i in tqdm.tqdm(range(param.N)):
        # Load background image as NumPy array.
        img = Image.open(bg_fnames[i % len(bg_fnames)]).convert('RGB')
        img = np.array(img, np.uint8)

        # Stamp foreground shapes into background image.
        img, bboxes, labels = stampImage(img, fg_shapes, param.num_stamps)

        # File name prefix for image and meta data.
        fname = os.path.join(dst_path, f'{i:04d}')

        # Save meta data.
        meta = {'bboxes': bboxes, 'labels': labels, 'int2name': int2name}
        pickle.dump(meta, open(fname + '-meta.pickle', 'wb'))

        # Save the stamped image.
        Image.fromarray(img).save(fname + '.jpg')
