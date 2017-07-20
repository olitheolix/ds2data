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
    bb_rects, bb_labels = [], []
    occupied = np.zeros_like(out)

    # Stamp N non-overlapping shapes onto the background. If this proves
    # difficult, abort after `max_attempts` and return what we have so far.
    attempts, max_attempts = 0, 20 * N
    while len(bb_labels) < N and attempts < max_attempts:
        attempts += 1

        # Pick a random foreground label and specimen.
        label = random.choice(list(fg_shapes.keys()))
        idx = np.random.randint(0, len(fg_shapes[label]))
        fg = np.array(fg_shapes[label][idx])
        im_height, im_width = np.array(fg.shape[:2], np.float32)

        # Randomly scale the shape. The scaling factors are drawn from a
        # non-uniform distribution such that most shapes have "medium" size.
        # For instance,if the input shapes were 256x256 then most shapes would
        # have a size in between 64x64 and 192x192 pixels
        r = np.random.uniform(0.25, 1.0)
        scale = np.interp(r, [0, 0.25, 0.85, 1.0], [0.12, 0.25, 0.75, 1.0])
        w, h = scale * np.array([im_width, im_height])
        w, h = int(w), int(h)

        # Compute random region in foreground image to put the object.
        x0 = np.random.randint(0, bg_width - w - 1)
        y0 = np.random.randint(0, bg_height - h - 1)
        x1, y1 = x0 + w, y0 + h
        assert 0 <= x0 < x1 < bg_width
        assert 0 <= y0 < y1 < bg_height

        # Verify if the region is already occupied. Do nothing if it is.
        if np.sum(occupied[y0:y1, x0:x1]) > 0:
            continue
        bb_labels.append(label)
        occupied[y0:y1, x0:x1] = 1
        bb_rects.append([x0, y0, x1, y1])

        # Scale the foreground image.
        fg = Image.fromarray(fg.astype(np.uint8)).resize((w, h), Image.BILINEAR)
        fg = np.array(fg, np.uint8)

        # Separate RGB from alpha and normalise alpha channel for blending.
        fg, alpha = fg[:, :, :3], fg[:, :, 3]
        alpha = np.array(alpha, np.float32) / 255
        alpha = np.stack([alpha, alpha, alpha], axis=2)

        # Stamp the foreground object into the background image.
        out[y0:y1, x0:x1] = (1 - alpha) * out[y0:y1, x0:x1] + alpha * fg
    assert len(bb_rects) == len(bb_labels)
    return out, bb_rects, bb_labels


def generate(dst_path, param, bg_fnames, fg_shapes, int2name):
    """Create N stamped background images and save them."""
    # Create N images.
    print('Stamping foreground shapes into background images')
    for i in tqdm.tqdm(range(param.N)):
        # Load background image as NumPy array.
        img = Image.open(bg_fnames[i % len(bg_fnames)]).convert('RGB')
        img = np.array(img, np.uint8)

        # Stamp foreground shapes into background image.
        img, bb_rects, bb_labels = stampImage(img, fg_shapes, param.num_stamps)

        # File name prefix for image and meta data.
        fname = os.path.join(dst_path, f'{i:04d}')

        # Save meta data.
        meta = {
            'bb_rects': bb_rects, 'labels': bb_labels,
            'int2name': int2name, 'param': param,
        }
        pickle.dump(meta, open(fname + '-meta.pickle', 'wb'))

        # Save the stamped image.
        Image.fromarray(img).save(fname + '.jpg')
