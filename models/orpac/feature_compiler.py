"""Compile BBox position from meta file into training vector.

The training output `y` is a feature map with 5 features: label, BBox centre
relative to anchor, and BBox absolute width/height.

The label values, ie the entries in y[0, :, :], are non-negative integers. A
label of zero always means background.
"""
import os
import bz2
import sys
import glob
import tqdm
import json
import pickle
import argparse
import multiprocessing
import feature_inspector

import numpy as np
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter
from feature_utils import ft2im, downsampleMatrix


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(description='Compile training data')
    parser.add_argument(
        'path', nargs='?', type=str,
        metavar='File or Path with training images and *-meta.json.bz2',
        help='')
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='Create debug plots for instant inspection')

    param = parser.parse_args()
    if param.path is None:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        param.path = os.path.join(cur_dir, 'data', '3dflight')

    if not os.path.exists(param.path):
        print(f'Error: cannot open <{param.path}>')
        sys.exit(1)

    if os.path.isdir(param.path):
        fnames = glob.glob(os.path.join(param.path, '*.jpg'))
    else:
        fnames = [param.path]

    param.fnames = [_[:-4] for _ in sorted(fnames)]
    return param


def _computeBBoxes(bb_rects, objID_at_pixel_ft, im_dim):
    # Find all feature map locations that show anything but background.
    fg_idx = np.nonzero(objID_at_pixel_ft)
    ft_dim = objID_at_pixel_ft.shape

    # Convert the absolute BBox corners to relative values with respect to
    # the anchor point (all in image coordinates).
    bboxes = np.zeros((4, *ft_dim), np.float32)
    for y, x in zip(*fg_idx):
        objID = objID_at_pixel_ft[y, x]
        anchor_x = ft2im(x, ft_dim[1], im_dim[1])
        anchor_y = ft2im(y, ft_dim[0], im_dim[0])
        x0, y0, x1, y1 = bb_rects[objID]
        x0 = x0 - anchor_x
        x1 = x1 - anchor_x
        y0 = y0 - anchor_y
        y1 = y1 - anchor_y
        bboxes[:, y, x] = (x0, y0, x1, y1)
    return bboxes


def _maskForeground(objID_at_pixel_ft):
    """Return the "this-is-not-a-background-pixel" mask."""
    mask = np.zeros(objID_at_pixel_ft.shape, np.uint8)

    # Activate all feature map locations that show anything but background.
    fg_idx = np.nonzero(objID_at_pixel_ft)
    mask[fg_idx] = 1
    return mask


def _maskBBox(objID_at_pixel_ft, obj_pixels_ft):
    """Return the "you-can-estimate-bbox-size-at-this-anchor" mask.

    To estimating the BBox it often suffices to see only a small portion of the
    object. In this case, all objects that have more than 10% of its pixels
    visible are considered "visible enough for BBox estimation". NOTE: just
    because it is possible to estimate the BBox size does *not* mean it is also
    possible to estimate its label (ie the number on the cube), as considerably
    more pixels may have to be visible for that (see `_maskFgLabel`).
    """
    # Iterate over each object and determine (mostly guess) if it is possible
    # to recognise the object.
    mask = np.zeros(objID_at_pixel_ft.shape, np.uint8)
    for objID, pixels in obj_pixels_ft.items():
        # Skip background locations.
        if objID == 0:
            continue

        # Find out which pixels belong to the current object AND are visible in
        # the scene right now.
        idx_visible = np.nonzero(objID_at_pixel_ft == objID)

        # We decide that BBox estimation is possible if at least 10% of all
        # pixels for the object are visible.
        num_visible = len(idx_visible[0])
        num_tot = np.count_nonzero(pixels)
        if num_visible >= 9 and num_visible >= 0.1 * num_tot:
            mask[idx_visible] = 1
    return mask


def _maskValid(objID_at_pixels_ft):
    """Return the "only-train-on-these-pixels" mask.

    The main purpose of this mask is to remove all pixels close to
    foreground/background boundaries. This will avoid anchor positions where
    the foreground/background label is ambiguous.
    """
    src = np.array(objID_at_pixels_ft, np.uint8)
    out = np.array(Image.fromarray(src).filter(ImageFilter.FIND_EDGES))

    mask = np.zeros(src.shape, np.uint8)
    mask[np.nonzero(out == 0)] = 1
    return mask


def _maskFgLabel(img, objID_at_pixel_ft, obj_pixels_ft):
    """Return the "it is possible to estimate the label at that pixel" mask.

    To estimate the label the object must be
      1. large enough to see the number
      2. bright enough to see the number
      3. unobstructed enough to see the number
      4. not too so close to the screen edge that the number is clipped

    For Condition 1, objects are large enough if it occupies least 9 pixels in
    feature space. 9 pixels (ie a 3x3 patch) corresponds to a 24x24 receptive
    field for our default feature and image sizes of 64x64 and 512x512,
    respectively.

    Objects are bright enough if their average pixel value is at least 40. I
    determine this threshold with an empirical study of one image :)

    We consider the object unobstructed if at least 50% of its pixels are
    actually visible in the scene (Condition 3).

    I do not know how to recognise Condition 4 with the given data.

    This is not foolproof but works well enough for now. A notable case where
    this fails is a cube close to the camera but clipped by the image boundary.
    These cubes may occupy a lot of screen real estate, yet its only visible
    portion is the red frame and parts of the white surface.
    """
    # Compute the average pixel intensity.
    img = np.mean(np.array(img, np.int64), axis=2)
    assert img.shape == objID_at_pixel_ft.shape

    # Iterate over each object and determine (mostly guess) if it is possible
    # to recognise the object.
    mask = np.zeros(objID_at_pixel_ft.shape, np.uint8)
    for objID, pixels in obj_pixels_ft.items():
        idx_visible = np.nonzero(objID_at_pixel_ft == objID)
        if len(idx_visible[0]) == 0:
            continue

        # Is it bright enough.
        avg_brightness = np.mean(img[idx_visible])
        if avg_brightness < 40:
            continue

        # Are enough pixels visible in the scene.
        num_visible = len(idx_visible[0])
        num_tot = np.count_nonzero(pixels)
        if num_visible >= 9 and num_visible >= 0.5 * num_tot:
            mask[idx_visible] = 1
    return mask


def compileFeatures(fname, img, rpcn_dims):
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    im_dim = img.shape[:2]

    # Load the True output and verify that all files use the same
    # int->label mapping.
    img_meta = bz2.open(fname + '-meta.json.bz2', 'rb').read()
    img_meta = json.loads(img_meta.decode('utf8'))

    # Undo JSON's int->str conversion for dict keys.
    int2name = {int(k): v for k, v in img_meta['int2name'].items()}
    bb_rects = {int(k): v for k, v in img_meta['bb_rects'].items()}
    obj_pixels = {int(k): v for k, v in img_meta['obj-pixels'].items()}
    objID2label = {int(k): v for k, v in img_meta['objID2label'].items()}
    objID_at_pixel = np.array(img_meta['objID-at-pixel'], np.int32)
    del img_meta

    # The label map *must* contain a None labels -> these are the background
    # pixels.
    assert int2name[0] == 'None'
    name2int = {v: k for k, v in int2name.items()}

    # For each non-zero pixel, map the object ID to its label. This
    # will produce an image where each pixel corresponds to a label
    # that can be looked up with `int2name`.
    label_at_pixel = np.zeros_like(objID_at_pixel)
    for idx in zip(*np.nonzero(objID_at_pixel)):
        label_name = objID2label[objID_at_pixel[idx]]
        assert label_name != 'None'
        label_at_pixel[idx] = name2int[label_name]

    # Add the int2name map to the function output.
    out = {}
    out['int2name'] = int2name

    # Compile dictionary with feature size specific data. This includes the
    # BBox data relative to the anchor point.
    for ft_dim in rpcn_dims:
        img_ft = Image.fromarray(img).resize((ft_dim[1], ft_dim[0]))
        img_ft = np.array(img_ft)

        # Downsample the label/objID maps to the feature size.
        label_at_pixel_ft = downsampleMatrix(label_at_pixel, ft_dim)
        objID_at_pixel_ft = downsampleMatrix(objID_at_pixel, ft_dim)
        obj_pixels_ft = {k: downsampleMatrix(v, ft_dim) for k, v in obj_pixels.items()}

        bboxes = _computeBBoxes(bb_rects, objID_at_pixel_ft, im_dim)
        mask_fg = _maskForeground(objID_at_pixel_ft)
        mask_bbox = _maskBBox(objID_at_pixel_ft, obj_pixels_ft)
        mask_valid = _maskValid(objID_at_pixel_ft)
        mask_cls = _maskFgLabel(img_ft, objID_at_pixel_ft, obj_pixels_ft)

        # Compile all the information into the output dictionary.
        out[ft_dim] = {
            'bboxes': np.array(bboxes, np.float32),
            'objID_at_pixel': objID_at_pixel_ft,
            'label_at_pixel': label_at_pixel_ft,
            'mask_fg': mask_fg,
            'mask_bbox': mask_bbox,
            'mask_cls': mask_cls,
            'mask_valid': mask_valid,
        }
    return out


def compileSingle(args):
    fname, rpcn_out_dims = args
    img = np.array(Image.open(fname + '.jpg').convert('RGB'))
    features = compileFeatures(fname, img, rpcn_out_dims)
    pickle.dump(features, open(fname + '-compiled.pickle', 'wb'))


def main():
    param = parseCmdline()
    rpcn_out_dims = [(64, 64), (32, 32)]

    args = [(_, rpcn_out_dims) for _ in param.fnames]

    if len(args) == 1:
        compileSingle(args[0])
    else:
        with multiprocessing.Pool() as pool:
            # Setup parallel execution and wrap it into a TQDM progress bar. Then
            # consume the iterator.
            progbar = tqdm.tqdm(
                pool.imap_unordered(compileSingle, args),
                total=len(args), desc='Compiling Features', leave=False
            )
            [_ for _ in progbar]

    # Show debug plots for the first file in the list.
    if param.debug:
        feature_inspector.main(param.fnames[0] + '.jpg')


if __name__ == '__main__':
    main()
