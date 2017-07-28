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
import random
import pickle
import argparse
import multiprocessing
import feature_inspector

import numpy as np
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter


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


def ft2im(val, ft_dim: int, im_dim: int):
    """Return `val` in image coordinates.

    Inputs:
        val: float, Array
            The values to interpolate
        ft_dim: in
            Size of feature dimension
        im_dim: in
            Size of image dimension

    Returns:
        float, Array: Same size as `val`
    """
    assert ft_dim <= im_dim
    assert isinstance(ft_dim, int) and isinstance(im_dim, int)

    # Each point in feature coordinate corresponds to an area in image
    # coordinates. The `ofs` value here is to ensure that we hit the centre of
    # that area.
    ofs = (im_dim / ft_dim) / 2
    return np.interp(val, [0, ft_dim - 1], [ofs, im_dim - ofs - 1])


def oneHotEncoder(labels, num_classes):
    """Return one-hot-label encoding for `labels`.

    The hot-labels will be the first dimension. For instance, if the shape of
    `labels` is (3, 4) then the output shape will be (`num_classes`, 3, 4).

    Inputs:
        labels: Array
            Array can have any data type but its entries *must* be integers and
            in the interval [0, num_classes - 1].
        num_classes: int
            Must be positive.

    Returns:
        Array: (num_classes, *labels.shape)
    """
    # Labels must be an array with at least one element.
    assert np.prod(np.array(labels).shape) > 0

    # Must be non-negative 16 Bit integer.
    assert isinstance(num_classes, int)
    assert 0 < num_classes < 2 ** 16

    # All labels must be integers in [0, num_labels - 1]
    labels_f64 = np.array(labels, np.float64)
    labels_i64 = np.array(labels_f64, np.int64)
    assert np.sum(np.abs(labels_i64 - labels_f64)) == 0, 'Labels must be integers'
    assert 0 <= np.min(labels_i64) <= np.max(labels_i64) < num_classes
    del labels, labels_f64

    # Backup the input dimension and flatten the label array. This is necessary
    # for some NumPy tricks below to avoid loops.
    dim = labels_i64.shape
    labels = labels_i64.flatten()
    out = np.zeros((num_classes, len(labels)), np.uint16)

    # Compute the positions of the non-zero entries in the array and set them.
    ix = (labels, np.arange(len(labels)))
    out[ix] = 1

    # Reshape the hot-label data and return it.
    return out.reshape((num_classes, *dim))


def getNumClassesFromY(y_dim):
    """ Return the number of possible class that a `y_dim` tensoer can hold.

    This is a convenience function only to remove code duplication and hard
    coded magic number throughout the code base.

    Raise AssertionError if `y_dim` does not satisfy (1, >6, *, *).

    Inputs:
        y_dim: 4 Integers

    Returns:
        int: number of classes that can be one-hot encoded.
    """
    assert len(y_dim) == 4 and y_dim[0] == 1 and y_dim[1] > 4 + 2
    return y_dim[1] - 6


def sampleMasks(m_valid, m_isFg, m_bbox, m_cls, N):
    """ Return binary valued masks with valid locations for each type.

    All returned masks except the one for foreground/background estimation will
    have N active pixels (fewer if there were less than N to begin with).
    The FGBG will have up to 2N activate pixels, namely at most N pixels that
    are active over foreground and background, respectively.

    NOTE: all input masks must have the same 2D shape.

    Inputs:
        m_valid: 2D Array
            Indicate the locations to consider for the masks.
        m_isFg: 2D Array
            Indicate the locations with foreground objects.
        m_bbox: 2D Array
            Indicate the locations where BBox estimation is possible.
        m_cls: 2D Array
            Indicate the locations were class label estimation is possible.
        N: int
            Number of locations to sample for each mask.

    Returns:
        mask_bbox: 2D Array
            BBox mask with up to N active pixels.
        mask_fgbg: 2D Array
            Fg/Bg mask with up to 2N active pixels. Of these 2N pixels, up to N
            will mark a foreground location, and up to another N a background
            location.
        mask_cls: 2D Array
            Mask with up to N active pixels where estimating the foreground
            class (eg cube number) is possible.
    """
    assert N > 0
    assert m_valid.ndim == 2
    assert m_valid.shape == m_isFg.shape == m_bbox.shape == m_cls.shape

    # Backup the input dimension because we will flatten the arrays afterwards
    # since it is easier to work with vectors.
    dim = m_valid.shape
    out_fgbg = np.zeros_like(m_isFg).flatten()
    out_bbox = np.zeros_like(m_bbox).flatten()
    out_cls = np.zeros_like(m_cls).flatten()

    # Remove all the locations prohibited by the 'valid' mask.
    bbox = ((m_valid * m_bbox).flatten())
    labl = ((m_valid * m_cls).flatten())
    isFg = ((m_valid * m_isFg).flatten())
    isBg = ((m_valid * (1 - m_isFg)).flatten())

    # BBox: sample subset of the valid positions.
    ix_bbox = np.nonzero(bbox)[0].tolist()
    ix_bbox = ix_bbox if len(ix_bbox) <= N else random.sample(ix_bbox, N)
    out_bbox[ix_bbox] = 1

    # Class: sample subset of the valid positions.
    ix_cls = np.nonzero(labl)[0].tolist()
    ix_cls = ix_cls if len(ix_cls) <= N else random.sample(ix_cls, N)
    out_cls[ix_cls] = 1

    # Foreground, background: need to find N valid foreground locations, and
    # another N valid background locations.
    ix_fg = np.nonzero(isFg)[0].tolist()
    ix_bg = np.nonzero(isBg)[0].tolist()
    ix_fg = ix_fg if len(ix_fg) <= N else random.sample(ix_fg, N)
    ix_bg = ix_bg if len(ix_bg) <= N else random.sample(ix_bg, N)
    out_fgbg[ix_fg] = 1
    out_fgbg[ix_bg] = 1

    # Restore the original matrix shapes and return them.
    out_bbox = out_bbox.reshape(dim)
    out_fgbg = out_fgbg.reshape(dim)
    out_cls = out_cls.reshape(dim)
    return out_bbox, out_fgbg, out_cls


def setBBoxRects(y, val):
    y = np.array(y)
    assert y.ndim == 3
    assert np.array(val).shape == y[:4].shape
    y[:4] = val
    return y


def getBBoxRects(y):
    return y[:4]


def setIsFg(y, val):
    y = np.array(y)
    assert y.ndim == 3
    assert np.array(val).shape == y[4:6].shape
    y[4:6] = val
    return y


def getIsFg(y):
    return y[4:6]


def setClassLabel(y, val):
    y = np.array(y)
    assert y.ndim == 3
    assert np.array(val).shape == y[6:].shape
    y[6:] = val
    return y


def getClassLabel(y):
    return y[6:]


def downsampleMatrix(mat, ft_dim):
    if not isinstance(mat, np.ndarray):
        mat = np.array(mat)
    x = np.linspace(0, mat.shape[1] - 1, ft_dim[1])
    y = np.linspace(0, mat.shape[0] - 1, ft_dim[0])
    x = np.round(x).astype(np.int64)
    y = np.round(y).astype(np.int64)
    return mat[y][:, x]


def unpackBBoxes(im_dim, bb_rects, bb_labels):
    ft_dim = bb_labels.shape[:2]
    assert bb_rects.shape == (4, *ft_dim)

    # Find all locations that are *not* background, ie every location where the
    # predicted label is anything but zero.
    pick_yx = np.nonzero(bb_labels)

    # Convert the picked locations from feature- to image dimensions.
    anchor_x = ft2im(pick_yx[1], ft_dim[1], im_dim[1])
    anchor_y = ft2im(pick_yx[0], ft_dim[0], im_dim[0])

    # Pick the labels and BBox parameters from the valid locations.
    x0 = bb_rects[0][pick_yx] + anchor_x
    y0 = bb_rects[1][pick_yx] + anchor_y
    x1 = bb_rects[2][pick_yx] + anchor_x
    y1 = bb_rects[3][pick_yx] + anchor_y

    # Ensure the BBoxes are confined to the image.
    x0 = np.clip(x0, 0, im_dim[1] - 1)
    x1 = np.clip(x1, 0, im_dim[1] - 1)
    y0 = np.clip(y0, 0, im_dim[0] - 1)
    y1 = np.clip(y1, 0, im_dim[0] - 1)

    # Stack the BBox data in the format: label, x0, y0, width, heigth. Return
    # it as a Python set to remove the many duplicates.
    bboxes = np.vstack([x0, y0, x1, y1]).T.astype(np.int16)
    assert bboxes.shape[0] == len(pick_yx[0])
    return bboxes, pick_yx


def _computeBBoxes(bb_data, objID_at_pixel_ft, ft_dim, im_dim):
    # Find all feature map locations that show anything but background.
    fg_idx = np.nonzero(objID_at_pixel_ft)

    # Convert the absolute BBox corners to relative values with respect to
    # the anchor point (all in image coordinates).
    bboxes = np.zeros((4, *ft_dim), np.float32)
    for y, x in zip(*fg_idx):
        objID = objID_at_pixel_ft[y, x]
        anchor_x = ft2im(x, ft_dim[1], im_dim[1])
        anchor_y = ft2im(y, ft_dim[0], im_dim[0])
        x0, y0, x1, y1 = bb_data[objID]['bbox']
        x0 = x0 - anchor_x
        x1 = x1 - anchor_x
        y0 = y0 - anchor_y
        y1 = y1 - anchor_y
        bboxes[:, y, x] = (x0, y0, x1, y1)
    return bboxes


def _maskFgBg(objID_at_pixel_ft):
    """Return the "this-is-not-a-background-pixel" mask.
    """
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
        idx_visible = np.nonzero(objID_at_pixel_ft == objID)

        # Are enough pixels visible in the scene.
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

    out = {}
    # Load the True output and verify that all files use the same
    # int->label mapping.
    img_meta = bz2.open(fname + '-meta.json.bz2', 'rb').read()
    img_meta = json.loads(img_meta.decode('utf8'))
    out['int2name'] = {int(k): v for k, v in img_meta['int2name'].items()}

    # Undo JSON's int->str conversion for dict keys.
    bb_data = {int(k): v for k, v in img_meta['bb_data'].items()}
    obj_pixels = {int(k): v for k, v in img_meta['obj-pixels'].items()}
    objID2label = {int(k): v for k, v in img_meta['objID2label'].items()}
    objID_at_pixel = np.array(img_meta['objID-at-pixel'], np.int32)
    del img_meta

    # For each non-zero pixel, map the object ID to its label. This
    # will produce an image where each pixel corresponds to a label
    # that can be looked up with `int2name`.
    label_at_pixel = np.zeros_like(objID_at_pixel)
    for idx in zip(*np.nonzero(objID_at_pixel)):
        label_at_pixel[idx] = objID2label[objID_at_pixel[idx]]

    # Compile dictionary with feature size specific data. This includes the
    # BBox data relative to the anchor point.
    for ft_dim in rpcn_dims:
        img_ft = Image.fromarray(img).resize((ft_dim[1], ft_dim[0]))
        img_ft = np.array(img_ft)

        # Downsample the label/objID maps to the feature size.
        label_at_pixel_ft = downsampleMatrix(label_at_pixel, ft_dim)
        objID_at_pixel_ft = downsampleMatrix(objID_at_pixel, ft_dim)
        obj_pixels_ft = {k: downsampleMatrix(v, ft_dim) for k, v in obj_pixels.items()}

        bboxes = _computeBBoxes(bb_data, objID_at_pixel_ft, ft_dim, im_dim)
        mask_fgbg = _maskFgBg(objID_at_pixel_ft)
        mask_bbox = _maskBBox(objID_at_pixel_ft, obj_pixels_ft)
        mask_valid = _maskValid(objID_at_pixel_ft)
        mask_fg_label = _maskFgLabel(img_ft, objID_at_pixel_ft, obj_pixels_ft)

        # Compile all the information into the output dictionary.
        out[ft_dim] = {
            'bboxes': np.array(bboxes, np.float32),
            'objID_at_pixel': objID_at_pixel_ft,
            'label_at_pixel': label_at_pixel_ft,
            'mask_fgbg': mask_fgbg,
            'mask_bbox': mask_bbox,
            'mask_fg_label': mask_fg_label,
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
