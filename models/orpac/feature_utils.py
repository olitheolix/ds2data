import random
import numpy as np


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
    """ Return the number of possible class that a `y_dim` tensor can hold.

    This is a convenience function only to remove code duplication and hard
    coded magic number throughout the code base.

    Raise AssertionError if `y_dim` does not satisfy (>6, *, *).

    Inputs:
        y_dim: 3 Integers

    Returns:
        int: number of classes that can be one-hot encoded.
    """
    assert len(y_dim) == 3 and y_dim[0] > 4 + 2
    return y_dim[0] - 6


def sampleMasks(m_valid, m_isFg, m_bbox, m_cls, m_id, N):
    """Return binary valued masks with valid locations for each type.

    Activate N locations for each object. If the object does not occupy N
    pixels then activate all that it occupies.

    In an ideal scenario where all distinct objects occupy at least N pixels,
    the BBox and Label masks will have "N * num_objects" activate pixels, and
    the BGFG mask "2 * N * num_objects".

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
            Visible objID at each pixel.
        m_id: 2D Array
            Object ID at each pixel. The IDs themselves are meaningless but
            denoted distinct elements in the render engine and will be used
            here to sample locations that involve all objects equally to the
            extent possible.
        N: int
            Number of locations to sample for each mask.

    Returns:
        mask_bbox: 2D Array
            BBox mask with up to N active pixels.
        mask_bgfg: 2D Array
            Fg/Bg mask with up to 2N active pixels. Of these 2N pixels, up to N
            will mark a foreground location, and up to another N a background
            location.
        mask_cls: 2D Array
            Mask with up to N active pixels where estimating the foreground
            class (eg cube number) is possible.

    """
    assert N > 0
    assert m_valid.ndim == 2
    assert m_valid.shape == m_isFg.shape == m_bbox.shape == m_cls.shape == m_id.shape

    # Backup the input dimension because we will flatten the arrays afterwards
    # since it is easier to work with vectors.
    dim = m_valid.shape
    out_bgfg = np.zeros_like(m_isFg).flatten()
    out_bbox = np.zeros_like(m_bbox).flatten()
    out_cls = np.zeros_like(m_cls).flatten()

    # Create the masks for all our cases.
    bbox = (m_valid * m_id * m_bbox).flatten()
    labl = (m_valid * m_id * m_cls).flatten()
    isFg = (m_valid * m_id * m_isFg).flatten()
    isBg = (m_valid * (m_id == 0) * (m_isFg == 0)).flatten()

    # Sample BBox locations for each object ID.
    for objID in np.unique(bbox):
        if objID != 0:
            idx = np.nonzero(bbox == objID)[0].tolist()
            idx = random.sample(idx, N) if len(idx) > N else idx
            out_bbox[idx] = 1
    del bbox

    # Sample Label locations for each object ID.
    for objID in np.unique(labl):
        if objID != 0:
            idx = np.nonzero(labl == objID)[0].tolist()
            idx = random.sample(idx, N) if len(idx) > N else idx
            out_cls[idx] = 1
    del labl

    # Sample Foreground locations for each object ID.
    for objID in np.unique(isFg):
        if objID != 0:
            idx = np.nonzero(isFg == objID)[0].tolist()
            idx = random.sample(idx, N) if len(idx) > N else idx
            out_bgfg[idx] = 1

    # Sample Background locations and add them to the `out_bgfg` array.
    num_fg = np.count_nonzero(out_bgfg)
    idx = np.nonzero(isBg)[0].tolist()
    idx = random.sample(idx, num_fg) if len(idx) > num_fg else idx
    out_bgfg[idx] = 1

    # Restore the original matrix shapes and return them.
    out_bbox = out_bbox.reshape(dim)
    out_bgfg = out_bgfg.reshape(dim)
    out_cls = out_cls.reshape(dim)
    return out_bbox, out_bgfg, out_cls


def setBBoxRects(y, val):
    y = np.array(y)
    assert y.ndim == 3
    assert np.array(val).shape == y[:4].shape
    y[:4] = val
    return y


def getBBoxRects(y):
    assert y.ndim == 3
    return y[:4]


def setIsFg(y, val):
    y = np.array(y)
    assert y.ndim == 3
    assert np.array(val).shape == y[4:6].shape
    y[4:6] = val
    return y


def getIsFg(y):
    assert y.ndim == 3
    return y[4:6]


def setClassLabel(y, val):
    y = np.array(y)
    assert y.ndim == 3
    assert np.array(val).shape == y[6:].shape
    y[6:] = val
    return y


def getClassLabel(y):
    assert y.ndim == 3
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
    assert np.array(bb_labels).ndim == 2
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
