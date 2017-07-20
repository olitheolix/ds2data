"""Compile BBox position from meta file into training vector.

The training output `y` is a feature map with 5 features: label, BBox centre
relative to anchor, and BBox absolute width/height.

The label values, ie the entries in y[0, :, :], are non-negative integers. A
label of zero always means background.
"""
import os
import glob
import tqdm
import pickle
import multiprocessing
import scipy.signal
import matplotlib.pyplot as plt

import numpy as np
import PIL.Image as Image


def computeOverlapScore(img_dim_hw, bb_rects):
    assert bb_rects.shape[1] == 4

    # Find out where the anchor box will overlap with each BBox. To do
    # this by stamping a block of 1's into the image and convolve it
    # with the anchor box.
    bb_score = np.zeros((len(bb_rects), *img_dim_hw), np.float16)
    for i, (x0, y0, x1, y1) in enumerate(bb_rects):
        # BBox size in pixels.
        bbox_area = (x1 - x0) * (y1 - y0)
        assert bbox_area > 0

        # Stamp a BBox sized region into the otherwise empty image. This
        # "box" is what we will convolve with the anchor to compute the
        # overlap.
        bb_score[i, y0:y1, x0:x1] = 1
        anchor = np.ones((y1 - y0, x1 - x0), np.float32)

        # Convolve the BBox with the anchor box. The FFT version is much
        # faster but introduces numerical artefacts which we must remove.
        # Fortunately, this is easy because we know the correct convolution
        # result contain real valued integers since both input signals also
        # only contained real valued integers.
        overlap = scipy.signal.fftconvolve(bb_score[i], anchor, mode='same')
        overlap = np.round(np.abs(overlap))

        # To compute the ratio of overlap we need to know which box (BBox
        # or anchor) is smaller. We need this because if one box is fully
        # inside the other we would like the overlap metric to be 1.0 (ie
        # 100%), and the convolution at that point will be identical to the
        # area of the smaller box. Therefore, find out which box has the
        # smaller area. Then compute the overlap ratio.
        bb_score[i] = overlap / bbox_area
        del i, x0, x1, y0, y1, overlap, bbox_area
    return bb_score


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


def genBBoxData(bb_rects, bb_labels, bb_scores, ft_dim, thresh):
    """Compute BBox training output for network.

    Returns a 3D array which, for each location, contains the label and BBox
    parameters. If the label is Zero then the BBox parameters are meaningless
    because it means the anchor is over the background.

    The returned BBox parameters are in image coordinates.

    Inputs:
        bb_rects: Array[N, 4]
            List of N BBox rectangles (x0, y0, x1, y1)
        bb_labels: Array[N]
            List of N machine readable labels (ie integers).
        bb_scores: Array[N, im_height, im_width]
            A score for each BBox at every image position.
        ft_dim: tuple(ft_height, ft_width)
            Feature map dimension
        thresh: 0.0 <= thresh <= 1.0
            Only those locations with a BBox score greater than `thresh` will
            be considered as having an object at that location.

    Output:
        Array[5, ft_height, ft_width]: BBox and label
            The 5 elements are [label_int, rel_x, rel_y, rel_w, rel_h] where
            `rel_{x,y}` is the BBox position relative to the anchor.
            `rel_{w,h}` denote the absolute size of the BBox. All 4 BBox
            parameters are in image coordinates.
    """
    # The first score dimension enumerates BBox, *not* labels.
    assert 0 <= thresh <= 1.0
    assert len(bb_labels) == len(bb_rects) == len(bb_scores)

    # Unpack dimension.
    ft_height, ft_width = ft_dim
    im_height, im_width = bb_scores.shape[1:]

    # Uncertainty when mapping from feature -> image dimensions. We will need
    # this to search a reasonable neighbourhood laer.
    ofs_x = int(1 + (im_width / ft_width) / 2)
    ofs_y = int(1 + (im_height / ft_height) / 2)

    out = np.zeros((5, *ft_dim), np.float16)
    for fy in range(ft_height):
        for fx in range(ft_width):
            # Convert the current feature coordinates to image coordinates. This
            # is the centre of the anchor box in image coordinates.
            anchor_x = ft2im(fx, ft_width, im_width)
            anchor_y = ft2im(fy, ft_height, im_height)
            anchor_x = int(np.round(anchor_x))
            anchor_y = int(np.round(anchor_y))

            # Extract the anchor neighbourhood in the score-map.
            x0, x1 = anchor_x - ofs_x, anchor_x + ofs_x
            y0, y1 = anchor_y - ofs_y, anchor_y + ofs_y
            x0, x1 = np.clip([x0, x1], 0, im_width - 1)
            y0, y1 = np.clip([y0, y1], 0, im_height - 1)
            neigh = bb_scores[:, y0:y1, x0:x1]
            del x0, x1, y0, y1

            # Skip this location if the neighbourhood is devoid of any BBoxes
            # that exceed the score.
            if np.amax(neigh) <= thresh:
                continue

            # Determine which BBox has the best score.
            neigh = np.amax(neigh, axis=(1, 2))
            best_idx = np.argmax(neigh)
            assert 0 <= best_idx < len(bb_rects)

            # Unpack the label and parameters of the best BBox.
            label_int = bb_labels[best_idx]
            bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bb_rects[best_idx]
            assert 0 <= bbox_x0 < bbox_x1 < im_width
            assert 0 <= bbox_y0 < bbox_y1 < im_height

            # Compute the BBox location, width and height relative to the
            # anchor (image coordinates).
            rel_x = np.mean([bbox_x0, bbox_x1]) - anchor_x
            rel_y = np.mean([bbox_y0, bbox_y1]) - anchor_y
            rel_w = bbox_x1 - bbox_x0
            rel_h = bbox_y1 - bbox_y0

            # Insert the BBox parameters into the training vector at the
            # respective image position.
            out[:, fy, fx] = [label_int, rel_x, rel_y, rel_w, rel_h]
    return out


def bboxFromNetOutput(im_dim, bb_rects, bb_labels):
    assert bb_rects.ndim == 3
    assert bb_rects.shape[0] == 4
    assert bb_labels.ndim == 2

    # Compute the ratio of feature/image dimension. From this, determine the
    # interpolation parameters to map feature locations to image locations.
    im_height, im_width = im_dim
    ft_height, ft_width = bb_labels.shape[:2]

    # Find all locations that are *not* background, ie every location where the
    # predicted label is anything but zero.
    pick_yx = np.nonzero(bb_labels)

    # Convert the picked locations from feature- to image dimensions.
    anchor_x = ft2im(pick_yx[1], ft_width, im_width)
    anchor_y = ft2im(pick_yx[0], ft_height, im_height)

    # Pick the labels and BBox parameters from the valid locations.
    x = bb_rects[0][pick_yx]
    y = bb_rects[1][pick_yx]
    w = bb_rects[2][pick_yx]
    h = bb_rects[3][pick_yx]

    # Convert the BBox centre positions, which are still relative to the
    # anchor, to absolute positions in image coordinates.
    x = x + anchor_x
    y = y + anchor_y

    # Compute BBox corners.
    x0 = x - w / 2
    x1 = x + w / 2
    y0 = y - h / 2
    y1 = y + h / 2

    # Ensure the BBoxes are confined to the image.
    x0 = np.clip(x0, 0, im_width - 1)
    x1 = np.clip(x1, 0, im_width - 1)
    y0 = np.clip(y0, 0, im_height - 1)
    y1 = np.clip(y1, 0, im_height - 1)

    # Stack the BBox parameters and labels and return it.
    bb_rect = np.vstack([x0, y0, x1, y1]).T.astype(np.int16)

    return bb_rect, pick_yx


def showBBoxData(img, y_bbox, y_score):
    assert img.ndim == 3
    assert img.shape[2] == 3

    # Convert the training output to BBox positions.
    labels, bb_xywh = y_bbox[0], y_bbox[1:]
    bb_rect, pick_yx = bboxFromNetOutput(img.shape[:2], bb_xywh, labels)
    bb_labels = labels[pick_yx]

    # Insert the BBox rectangle into the image.
    img_bbox = np.array(img)
    for label, (x0, y0, x1, y1) in zip(bb_labels, bb_rect):
        img_bbox[y0:y1, x0, :] = 255
        img_bbox[y0:y1, x1, :] = 255
        img_bbox[y0, x0:x1, :] = 255
        img_bbox[y1, x0:x1, :] = 255

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(img_bbox)
    plt.title('BBoxes')

    plt.subplot(2, 2, 3)
    plt.imshow(labels.astype(np.float32))
    plt.title(f'GT Label {labels.shape[1]}x{labels.shape[0]}')

    score = np.amax(y_score, axis=0).astype(np.float32)
    plt.subplot(2, 2, 4)
    plt.imshow(score, cmap='hot')
    plt.title(f'GT Score {score.shape[1]}x{score.shape[0]}')


def compileBBoxData(args):
    # Unpack arguments. This is necessary because this function can only have a
    # single argument since it will be called via a process pool.
    fname, num_pools, thresh = args

    # Load meta data and the image, then convert the image to CHW.
    meta = pickle.load(open(fname + '-meta.pickle', 'rb'))
    img = np.array(Image.open(fname + '.jpg', 'r').convert('RGB'), np.uint8)
    img = np.transpose(img, [2, 0, 1])
    im_dim = img.shape[1:]

    # Unpack the BBox data and map the human readable labels to numeric ones.
    bb_rect = np.array(meta['bb_rects'], np.int32)
    name2int = {v: k for k, v in meta['int2name'].items()}
    bb_labels = [name2int[_] for _ in meta['labels']]

    # Compute the score map for each individual bounding box.
    y_score = computeOverlapScore(im_dim, bb_rect)
    assert y_score.shape == (bb_rect.shape[0], *im_dim)

    # Determine the image- and feature dimensions.
    out_bbox = {}
    while True:
        ft_dim = (np.array(im_dim) / 2 ** num_pools).astype(np.int32).tolist()
        num_pools += 1
        if min(ft_dim) < 2:
            break

        y_bbox = genBBoxData(bb_rect, bb_labels, y_score, ft_dim, thresh)
        assert y_bbox.shape == (5, *ft_dim)
        out_bbox[tuple(ft_dim)] = y_bbox

    # Save the pickled ground truth training data.
    pickle.dump({'y_bbox': out_bbox}, open(fname + '-bbox.pickle', 'wb'))
    return img, out_bbox, y_score


def generate(path, thresh, num_pools, debug):
    """
    Produce the expected network output for each image based on BBoxes.

    Produce one training output for every image in `path`. Write the pickled
    results to the corresponding '*-bbox.pickle' file.

    Args:
        path: str
           Path to image/meta files
        thresh: float
           Must be in interval [0, 1]. If BBox overlaps more than `thresh` with
           anchor then the location will be marked as containing the respective
           object.

        num_pools: int
           Number of pooling (or similar) operation. This value will be used to
           compute the downsampling ratio from original image size to features.
           For instance, if num_pool=3 then the functions will assume that the
           feature map will be on eight (1 / 2 ** 3) in size.

        debug: bool
           Show a debug plot with BBox positions compiled from network training
           data.
    """
    # Find all background image files and strip off the file extension (we will
    # need to load meta file with the same prefix).
    fnames = glob.glob(os.path.join(path, '*.jpg'))
    fnames = [_[:-4] for _ in sorted(fnames)]
    if len(fnames) == 0:
        print(f'Warning: found no images in {path}')
        return

    # Compile and save the BBox data for each image. Farm this task out to
    # multiple processes.
    with multiprocessing.Pool() as pool:
        # Setup parallel execution.
        args = [(fname, num_pools, thresh) for fname in fnames]
        it = pool.imap(compileBBoxData, args)

        # Consume the iterator to actually start the processes.
        print('Compiling training output for network')
        for i in tqdm.tqdm(range(len(args))):
            next(it)

    # Create a debug plot to verify everything went fine.
    if debug:
        img, y_bbox, y_score = compileBBoxData(args[0])
        img = np.transpose(img, [1, 2, 0])
        for _, bbox in sorted(y_bbox.items()):
            showBBoxData(img, bbox, y_score)
        plt.show()
