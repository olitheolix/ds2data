import os
import glob
import json
import tqdm
import pickle
import scipy.signal
import matplotlib.pyplot as plt

import numpy as np
import PIL.Image as Image
from collections import namedtuple, Counter

MetaData = namedtuple('MetaData', 'filename score')


def computeScore(meta, img_dim_hw, bboxes):
    assert bboxes.shape[1] == 4

    # Find out where the anchor box will overlap with each BBox. To do
    # this by stamping a block of 1's into the image and convolve it
    # with the anchor box.
    bbox_score = np.zeros((len(bboxes), *img_dim_hw), np.float16)
    for i, (x0, y0, x1, y1) in enumerate(bboxes):
        # BBox size in pixels.
        bbox_area = (x1 - x0) * (y1 - y0)
        assert bbox_area > 0

        # Stamp a BBox sized region into the otherwise empty image. This
        # "box" is what we will convolve with the anchor to compute the
        # overlap.
        bbox_score[i, y0:y1, x0:x1] = 1
        anchor = np.ones((y1 - y0, x1 - x0), np.float32)

        # Convolve the BBox with the anchor box. The FFT version is much
        # faster but introduces numerical artefacts which we must remove.
        # Fortunately, this is easy because we know the correct convolution
        # result contain real valued integers since both input signals also
        # only contained real valued integers.
        overlap = scipy.signal.fftconvolve(bbox_score[i], anchor, mode='same')
        overlap = np.round(np.abs(overlap))

        # To compute the ratio of overlap we need to know which box (BBox
        # or anchor) is smaller. We need this because if one box is fully
        # inside the other we would like the overlap metric to be 1.0 (ie
        # 100%), and the convolution at that point will be identical to the
        # area of the smaller box. Therefore, find out which box has the
        # smaller area. Then compute the overlap ratio.
        bbox_score[i] = overlap / bbox_area
        del i, x0, x1, y0, y1, overlap, bbox_area
    return bbox_score


def genBBoxData(bboxes, bbox_labels, bbox_score, ft_dim, anchor_dim, thresh):
    assert {isinstance(_, int) for _ in bbox_labels} == {True}

    # Compute the BBox parameters that the network will ultimately learn.
    # These are two values to encode the BBox centre (relative to the
    # anchor in the full image), and another two value to encode the
    # width/height difference compared to the anchor.
    img_height, img_width = bbox_score.shape[1:]
    mul = img_height / ft_dim[0]
    ofs = mul / 2
    out = np.zeros((5, *ft_dim), np.float16)
    for fy in range(ft_dim[0]):
        for fx in range(ft_dim[1]):
            # Convert the current feature coordinates to image coordinates. This
            # is the centre of the anchor box in image coordinates.
            anchor_centre_x = int(fx * mul + ofs)
            anchor_centre_y = int(fy * mul + ofs)

            # Ignore this position unless the anchor is fully inside the image.
            x0 = anchor_centre_x - anchor_dim[1] // 2
            y0 = anchor_centre_y - anchor_dim[0] // 2
            x1 = anchor_centre_x + anchor_dim[1] // 2
            y1 = anchor_centre_y + anchor_dim[0] // 2
            if x0 < 0 or x1 >= img_width or y0 < 0 or y1 >= img_height:
                continue
            del x0, y0, x1, y1

            # Find out if the score in the neighbourhood of the anchor position
            # exceeds the threshold. We need search the neighbourhood, not just
            # a single point, because of the inaccuracy when mapping feature
            # coordinates to image coordinates.
            x0, x1 = int(anchor_centre_x - ofs), int(anchor_centre_x + ofs)
            y0, y1 = int(anchor_centre_y - ofs), int(anchor_centre_y + ofs)
            tmp = bbox_score[:, y0:y1, x0:x1]
            best = np.argmax(np.amax(np.amax(tmp, axis=2), axis=1))
            if bbox_score[best, anchor_centre_y, anchor_centre_x] <= thresh:
                continue
            del x0, x1, y0, y1, tmp

            # Unpack the parameters and label for the BBox with the best score.
            bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bboxes[best]
            label_int = bbox_labels[best]

            # Compute the BBox location, width and height relative to the
            # anchor (image coordinates).
            rel_x = np.mean([bbox_x0, bbox_x1]) - anchor_centre_x
            rel_y = np.mean([bbox_y0, bbox_y1]) - anchor_centre_y
            rel_w = (bbox_x1 - bbox_x0)
            rel_h = (bbox_y1 - bbox_y0)

            # Insert the BBox parameters into the training vector at the
            # respective image position.
            out[:, fy, fx] = [label_int, rel_x, rel_y, rel_w, rel_h]
    return out, bbox_score


def bboxFromTrainingData(im_dim, y_bbox):
    assert y_bbox.ndim == 3
    assert y_bbox.shape[0] == 5

    im_height, im_width = im_dim
    ft_height, ft_width = y_bbox[0].shape

    mul = im_height / ft_height
    ofs = mul / 2
    labels, bboxes = y_bbox[0], y_bbox[1:]

    # Iterate over every position of the feature map and determine if the
    # network found an object. Add the estimated BBox if it did.
    out = []
    for fy in range(ft_height):
        for fx in range(ft_width):
            label = labels[fy, fx]
            if label == 0:
                continue

            # Convert the current feature map position to the corresponding
            # image coordinates.
            anchor_x = fx * mul + ofs
            anchor_y = fy * mul + ofs

            # BBox in image coordinates.
            bbox = bboxes[:, fy, fx]

            # The BBox parameters are relative to the anchor position and
            # size. Here we convert those relative values back to absolute
            # values in the original image.
            bbox_x = bbox[0] + anchor_x
            bbox_y = bbox[1] + anchor_y
            bbox_half_width = bbox[2] / 2
            bbox_half_height = bbox[3] / 2

            # Ignore invalid BBoxes.
            if bbox_half_width < 2 or bbox_half_height < 2:
                continue

            # Compute BBox corners and clip them at the image boundaries.
            x0, y0 = bbox_x - bbox_half_width, bbox_y - bbox_half_height
            x1, y1 = bbox_x + bbox_half_width, bbox_y + bbox_half_height
            x0, x1 = np.clip([x0, x1], 0, im_dim[1] - 1)
            y0, y1 = np.clip([y0, y1], 0, im_dim[0] - 1)
            out.append([label, x0, y0, x1, y1])

    # The BBox label and corner coordinates are integers, even though we used
    # floating point numbers to compute them. Here we convert all data to
    # integer precision.
    return np.array(np.round(out), np.int16)


def showBBoxData(img, y_bbox, y_score):
    assert img.ndim == 3
    assert img.shape[2] == 3

    # Convert the training output to BBox positions.
    bboxes = bboxFromTrainingData(img.shape[:2], y_bbox)

    # Insert the BBox rectangle into the image.
    img_bbox = np.array(img)
    for (label, x0, y0, x1, y1) in bboxes:
        img_bbox[y0:y1, x0, :] = 255
        img_bbox[y0:y1, x1, :] = 255
        img_bbox[y0, x0:x1, :] = 255
        img_bbox[y1, x0:x1, :] = 255

    # Matplotlib cannot deal with float16, so convert it.
    y_bbox = y_bbox.astype(np.float32)
    y_score = y_score.astype(np.float32)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(img_bbox)
    plt.title('Pred BBoxes')

    plt.subplot(2, 2, 3)
    plt.imshow(y_bbox[0])
    plt.title('GT Label')

    plt.subplot(2, 2, 4)
    plt.imshow(np.amax(y_score, axis=0), cmap='hot')
    plt.title('GT Score')

    plt.show()


def saveBBoxPatches(fnames, bbox_path):
    # Use this counter to create enumerated file names for each label.
    tot_label_cnt = Counter()

    for i, fname in enumerate(tqdm.tqdm(fnames)):
        # Load meta data and clean up the JSON idiosyncracy that converts
        # integers to strings when used as keys in a map.
        meta = json.load(open(fname + '.json', 'r'))
        int2name = {int(k): v for k, v in meta['int2name'].items()}

        # Load the BBox data for the current image. Then load the Image itself
        # into a Numpy array.
        y_bbox = pickle.load(open(fname + '.pickle', 'rb'))['y_bbox']
        img = np.array(Image.open(fname + '.jpg', 'r').convert('RGB'), np.uint8)
        height, width = img.shape[:2]

        # Compile the BBox positions from the training data.
        bboxes = bboxFromTrainingData((height, width), y_bbox)

        # A simple method to identify occupied regions.
        mask = np.zeros((height, width))

        # Save each BBox as a separate image.
        for (label, x0, y0, x1, y1) in bboxes:
            # Convert machine readable label to human readable one.
            label = int2name[label]

            # Ensure the output path exists.
            path = os.path.join(bbox_path, label)
            if tot_label_cnt[label] == 0:
                os.makedirs(path, exist_ok=True)

            # Save the image in the correct path and increment the label count.
            fname = os.path.join(path, f'{tot_label_cnt[label]:04d}.jpg')
            Image.fromarray(img[y0:y1, x0:x1, :]).convert('RGB').save(fname)
            tot_label_cnt[label] += 1

            # Mark the image region as used.
            mask[y0:y1, x0:x1] = 1

        # Find (mostly) empty background patches.
        label = 'background'
        path = os.path.join(bbox_path, label)
        os.makedirs(path, exist_ok=True)
        for (_, x0, y0, x1, y1) in bboxes:
            w = x1 - x0
            h = y1 - y0
            for i in range(10):
                x0 = np.random.randint(0, width - w - 1)
                y0 = np.random.randint(0, height - h - 1)
                x1, y1 = x0 + w, y0 + h
                overlap = np.sum(mask[y0:y1, x0:x1]) / (w * h)
                if overlap < 0.5:
                    break

            if overlap <= 0.5:
                # Save the image in the correct path and increment the label count.
                fname = os.path.join(path, f'{tot_label_cnt[label]:04d}.jpg')
                Image.fromarray(img[y0:y1, x0:x1, :]).convert('RGB').save(fname)
                tot_label_cnt[label] += 1


def main():
    # Folders with background images, and folder where to put output images.
    base = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(base, 'data')
    bg_path = os.path.join(base, 'stamped')
    bbox_path = os.path.join(base, 'bbox')

    # If BBox overlaps more than `thresh` with anchor then the location will be
    # marked as containing the respective object.
    thresh = 0.8
    anchor_dim = (16, 16)

    # Sampling ratio between original image and feature map.
    sample_rat = 4

    # Find all background image files and strip of the file extension (we will
    # need to load meta file with the same prefix).
    fnames = glob.glob(os.path.join(bg_path, '*.jpg'))
    fnames = [_[:-4] for _ in sorted(fnames)]
    if len(fnames) == 0:
        print(f'Warning: found no images in {bg_path}')
        return

    for i, fname in enumerate(tqdm.tqdm(fnames)):
        # Load meta data and the image, then convert the image to CHW.
        meta = json.load(open(fname + '.json', 'r'))
        img = np.array(Image.open(fname + '.jpg', 'r').convert('RGB'), np.uint8)
        img = np.transpose(img, [2, 0, 1])
        int2name = {int(k): v for k, v in meta['int2name'].items()}

        # Define the image- and feature dimensions.
        im_dim = img.shape[1:]
        ft_dim = (np.array(im_dim) / sample_rat).astype(np.int32).tolist()

        # Unpack the BBox data and map the human readable labels to numeric ones.
        bboxes = np.array(meta['bboxes'], np.int32)
        name2int = {v: k for k, v in int2name.items()}
        bbox_labels = [name2int[_] for _ in meta['labels']]
        assert 0 not in meta['int2name'], 'Zero is reserved for background'

        # Compute the score map for each individual bounding box.
        bbox_score = computeScore(meta, im_dim, bboxes)
        y_bbox, y_score = genBBoxData(
            bboxes, bbox_labels, bbox_score, ft_dim, anchor_dim, thresh)
        assert y_bbox.shape == (5, *ft_dim)
        assert y_score.shape == (bboxes.shape[0], *im_dim), y_score.shape

        # Save the expected training output in a meta data file.
        fname = os.path.join(bg_path, f'{i:04d}.pickle')
        pickle.dump({'y_bbox': y_bbox}, open(fname, 'wb'))

    # Show debug data for last image.
    img = np.transpose(img, [1, 2, 0])
    showBBoxData(img, y_bbox, y_score)


if __name__ == '__main__':
    main()
