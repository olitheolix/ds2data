""" Save the BBox content into individual files.

This will produce one image for each BBox in each image. The size of these
images is inhomogeneous and depends on the BBox size.
"""
import os
import glob
import json
import tqdm
import pickle
import gen_bbox_labels

import numpy as np
import PIL.Image as Image
from collections import Counter


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
        bboxes = gen_bbox_labels.bboxFromTrainingData((height, width), y_bbox)

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
    bbox_path = os.path.join(base, 'bbox')
    stamped_path = os.path.join(base, 'stamped')

    # Find all background image files and strip of the file extension (we will
    # need to load meta file with the same prefix).
    fnames = glob.glob(os.path.join(stamped_path, '*.jpg'))
    fnames = [_[:-4] for _ in sorted(fnames)]
    if len(fnames) == 0:
        print(f'Warning: did not find stamped images in {stamped_path}')
        return 1

    # Produce the image patches inside the BBox.
    saveBBoxPatches(fnames, bbox_path)


if __name__ == '__main__':
    main()
