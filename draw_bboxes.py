#!env python
# -*- coding: utf-8 -*-
"""
Draw a bounding box around the visible cubes in every frame.

This script shows how to parse the pickled meta data and extract the BBoxes and
their labels. It takes the image paths as an argument and will *overwrite* all
images with new ones that contain the BBoxes.
"""

import os
import pickle
import argparse
import textwrap
import PIL.Image as Image

import numpy as np


def parseCmdline():
    """Parse the command line arguments."""
    description = textwrap.dedent(f'''\
        Draw a box around all cubes in the flight path images.
        NOTE: this will overwrite all images!
    ''')

    # Create a parser and program description.
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    padd = parser.add_argument

    # Add the command line options.
    padd('path', type=str, default=os.getcwd(),
         help='Path to <meta.pickle> file')

    # Parse the actual arguments.
    param = parser.parse_args()
    return param


def main():
    param = parseCmdline()

    # Open the meta data file; abort if it does not exist.
    fname = os.path.join(param.path, 'meta.pickle')
    fname = os.path.abspath(fname)
    try:
        meta = pickle.load(open(fname, 'rb'))
    except FileNotFoundError:
        print(f'Cannot open <{fname}>')
        return
    print(f'Loaded <{fname}>')
    del fname

    # Pixel dimensions of each image.
    width, height = meta['width'], meta['height']

    # The 'projected' field contains another dictionary. Each key denotes a
    # jpg file (eg '0052.jpg') and the corresponding value is yet another
    # dictionary with information about where the object ended up in the image.
    for fname, objdata in meta['projected'].items():
        # Open the JPG file and convert it to a NumPy image.
        fname = os.path.join(param.path, fname)

        img = Image.open(fname)
        img = np.array(img, np.uint8)

        """
        The position of all visible cubes is in 'pos' and their numeric
        labels in 'labels'. The 'hlen' tuple denotes the size of the bounding
        box in x/y dimensions.

        Position and hlen values are relative to the image size and usually
        in the interval [0, 1]. If the 'pos' values are outside the interval,
        it means a cube is only partially visible on the boundary.
        """
        it = zip(objdata['pos'], objdata['labels'], objdata['hlen'])

        # Draw the rectangle for each cube.
        for pos, label, hlen in it:
            # Convert the relative coordinates to pixel values.
            x, y = int(width * pos[0]), int(height * pos[1])
            hlen_x, hlen_y = int(width * hlen[0]), int(height * hlen[1])

            # Determine the top left (x0, y0) and bottom right (x1, y1)
            # coordinates of the bounding box.
            x0, x1 = max(0, x - hlen_x), min(width, x + hlen_x)
            y0, y1 = max(0, y - hlen_y), min(height, y + hlen_y)

            # Safety check, to guard against degenerate boxes (eg cubes that
            # close to an edge and only just visible in the scene).
            if x0 == x1 or y0 == y1:
                continue

            # To draw the box, we will first backup the original content of
            # that image region, then fill that region with a uniform colour,
            # and finally restore the original content except for the outermost
            # pixel layer. This is easier than manually drawing four lines.
            bak = np.array(img[y0:y1, x0:x1, :])
            img[y0:y1, x0:x1, :] = 100
            bak = bak[1:-1, 1:-1, :]
            img[y0 + 1:y1 - 1, x0 + 1:x1 - 1, :] = bak

        # Overwrite the image.
        Image.fromarray(img).convert('RGB').save(fname)


if __name__ == '__main__':
    main()
