import os
import glob
import numpy as np

from PIL import Image


def createForegroundShapes(width, height):
    margin = 2

    # Box shape.
    box = np.zeros((height, width, 3), np.uint8)
    box[margin:-margin, margin:-margin, :] = 255

    # Disc shape.
    centre_x = width / 2
    centre_y = height / 2
    circle_radius = min(height, width) // 2 - margin
    disc = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        for x in range(width):
            dist = np.sqrt(((x - centre_x) ** 2 + (y - centre_y) ** 2))
            disc[y, x, :] = 255 if dist < circle_radius else 0
    return dict(box=box, disc=disc)


def loadBackgroundPatches(N, width, height):
    # Location to data folder.
    data_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_path, 'data', 'background')

    # Abort if the data set does not exist.
    fnames = glob.glob(f'{data_path}/*.jpg')
    if len(fnames) == 0:
        # fixme: correct data path and download location.
        print(f'\nError: No files in {data_path}')
        print('\nPlease download '
              'https://github.com/olitheolix/ds2data/blob/master/ds2.tar.gz'
              '\nand unpack it to data/\n')
        raise FileNotFoundError
    del data_path

    # Determine how many patches we need to cut from each image to create a
    # total of N patches.
    if N % len(fnames) == 0:
        patches_per_image = N // len(fnames)
    else:
        patches_per_image = 1 + int(N // len(fnames))

    # Load each image and cut out the required number of patches.
    background = []
    for i, fname in enumerate(fnames):
        # Load image, ensure force it to RGB and convert to NumPy.
        img = Image.open(fname).convert('RGB')
        img = np.array(img, np.uint8)

        # Ensure the image is at least as large as the requested patch size.
        assert img.shape[0] > height and img.shape[1] > width

        # Cut out the patches.
        for j in range(patches_per_image):
            y0 = np.random.randint(0, img.shape[0] - height)
            x0 = np.random.randint(0, img.shape[1] - width)
            y1, x1 = y0 + height, x0 + width
            background.append(np.array(img[y0:y1, x0:x1]))

        # Abort early once we have enough patches.
        if len(background) >= N:
            break

    assert len(background) == N
    return background


def addForegroundObject(obj, background):
    # Ensure `obj` and `background` have the same size.
    assert obj.shape == background.shape
    assert obj.ndim == background.ndim == 3

    # Background image must be at least as large as the foreground object.
    assert obj.shape[0] <= background.shape[0]
    assert obj.shape[1] <= background.shape[1]

    # Convenience
    height, width, _ = obj.shape

    # Randomly scale the colour channel(s) of the foreground image.
    img = np.array(obj, np.float32)
    for i in range(img.shape[2]):
        img[:, :, i] = img[:, :, i] * np.random.uniform(0.3, 1)
    img = img.astype(np.uint8)

    # Randomly scale down the image.
    img = Image.fromarray(img)
    scale = np.random.uniform(0.35, 1)
    w, h = int(width * scale), int(height * scale)
    img = img.resize((w, h), Image.BILINEAR)
    img = np.array(img, np.uint8)

    # Compute random position in background image.
    x0 = np.random.randint(0, width - w)
    y0 = np.random.randint(0, height - h)
    x1, y1 = x0 + w, y0 + h

    # Compute a mask to only copy the image portion that contains the
    # object but not those that contain only the black background.
    idx = np.nonzero(img > 30)
    mask = np.zeros_like(img)
    mask[idx] = 1

    # Stamp the foreground object into the background image.
    out = np.array(background)
    out[y0:y1, x0:x1, :] = (1 - mask) * out[y0:y1, x0:x1, :] + mask * img
    return out


def generateImages(N, width, height, dst):
    # Load the objects we want to place over the background.
    shapes = createForegroundShapes(width, height)

    # Load enough random background patches to create N images with each
    # object in the foreground.
    background = loadBackgroundPatches(N * len(shapes), width, height)

    # The first N features are random background patches.
    folder = os.path.join(dst, 'background')
    os.makedirs(folder, exist_ok=True)
    for cnt, idx in enumerate(np.random.permutation(N)):
        img = Image.fromarray(background[idx])
        img.save(os.path.join(folder, f'{cnt:04d}.jpg'))

    # Add N images for each object type. Each of these images shows an object
    # in front of a random background patch.
    for name, shape in shapes.items():
        folder = os.path.join(dst, name)
        os.makedirs(folder, exist_ok=True)

        for cnt, idx in enumerate(np.random.permutation(N)):
            img = addForegroundObject(shape, background[idx])
            img = Image.fromarray(img)
            img.save(os.path.join(folder, f'{cnt:04d}.jpg'))


def main():
    dst = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(dst, 'data', 'basic')
    N, width, height = 1000, 32, 32
    generateImages(N, width, height, dst)


if __name__ == '__main__':
    main()
