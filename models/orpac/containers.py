class Shape:
    """Container to store 3D Image/array/feature/tensor sizes.

    This is a convenience class because size specifications are often required
    yet their format is ambigous. Sometimes, images are specified as CHW
    (Tensorflow), sometimes as HWC (NumPy, Matplotlib). Sometimes, only the
    width and height are needed which Tensorflow needs as (height, width) yet
    eg. PIL returns as (width, height).

    This container class accepts the three size parameters and can return them
    in all possible formats.

    Inputs:
        chan: int
            Number of channels. Must be non-negative or None.
        height: int
            Must be non-negative (can *not* be None).
        width: int
            Must be non-negative (can *not* be None).
    """
    def __init__(self, chan, height, width):
        # Sanity checks.
        assert chan is None or isinstance(chan, int) and chan >= 0
        assert isinstance(width, int) and width >= 0
        assert isinstance(height, int) and height >= 0

        # Store the parameters.
        self.chan = chan
        self.height = height
        self.width = width

    def __repr__(self):
        return f'Shape(chan={self.chan}, height={self.height}, width={self.width})'

    def __eq__(self, ref):
        try:
            assert isinstance(ref, Shape)
            assert ref.chan == self.chan
            assert ref.height == self.height
            assert ref.width == self.width
            return True
        except AssertionError:
            return False

    def copy(self):
        return Shape(self.chan, self.height, self.width)

    def chw(self):
        return (self.chan, self.height, self.width)

    def hwc(self):
        return (self.height, self.width, self.chan)

    def hw(self):
        return (self.height, self.width)

    def wh(self):
        return (self.width, self.height)
