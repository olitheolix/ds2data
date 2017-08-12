import pytest
import containers


class TestSize:
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_size_basic(self):
        chan, height, width = 3, 10, 20
        s = containers.Shape(chan=chan, height=height, width=width)

        assert s.chan == chan
        assert s.height == height
        assert s.width == width

        assert s.chw() == (chan, height, width)
        assert s.hwc() == (height, width, chan)
        assert s.hw() == (height, width)
        assert s.wh() == (width, height)

    def test_size_err(self):
        """Class must accept None value for channel only"""
        # Must succeed because all values are non-negative integers and None is
        # admissible for 'chan'.
        containers.Shape(chan=None, height=0, width=10)

        # Must fail because None is acceptable *only* for 'chan'.
        with pytest.raises(AssertionError):
            containers.Shape(chan=3, height=None, width=10)
        with pytest.raises(AssertionError):
            containers.Shape(chan=3, height=10, width=None)

        # Class must reject floating point values.
        with pytest.raises(AssertionError):
            containers.Shape(chan=3, height=10.5, width=10)

        # Class must reject negative values.
        with pytest.raises(AssertionError):
            containers.Shape(chan=3, height=-10, width=10)

    def test_comparison(self):
        s1 = containers.Shape(chan=1, height=2, width=3)
        s2 = containers.Shape(chan=1, height=2, width=3)
        s3 = containers.Shape(chan=1, height=2, width=4)

        # All containers must be identical to themselves.
        assert s1 == s1 and s2 == s2 and s3 == s3

        # First two containers are different objects but encode the same
        # dimensions. Therefore, they must be equal.
        assert s1 == s2

        # The third container is different.
        assert s1 != s3 and s2 != s3

        # Comparison to an arbitrary object (eg string) must return False.
        assert s1 != 'foo'

    def test_copy(self):
        s1 = containers.Shape(chan=1, height=2, width=3)
        s2 = s1.copy()
        assert s2 is not s1
        assert s1 == s2

    def test_print(self):
        print(containers.Shape(chan=1, height=2, width=3))

    def test_isSquare(self):
        Shape = containers.Shape
        assert Shape(1, 2, 3).isSquare() is False
        assert Shape(1, 3, 2).isSquare() is False
        assert Shape(1, 3, 3).isSquare() is True
        assert Shape(1, 0, 0).isSquare() is True

    def test_isPow2(self):
        Shape = containers.Shape

        # Powers of 2.
        assert Shape(1, 2, 2).isPow2() is True
        assert Shape(1, 2, 4).isPow2() is True
        assert Shape(1, 4, 8).isPow2() is True

        # Special cases: dimensions of 0 and 1 do not count as power of 2.
        assert Shape(1, 0, 0).isPow2() is False
        assert Shape(1, 1, 1).isPow2() is False

        # Not all dimensions are powers of 2.
        assert Shape(1, 2, 3).isPow2() is False
        assert Shape(1, 3, 2).isPow2() is False
