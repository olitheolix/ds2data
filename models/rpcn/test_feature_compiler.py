import pytest
import numpy as np
import feature_compiler

setBBoxRects = feature_compiler.setBBoxRects
getBBoxRects = feature_compiler.getBBoxRects
setIsFg = feature_compiler.setIsFg
getIsFg = feature_compiler.getIsFg
setClassLabel = feature_compiler.setClassLabel
getClassLabel = feature_compiler.getClassLabel


class TestFeatureCompiler:
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

    def test_getSetBBox(self):
        """Assign and retrieve BBox data."""
        ft_dim = (64, 64)
        num_classes = 10

        # Allocate empty feature tensor and random BBox tensor.
        y = np.zeros((1, 4 + 2 + num_classes, *ft_dim))
        bbox = np.random.random((4, *ft_dim))

        # Assign the BBox data and ensure the original array was not modified.
        y2 = setBBoxRects(y, bbox)
        assert np.array_equal(y, np.zeros_like(y))

        # Retrieve the BBox data and ensure it is correct.
        bbox_ret = getBBoxRects(y2)
        assert np.array_equal(bbox, bbox_ret)

    def test_getSetIsFg(self):
        """Assign and retrieve binary is-foreground flag."""
        ft_dim = (64, 64)
        num_classes = 10

        # Allocate empty feature tensor and random BBox tensor.
        y = np.zeros((1, 4 + 2 + num_classes, *ft_dim))
        isFg = np.random.random((2, *ft_dim))

        # Assign the BBox data and ensure the original array was not modified.
        y2 = setIsFg(y, isFg)
        assert np.array_equal(y, np.zeros_like(y))

        # Retrieve the BBox data and ensure it is correct.
        isFg_ret = getIsFg(y2)
        assert np.array_equal(isFg, isFg_ret)

    def test_getSetClassLabel(self):
        """Assign and retrieve foreground class labels."""
        ft_dim = (64, 64)
        num_classes = 10

        # Allocate empty feature tensor and random BBox tensor.
        y = np.zeros((1, 4 + 2 + num_classes, *ft_dim))
        class_labels = np.random.random((num_classes, *ft_dim))

        # Assign the BBox data and ensure the original array was not modified.
        y2 = setClassLabel(y, class_labels)
        assert np.array_equal(y, np.zeros_like(y))

        # Retrieve the BBox data and ensure it is correct.
        class_labels_ret = getClassLabel(y2)
        assert np.array_equal(class_labels, class_labels_ret)

    def test_setClassLabel_err(self):
        """Assign invalid class labels. A label tensor is valid iff its entries
        are non-negative integers, and its dimension matches `num_classes`.
        """
        ft_dim = (64, 64)
        num_classes = 10

        # Allocate empty feature tensor and random BBox tensor.
        y = np.zeros((1, 4 + 2 + num_classes, *ft_dim))

        # Wrong shape: too few classes.
        with pytest.raises(AssertionError):
            class_labels = np.random.random((num_classes - 1, *ft_dim))
            setClassLabel(y, class_labels)

        # Wrong shape: too many classes.
        with pytest.raises(AssertionError):
            class_labels = np.random.random((num_classes + 1, *ft_dim))
            setClassLabel(y, class_labels)

        # Wrong shape: class labels shape is incompatible.
        with pytest.raises(AssertionError):
            class_labels = np.random.random((num_classes + 1, 30))
            setClassLabel(y, class_labels)

        # The feature vector must have four dimensions and the first one (ie
        # batch dimension) must be One.
        false_dims = [
            (0, 4 + 2 + num_classes, *ft_dim),
            (2, 4 + 2 + num_classes, *ft_dim),
            (4 + 2 + num_classes, *ft_dim),
        ]
        class_labels = np.random.random((num_classes, *ft_dim))
        for false_dim in false_dims:
            with pytest.raises(AssertionError):
                setClassLabel(np.zeros(false_dim), class_labels)
