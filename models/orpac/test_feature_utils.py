import random
import pytest
import numpy as np
import feature_utils

setBBoxRects = feature_utils.setBBoxRects
getBBoxRects = feature_utils.getBBoxRects
setIsFg = feature_utils.setIsFg
getIsFg = feature_utils.getIsFg
setClassLabel = feature_utils.setClassLabel
getClassLabel = feature_utils.getClassLabel


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
        y = np.zeros((4 + 2 + num_classes, *ft_dim))
        bbox = np.random.random((4, *ft_dim))

        # Assign BBox data. Ensure the original array was not modified.
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
        y = np.zeros((4 + 2 + num_classes, *ft_dim))
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
        y = np.zeros((4 + 2 + num_classes, *ft_dim))
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
        y = np.zeros((4 + 2 + num_classes, *ft_dim))

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
            (4 + 2 + num_classes, 10),
        ]
        class_labels = np.random.random((num_classes, *ft_dim))
        for false_dim in false_dims:
            with pytest.raises(AssertionError):
                setClassLabel(np.zeros(false_dim), class_labels)

    def test_oneHotEncoder(self):
        """Create valid label arrays and verify the one-hot-encoder."""
        # Create random labels in [0, 9].
        num_labels = 10

        # Test various array shapes and data types.
        dims = [(3,), (2, 3), (64, 64)]
        dtypes = (np.uint8, np.int8, np.int16, np.float16, np.float32, None)
        for dim in dims:
            for dtype in dtypes:
                # Compute random labels and convert the array to the test type.
                labels = np.random.randint(0, num_labels, dim)
                labels = labels.astype(dtype) if dtype else labels.tolist()

                # Encode the labels and verify shape and data type.
                hot = feature_utils.oneHotEncoder(labels, num_labels)
                assert hot.dtype == np.uint16
                assert np.array(hot).shape == (num_labels, *dim)

                # The encoding is along the first axis and each column must
                # therefore contain exactly one non-zero entry, and that entry
                # must be 1.
                assert np.array_equal(np.count_nonzero(hot, axis=0), np.ones(dim))
                assert np.array_equal(np.sum(hot, axis=0), np.ones(dim))

                # Convert the hot-label to normal label and ensure it is correct.
                assert np.array_equal(np.argmax(hot, axis=0), labels)

    def test_oneHotEncoder_err(self):
        """Degenerate inputs to one-hot-encoder."""
        enc = feature_utils.oneHotEncoder

        # Must not raise any errors.
        enc([0, 2], 3)

        # Degenerate input array.
        with pytest.raises(AssertionError):
            feature_utils.oneHotEncoder([], 10)

        # Invalid number of labels.
        for num_classes in [-1, 0, 0.5, 1.5]:
            with pytest.raises(AssertionError):
                enc([0, 2], num_classes)

        # Label ID is larger than the number of labels.
        with pytest.raises(AssertionError):
            enc([1, 2], 2)

        # Label ID is negative.
        with pytest.raises(AssertionError):
            enc([-1, 2], 10)

        # Label ID is a float.
        with pytest.raises(AssertionError):
            enc([0, 1.5], 10)

        # Number of classes is larger than 16 Bit number.
        with pytest.raises(AssertionError):
            enc([0, 1.5], 2 ** 16)

    def test_getNumClassesFromY(self):
        assert feature_utils.getNumClassesFromY((4 + 2 + 1, 64, 64)) == 1
        assert feature_utils.getNumClassesFromY((4 + 2 + 5, 64, 64)) == 5

        wrong_dims = [
            (4 + 2, 64, 64),
            (1, 4 + 2, 64, 64),
            (4 + 2, 64),
        ]
        for dim in wrong_dims:
            with pytest.raises(AssertionError):
                feature_utils.getNumClassesFromY(dim)

    def test_sampleMasks(self):
        """Use a tiny test matrix that is easy to verify manually."""
        random.seed(0)
        np.random.seed(0)
        sampleMasks = feature_utils.sampleMasks

        mask_valid = np.zeros((1, 4), np.uint8)
        mask_fgbg = np.zeros_like(mask_valid)
        mask_bbox = np.zeros_like(mask_valid)
        mask_cls = np.zeros_like(mask_valid)
        mask_objid_at_pix = np.zeros_like(mask_valid)

        # Rows 0-3 are valid, rows 0-1 & 4-5 are suitable for FG/BG estimation,
        # rows 0 & 4 are suitable for BBox estimation and, finally, rows 1 & 5
        # are suitable for class estimation (eg the cube number).
        mask_valid[0] = [1, 1, 0, 0]
        mask_fgbg[0] = [1, 1, 0, 0]
        mask_bbox[0] = [1, 0, 1, 0]
        mask_cls[0] = [0, 1, 1, 0]
        mask_objid_at_pix[0] = [1, 2, 0, 3]

        for N in [1, 20]:
            sm_bbox, sm_isFg, sm_cls = sampleMasks(
                mask_valid, mask_fgbg, mask_bbox, mask_cls, mask_objid_at_pix, N)
            assert sm_bbox.shape == sm_isFg.shape == sm_cls.shape == mask_valid.shape
            assert sm_bbox.dtype == sm_isFg.dtype == sm_cls.dtype == mask_valid.dtype

            # FGBG mask must be a subset of valid m_fgbg.
            assert sm_bbox[0].tolist() == [1, 0, 0, 0]
            assert sm_cls[0].tolist() == [0, 1, 0, 0]
            assert sm_isFg[0].tolist() == [1, 1, 0, 0]

    def test_sampleMask_objID(self):
        """ Verify that the algorithm samples objIDs uniformly.

        For this test we assume all locations are valid and suitable for BBox
        and Label estimation.

        The test will create a distinct regions that belong to different
        objects. The sampled BBox/Label masks must have been chosen uniformly
        from it.
        """
        # Number of distinct object IDs (exluding Zero) in this test.
        num_objIDs = 5

        mask_valid = np.ones((2 * num_objIDs, 2 * num_objIDs), np.uint8)
        mask_bbox = np.ones_like(mask_valid)
        mask_cls = np.ones_like(mask_valid)
        mask_objid_at_pix = np.zeros_like(mask_valid)

        # Create block-diagonal matrix. Each block belongs to a different
        # object. Note: IDs cannot be zero because that would be tantamount ot
        # background.
        block_len = 2
        block_area = block_len ** 2
        for i in range(num_objIDs):
            a, b = i * block_len, (i + 1) * block_len
            mask_objid_at_pix[a:b, a:b] = i + 1

        # Mark all foreground locations.
        mask_fgbg = np.zeros_like(mask_valid)
        mask_fgbg[np.nonzero(mask_objid_at_pix)] = 1

        # Sample various number of locations. We also want to verify that
        # sampleMasks picks locations that belong to different objIDs. This
        # implies that we have to sample at least num_objIDs per mask. The
        # upper limit in the range expression simply means that we are asking
        # for mores locations than have a non-zero objID and the sampling must
        # saturate there.
        for N in range(1, block_area + 10):
            # Sample the masks.
            sm_bbox, sm_isFg, sm_cls = feature_utils.sampleMasks(
                mask_valid, mask_fgbg, mask_bbox, mask_cls, mask_objid_at_pix, N)

            # We must have N entries per distinct objID, unless N is larger
            # than the number of valid objects.
            num_hits = min(N * num_objIDs, block_area * num_objIDs)

            # Cls and BBox must contain N non-zero entries whereas `isFg` must
            # contain twice that because we sample N foreground and background
            # locations each.
            assert np.count_nonzero(sm_cls) == num_hits
            assert np.count_nonzero(sm_bbox) == num_hits
            assert np.count_nonzero(sm_isFg) == 2 * num_hits

            # We must have the same number of samples for foreground and
            # background.
            ids = mask_objid_at_pix[np.nonzero(sm_isFg)]
            assert np.count_nonzero(ids != 0) == num_hits
            assert np.count_nonzero(ids == 0) == num_hits
            del num_hits

            # BBox and Label must each have the samples from every objID except
            # Zero, whereas the isFg must contain objID Zero as well.
            num_hits = min(N, block_area)
            for objID in range(1, num_objIDs + 1):
                ids = mask_objid_at_pix[np.nonzero(sm_bbox)]
                np.count_nonzero(ids == objID) == num_hits

                ids = mask_objid_at_pix[np.nonzero(sm_cls)]
                np.count_nonzero(ids == objID) == num_hits

                ids = mask_objid_at_pix[np.nonzero(sm_isFg)]
                np.count_nonzero(ids == objID) == num_hits

            ids = mask_objid_at_pix[np.nonzero(sm_isFg)]
            np.count_nonzero(ids == 0) == num_hits
