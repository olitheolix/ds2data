import pytest
import data_loader
import numpy as np

from config import NetConf


class TestDataset:
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self, method):
        self.defaults = NetConf(
            width=32, height=32, colour='L', seed=0, num_trans_regions=20,
            num_dense=32, keep_net=0.9, keep_trans=0.9, batch_size=16,
            epochs=10000, train=0.8, sample_size=None
        )

    def teardown_method(self, method):
        pass

    def test_basic(self):
        conf = self.defaults._replace(train=0.8)
        ds = data_loader.DataSet(conf)
        assert ds.lenOfEpoch('train') == 8
        assert ds.lenOfEpoch('test') == 2
        assert ds.posInEpoch('train') == 0
        assert ds.posInEpoch('test') == 0
        dim = ds.imageDimensions()
        assert isinstance(dim, np.ndarray)
        assert dim.dtype == np.uint32
        assert dim.tolist() == [1, 2, 2]
        assert ds.classNames() == {0: '0', 1: '1', 2: '2'}

    def test_change_training_size(self):
        conf = self.defaults._replace(train=1.0)
        ds = data_loader.DataSet(conf)
        assert ds.lenOfEpoch('train') == 10
        assert ds.lenOfEpoch('test') == 0

        conf = self.defaults._replace(train=0.0)
        ds = data_loader.DataSet(conf)
        assert ds.lenOfEpoch('train') == 0
        assert ds.lenOfEpoch('test') == 10

    def test_limitSampleSize(self):
        conf = self.defaults._replace(train=1.0, sample_size=10)
        ds = data_loader.DataSet(conf)

        for i in range(1, 5):
            x = np.arange(100)
            y = x % 3
            m = x % 5
            x, y, m = ds.limitSampleSize(x, y, m, i)
            assert len(x) == len(y) == len(m) == 3 * i
            assert set(y) == {0, 1, 2}

        x = np.arange(6)
        y = x % 3
        m = x % 5
        x2, y2, m2 = ds.limitSampleSize(x, y, m, 0)
        assert len(x2) == len(y2) == len(m2) == 0

        x2, y2, m2 = ds.limitSampleSize(x, y, m, 100)
        assert len(x2) == len(y2) == len(m2) == len(x)

    def test_limit_dataset_size(self):
        # We only have 3-4 features per label, which means asking for 10 will
        # do nothing.
        conf = self.defaults._replace(train=1.0, sample_size=10)
        ds = data_loader.DataSet(conf)
        assert ds.lenOfEpoch('train') == 10

        # Now there must be exactly 1 feature for each of the three labels,
        # which means a total of 3 features in the data_loader.
        conf = self.defaults._replace(train=1.0, sample_size=1)
        ds = data_loader.DataSet(conf)
        assert ds.lenOfEpoch('train') == 3

    def test_basic_err(self):
        with pytest.raises(AssertionError):
            data_loader.DataSet(self.defaults._replace(train=-1.0))
        with pytest.raises(AssertionError):
            data_loader.DataSet(self.defaults._replace(train=1.1))

    def test_nextBatch(self):
        ds = data_loader.DataSet(self.defaults._replace(train=0.8))

        # Basic parameters.
        assert ds.lenOfEpoch('train') == 8
        assert ds.lenOfEpoch('test') == 2
        assert ds.posInEpoch('train') == 0
        assert ds.posInEpoch('test') == 0

        # Fetch one feature/label/handle.
        x, y, handles = ds.nextBatch(1, 'train')
        assert len(x) == len(y) == len(handles) == 1
        assert ds.posInEpoch('train') == 1
        assert ds.posInEpoch('test') == 0
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert isinstance(handles, np.ndarray)
        assert x.shape == (1, 1, 2, 2) and x.dtype == np.float32
        assert y.shape == (1, ) and y.dtype == np.int32
        assert handles.shape == (1,) and handles.dtype == np.int64

        # Fetch the remaining 7 training data elements.
        x, y, _ = ds.nextBatch(7, 'train')
        assert len(x) == len(y) == len(_) == 7
        assert ds.posInEpoch('train') == 8
        assert ds.posInEpoch('test') == 0

        # Another query must yield nothing because the epoch is exhausted.
        x, y, _ = ds.nextBatch(1, 'train')
        assert np.array_equal(x, np.zeros((0, 1, 2, 2)))
        assert np.array_equal(y, [])
        assert len(_) == 0

    def test_nextBatch_reset(self):
        ds = data_loader.DataSet(self.defaults._replace(train=0.8))

        ds.nextBatch(2, 'train')
        assert ds.posInEpoch('train') == 2
        assert ds.posInEpoch('test') == 0

        ds.nextBatch(1, 'test')
        assert ds.posInEpoch('train') == 2
        assert ds.posInEpoch('test') == 1

        ds.reset('train')
        assert ds.posInEpoch('train') == 0
        assert ds.posInEpoch('test') == 1

        ds.reset('test')
        assert ds.posInEpoch('train') == 0
        assert ds.posInEpoch('test') == 0

        ds.nextBatch(2, 'train')
        ds.nextBatch(2, 'test')
        assert ds.posInEpoch('train') == 2
        assert ds.posInEpoch('test') == 2
        ds.reset()
        assert ds.posInEpoch('train') == 0
        assert ds.posInEpoch('test') == 0

    def test_nextBatch_invalid(self):
        ds = data_loader.DataSet(self.defaults._replace(train=0.8))

        # Invalid value for N.
        with pytest.raises(AssertionError):
            ds.nextBatch(-1, 'train')

        # Unknown dataset name (must be 'train' or 'test').
        with pytest.raises(AssertionError):
            ds.nextBatch(1, 'foo')
