# Simple Classifier

This is the MNIST classifier from the Tensorflow tutorial: 2 convolution
layers, one dense layer, one output layer.

My results of my experiments are [here](https://olitheolix.github.io/ds2data/).

# Reproduce Results
You will need the usual suspects: Numpy, SciPy, Matplotlib, Pillow, and Tensorflow.

```bash
wget https://github.com/olitheolix/ds2data/blob/master/ds2.tar.gz
mkdir data
tar -xvzf ds2.tar.gz -C data/
rm ds2.tar.gz
./runall.sh
```
