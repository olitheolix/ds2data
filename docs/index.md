---
layout: default
title: Overview
---

The DS2 data set is challenging for several reasons. First, it is small
compared to MNIST (ie 1,000 images for 10 labels vs 70,000 images for 10
labels). Second, the images are projected onto a 3D cube with random
orientation and third, the lighting conditions are inconsistent. Its only
redeeming feature is that each digit has exactly one prototype whereas in
MNIST, different author write the same digit slightly differently.

# Simple Network
This set of experiments is based on the Tensorflow tutorial network for MNIST.
That network has two convolution layers, one dense layer, and one output layer.
Each filter is 5x5 in size, uses ReLU activation and is followed by a max-pool
layer. All images are 32x32 in size and the drop out probability is 10%.


## [RGB Images](simple/a_simple_rgb)
The accuracy for RGB images is 90%.

## [Gray Scale Images](simple/a_simple_gray)
The accuracy drops to ~40% for Gray scale images.

## [Spatial Transformer Network (SPT)](simple/a_simple_gray_transformer)
The accuracy increases to ~60% with the help of a spatial transformer.

## [Small Network With SPT](simple/a_simple_gray_transformer_small)
The accuracy increases to ~80% if the dense layer has *fewer* neurons.

## [Small Network Without SPT](simple/a_simple_gray_no_transformer_small)
The accuracy of a small network *without* SPT is (almost)
on par with that of a large network *with* SPT, namely ~60%.

## Conclusion
Bigger is not always better but smarter layouts (eg a spatial transformer) can
boost the accuracy.

I did not augment the training set for these experiments because it feels
like cheating. With enough random rotations, flipping, and colour space
manipulations it is rather likely that the training images will eventually
contain close matches for every image in the test set. Once that happens the
primary purpose of the test set - to present genuinely new images it has not
seen before - becomes void.
