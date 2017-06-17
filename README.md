# DS2Data

A Space Sim inspired data set to experiment with machine learning.

The objects of interest are cubes with a number on them. You may think of it as
a 3D version of the [MNIST](http://yann.lecun.com/exdb/mnist/) data set.

<img src="docs/img/cube_samples.png">

In addition to the training samples, the data also contains pre-rendered flight
through a 3D space scene.

The meta data for the flight is in `flightpath/meta.pickle`. It contains, among
other things, the screen position of each cube in each frame. The
`draw_bboxes.py` shows how to use this information to draw bounding boxes the
cubes in the images.

| Original Scene               | After `draw_bboxes.py`      |
| ---------------------------- | --------------------------- |
|![](docs/img/scene_plain.jpg) | ![](docs/img/scene_bbox.jpg)|


# Installation
Just download and unpack the `ds2.tar.gz` file and use it in your ML model.

In case you wonder, all images were created by
the [PyHorde](https://github.com/olitheolix/pyhorde) sister project which
uses [Horde3D](http://www.horde3d.org/) to render the scene.

# License
The data set and code are available under the Apache Software License 2.0.

# Disclaimer
The numeral textures are royalty free images
from [PixaBay](https://pixabay.com/en/photos/?hp=&image_type=&cat=&min_width=&min_height=&q=counting+math+numbers+numerals+funny&order=popular).


# Feeback
I would be happy to hear from anyone who uses it. Does it work for you? Is
something missing? Which ML models have you built for it, and how well did they
do?
