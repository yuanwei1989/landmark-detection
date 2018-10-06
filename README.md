# Landmark Localisation in 3D Medical Images

This software implements a Convolutional Neural Network (CNN) for automatic simultaneous localisation of multiple landmarks in 3D medical images (eg. fetal head ultrasound).

Tensorflow implementation of the MICCAI 2018 paper [Fast Multiple Landmark Localisation Using a Patch-based Iterative Network](https://arxiv.org/abs/1806.06987).
<br/><br/>

![pipeline](pipeline.png)


## Prerequisites

- Python 2.7
- [Tensorflow 1.8.0](https://github.com/tensorflow/tensorflow/tree/r1.8)
- [SciPy](http://www.scipy.org/install.html)
- (Optional) [NiBabel](http://nipy.org/nibabel/installation.html#installation) (for reading NIfTI input)
- (Optional) [Matplotlib](https://matplotlib.org/users/installing.html) (for visualization)
- (Optional) [MATLAB](https://www.mathworks.com/products/matlab.html) (for building shape model)


## Usage

To train a CNN model:

    $ python train.py
    
To test with an existing CNN model (with full evaluation):

    $ python infer.py

To test with an existing CNN model (without evaluation)

    $ python infer_simple.py
    
To build the statistical shape model, run `./shape_model/CreateShapeModel.m` in MATLAB.

Pretrained CNN model available at `./cnn_model`.

Pretrained shape model available at `./shape_model/shape_model`.


## Data

We are not able to share the dataset we used for the paper due to sensitive patient information. However, we have provided a dummy data which is a random noise matrix to mimic the actual data we used.
    

## Results

Predicted landmarks (green) and ground truth landmarks (red) overlaid on the two standard scan planes of fetal brain.

![result1](results_sample/landmarks_visual2D/test/data2.png)

Path taken by the landmarks for 10 iterations during inference. Each colour represent a different landmark. We initialise 6 instances for each landmark during inference and take the mean as the final prediction.

![result1](results_sample/landmark_path/test/data2.gif)


## Author

Yuanwei Li / yuanwei_li@hotmail.com
