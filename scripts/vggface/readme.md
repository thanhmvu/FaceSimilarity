Exporting VGG Face Recognition model from Torch to Tensor Flow.
Please download VGG_FACE.t7 and run export.lua to generate network.h5

Similar to code written for exporting caffe models: https://github.com/ethereon/caffe-tensorflow




Network Structure
-------------------
<pre><code>
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> (17) -> (18) -> (19) -> (20) -> (21) -> (22) -> (23) -> (24) -> (25) -> (26) -> (27) -> (28) -> (29) -> (30) -> (31) -> (32) -> (33) -> (34) -> (35) -> (36) -> (37) -> (38) -> (39) -> (40) -> output]
  (1): nn.SpatialConvolutionMM(3 -> 64, 3x3, 1,1, 1,1)
  (2): nn.ReLU
  (3): nn.SpatialConvolutionMM(64 -> 64, 3x3, 1,1, 1,1)
  (4): nn.ReLU
  (5): nn.SpatialMaxPooling(2,2,2,2)
  (6): nn.SpatialConvolutionMM(64 -> 128, 3x3, 1,1, 1,1)
  (7): nn.ReLU
  (8): nn.SpatialConvolutionMM(128 -> 128, 3x3, 1,1, 1,1)
  (9): nn.ReLU
  (10): nn.SpatialMaxPooling(2,2,2,2)
  (11): nn.SpatialConvolutionMM(128 -> 256, 3x3, 1,1, 1,1)
  (12): nn.ReLU
  (13): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
  (14): nn.ReLU
  (15): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
  (16): nn.ReLU
  (17): nn.SpatialMaxPooling(2,2,2,2)
  (18): nn.SpatialConvolutionMM(256 -> 512, 3x3, 1,1, 1,1)
  (19): nn.ReLU
  (20): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
  (21): nn.ReLU
  (22): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
  (23): nn.ReLU
  (24): nn.SpatialMaxPooling(2,2,2,2)
  (25): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
  (26): nn.ReLU
  (27): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
  (28): nn.ReLU
  (29): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
  (30): nn.ReLU
  (31): nn.SpatialMaxPooling(2,2,2,2)
  (32): nn.View
  (33): nn.Linear(25088 -> 4096)
  (34): nn.ReLU
  (35): nn.Dropout(0.500000)
  (36): nn.Linear(4096 -> 4096)
  (37): nn.ReLU
  (38): nn.Dropout(0.500000)
  (39): nn.Linear(4096 -> 2622)
  (40): nn.SoftMax
}
</pre></code>


==============================================================
Deep Face Recognition

Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman
Visual Geometry Group, University of Oxford
==============================================================

--------
Overview
--------

This package contains the Caffe[3] models for computation of "VGG Face" descriptor.
The algorithm details can be found in [1]. The model was trained using MatConvNet[2] Library.

The source code and data packages can be downloaded from: 
http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

The models and other contents of this package can only be used for non-commercial research purposes. 
(Please read the licence terms here. http://creativecommons.org/licenses/by-nc/4.0/) 

Please cite [1] if you use the code or the data. If you have any questions regarding the package, 
please contact Omkar M. Parkhi <omkar@robots.ox.ac.uk>


----------
References
----------

[1] O. M. Parkhi, A. Vedaldi, A. Zisserman
Deep Face Recognition
British Machine Vision Conference, 2015.

[2] A. Vedaldi, K. Lenc
MatConvNet - Convolutional Neural Networks for MATLAB
arXiv:1412.4564, 2014.

[3] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, T. Darrell 
Caffe: Convolutional Architecture for Fast Feature Embedding
arXiv:1408.5093, 2014

