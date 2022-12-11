# downSamplePointsWithDesiredNum

python 3.8

open3d 0.14


Sample an array of points with shape (n,3) to the desired number of points.


Method (1) Implementation of Algorithm "Farthest Point Sampling" using numpy. Accelerate the code using numba.

Method (2) Use the method voxel_down_sample defined in open3d and do bisection iteratively to get the appropriate voxel_size which yields the points with the desired number.

