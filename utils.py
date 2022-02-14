import open3d as o3d
import numpy as np
import copy

def fixedNumDownSample(vertices, desiredNumOfPoint, leftVoxelSize, rightVoxelSize):
    """ Use the method voxel_down_sample defined in open3d and do bisection iteratively 
        to get the appropriate voxel_size which yields the points with the desired number.
        INPUT:
            vertices: numpy array shape (n,3)
            desiredNumOfPoint: int, the desired number of points after down sampling
            leftVoxelSize: float, the initial bigger voxel size to do bisection
            rightVoxelSize: float, the initial smaller voxel size to do bisection
        OUTPUT:
            downSampledVertices: down sampled points with the original data type
    
    """
    assert leftVoxelSize > rightVoxelSize, "leftVoxelSize should be larger than rightVoxelSize"
    assert vertices.shape[0] > desiredNumOfPoint, "desiredNumOfPoint should be less than or equal to the num of points in the given array."
    if vertices.shape[0] == desiredNumOfPoint:
        return vertices
    
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd = pcd.voxel_down_sample(leftVoxelSize)
    assert len(pcd.points) <= desiredNumOfPoint, "Please specify a larger leftVoxelSize."
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd = pcd.voxel_down_sample(rightVoxelSize)
    assert len(pcd.points) >= desiredNumOfPoint, "Please specify a smaller rightVoxelSize."
    
    pcd.points = o3d.utility.Vector3dVector(vertices)
    midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
    pcd = pcd.voxel_down_sample(midVoxelSize)
    while len(pcd.points) != desiredNumOfPoint:
        if len(pcd.points) < desiredNumOfPoint:
            leftVoxelSize = copy.copy(midVoxelSize)
        else:
            rightVoxelSize = copy.copy(midVoxelSize)
        midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd = pcd.voxel_down_sample(midVoxelSize)
    
    # print("final voxel size: ", midVoxelSize)
    downSampledVertices = np.asarray(pcd.points, dtype=vertices.dtype)
    return downSampledVertices


def fixedNumDownSamplePCL(initPcd, desiredNumOfPoint, leftVoxelSize, rightVoxelSize):
    """ Use the method voxel_down_sample defined in open3d and do bisection iteratively 
        to get the appropriate voxel_size which yields the points with the desired number.
        INPUT:
            initPcd: open3d.geometry.PointCloud
            desiredNumOfPoint: int, the desired number of points after down sampling
            leftVoxelSize: float, the initial bigger voxel size to do bisection
            rightVoxelSize: float, the initial smaller voxel size to do bisection
        OUTPUT:
            pcd: down sampled pointcloud
    
    """
    assert leftVoxelSize > rightVoxelSize, "leftVoxelSize should be larger than rightVoxelSize"
    assert len(initPcd.points) > desiredNumOfPoint, "desiredNumOfPoint should be less than or equal to the num of points in the given point cloud."
    if len(initPcd.points) == desiredNumOfPoint:
        return initPcd
    
    pcd = copy.deepcopy(initPcd)
    pcd = pcd.voxel_down_sample(leftVoxelSize)
    assert len(pcd.points) <= desiredNumOfPoint, "Please specify a larger leftVoxelSize."
    pcd = copy.deepcopy(initPcd)
    pcd = pcd.voxel_down_sample(rightVoxelSize)
    assert len(pcd.points) >= desiredNumOfPoint, "Please specify a smaller rightVoxelSize."
    
    pcd = copy.deepcopy(initPcd)
    midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
    pcd = pcd.voxel_down_sample(midVoxelSize)
    while len(pcd.points) != desiredNumOfPoint:
        if len(pcd.points) < desiredNumOfPoint:
            leftVoxelSize = copy.copy(midVoxelSize)
        else:
            rightVoxelSize = copy.copy(midVoxelSize)
        midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
        pcd = copy.deepcopy(initPcd)
        pcd = pcd.voxel_down_sample(midVoxelSize)
    
    return pcd


