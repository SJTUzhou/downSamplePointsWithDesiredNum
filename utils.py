import open3d as o3d
import numpy as np
import numba 
import copy


def farthestPointDownSample(vertices, num_point_sampled, return_flag=False):
    """ Use Farthest Point Sampling [FPS] to get a down sampled pointcloud
	INPUT:
            vertices: numpy array, shape (n,3) or (n,2)
            num_point_sampled: int, the desired number of points after down sampling
            return_flag: numpy boolean array, the mask of selected points in the vertices
        OUTPUT:
            downSampledVertices: down sampled points with the original data type
	""" 
    N = len(vertices)
    n = num_point_sampled
    assert n <= N, "Num of sampled point should be less than or equal to the size of vertices."
    _G = np.mean(vertices, axis=0) # centroid of vertices
    _d = np.linalg.norm(vertices - _G, axis=1, ord=2)
    farthest = np.argmax(_d) 
    distances = np.inf * np.ones((N,))
    flags = np.zeros((N,), np.bool_) 
    for i in range(n):
        flags[farthest] = True
        distances[farthest] = 0.
        p_farthest = vertices[farthest]
        dists = np.linalg.norm(vertices[~flags] - p_farthest, axis=1, ord=2)
        distances[~flags] = np.minimum(distances[~flags], dists)
        farthest = np.argmax(distances)
    if return_flag == True:
        return vertices[flags], flags
    else:
        return vertices[flags]


@numba.njit(numba.float64[:,:](numba.float64[:,:],numba.intc), cache=True, parallel=True)
def numbaFarthestPointDownSample(vertices, num_point_sampled):
    """ Use Farthest Point Sampling [FPS] to get a down sampled pointcloud
	INPUT:
            vertices: numpy array, shape (n,3) or (n,2)
            num_point_sampled: int, the desired number of points after down sampling
        OUTPUT:
            downSampledVertices: down sampled points with the original data type
	""" 
    N = vertices.shape[0]
    D = vertices.shape[1]
    assert num_point_sampled <= N, "Num of sampled point should be less than or equal to the size of vertices."
    _G = np.empty((D,),np.float64)
    for d in range(D):
        _G[d] = np.mean(vertices[:,d])

    dists = np.zeros((N,),np.float64)
    for i in numba.prange(N):
        for d in range(D):
            dists[i] += (vertices[i,d] - _G[d])**2
    farthest = np.argmax(dists) 
    distances = np.inf * np.ones((N,))
    flags = np.zeros((N,), np.bool_)
    for _ in range(num_point_sampled):
        flags[farthest] = True
        distances[farthest] = 0.
        p_farthest = vertices[farthest]
        for i in numba.prange(N):
            dist = 0.
            if not flags[i]:
                for d in range(D):
                    dist += (vertices[i,d] - p_farthest[d])**2
            distances[i] = min(distances[i], dist)
        farthest = np.argmax(distances)
    return vertices[flags]



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


