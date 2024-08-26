import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate the eucledian disatance between two points
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Return:
        dist: per-point squared distance, [B, N, M]
    """ 
    B, N, _ = src.shape
    B, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)

    return dist

def index_points(points, idx):

    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]

    Output:
        new_points:, indexed points data, [B, S, C]        
    """

    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud_data, [B, N, 3]
        npoint: No of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):

        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points [B, N, 3]
        new_xyz: query points [B, S, 3]
    
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """

    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):

    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, D, N]

    Return:
        new_xyz: sampled points positioned data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """

    B, N, C = xyz.shape
    S = npoint

    print(xyz.shape)

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points



class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        Last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(Last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            Last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):

        """
        Inputs:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points positioned data, [B, C, S]
            new_points_concat: sampled points feature data, [B, D', S]
        """
        
        xyz = xyz.permute(0, 2, 1)

        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            # new_xyz, new_points = sample_and_group_all(xyz, points)    # yet to be implemented
            pass

        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_xyz: sampled points position data, [B, npoints, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoints]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        
        return new_xyz, new_points

class PointNet2(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(PointNet2, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.norm_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)

    dummy_ip = torch.randn(1, 3, 1000).to(device)

    new_xyz, new_points = sa1.forward(dummy_ip, None)

    # idx = torch.randint(low=0, high=1000, size=(1, 26), dtype=torch.int).to(device)
    # idx = farthest_point_sample(dummy_ip, 26)
    # op = index_points(dummy_ip, idx)
