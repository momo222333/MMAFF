import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


limb_num_node = 22
limb_self_link = [(i, i) for i in range(limb_num_node)]
limb_inward_ori_index = [(2,1),(3,2),(4,3),(5,4),(6,5),(7,5),
                    (8,1),(9,8),(10,9),(11,10),(12,11),(13,11),
                     (15, 14), (16, 15), (17, 16), (18, 17),
                     (19, 14), (20, 19), (21, 20), (22, 21)]

limb_inward = [(i - 1, j - 1) for (i, j) in limb_inward_ori_index]
limb_outward = [(j, i) for (i, j) in limb_inward]
limb_neighbor = limb_inward + limb_outward


class Graph:
    def __init__(self, labeling_mode='spatial', center=1, max_hop=1, dilation=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.edge = self_link + inward
        self.max_hop = max_hop
        self.dilation = dilation
        self.hop_dis = tools.get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.center = center - 1
        self.A_center = self.get_adjacency(labeling_mode)
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.limb_A = self.get_limb_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

    def get_limb_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.limb_A
        if labeling_mode == 'spatial':
            limb_A = tools.get_spatial_graph(limb_num_node, limb_self_link, limb_inward, limb_outward)
        else:
            raise ValueError()
        return limb_A


        # 归一化以及快速图卷积的预处理

    def get_adjacency(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        valid_hop = range(0, self.max_hop + 1, self.dilation)  # 合法的距离值：0或1
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:  # hop=0,1
            adjacency[self.hop_dis == hop] = 1  # 将0|1的位置置1,inf抛弃
        normalize_adjacency = tools.normalize_digraph(adjacency)  # 图卷积的预处理, 这里的normalize_adjacency已经是归一化之后的A了

        if labeling_mode == 'spatial':
            A = []
            for hop in valid_hop:  # hop=0,1
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:  # j可以视为根节点的本身（hop=0）或者其邻接节点（hop=1）
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    # A.append(a_root + a_close)
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            return A
        else:
            raise ValueError("Do Not Exist This Strategy")