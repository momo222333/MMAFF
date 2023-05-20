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
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

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


