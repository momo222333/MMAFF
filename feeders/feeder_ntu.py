import numpy as np

from torch.utils.data import Dataset

from feeders import tools
from .bone_pairs import ntu_pairs, kinect_limb_pairs

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, stream="body"):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.stream = stream
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        C, T, V, M = data_numpy.shape

        if self.stream == "limb":

            limb_data = np.zeros((C, T, 22, M))
            limb_data[:, :, 0, :] = data_numpy[:, :, 20, :]  # 1  21
            limb_data[:, :, 1, :] = data_numpy[:, :, 4, :]  # 2  5
            limb_data[:, :, 2, :] = data_numpy[:, :, 5, :]  # 3  6
            limb_data[:, :, 3, :] = data_numpy[:, :, 6, :]  # 4  7
            limb_data[:, :, 4, :] = data_numpy[:, :, 7, :]  # 5  8
            limb_data[:, :, 5, :] = data_numpy[:, :, 21, :]  # 6  22
            limb_data[:, :, 6, :] = data_numpy[:, :, 22, :]  # 7  23
            limb_data[:, :, 7, :] = data_numpy[:, :, 8, :]  # 8  9
            limb_data[:, :, 8, :] = data_numpy[:, :, 9, :]  # 9  10
            limb_data[:, :, 9, :] = data_numpy[:, :, 10, :]  # 10 11
            limb_data[:, :, 10, :] = data_numpy[:, :, 11, :]  # 11 12
            limb_data[:, :, 11, :] = data_numpy[:, :, 23, :]  # 12 24
            limb_data[:, :, 12, :] = data_numpy[:, :, 24, :]  # 13 25

            limb_data[:, :, 13, :] = data_numpy[:, :, 0, :]  # 14  1
            limb_data[:, :, 14, :] = data_numpy[:, :, 12, :]  # 15  13
            limb_data[:, :, 15, :] = data_numpy[:, :, 13, :]  # 16  14
            limb_data[:, :, 16, :] = data_numpy[:, :, 14, :]  # 17  15
            limb_data[:, :, 17, :] = data_numpy[:, :, 15, :]  # 18  16
            limb_data[:, :, 18, :] = data_numpy[:, :, 16, :]  # 19  17
            limb_data[:, :, 19, :] = data_numpy[:, :, 17, :]  # 20  18
            limb_data[:, :, 20, :] = data_numpy[:, :, 18, :]  # 21  19
            limb_data[:, :, 21, :] = data_numpy[:, :, 19, :]  # 22  20

            data_numpy = limb_data
            pairs = kinect_limb_pairs


        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        if self.stream == "limb":
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]

            joint_vel_data = np.zeros_like(data_numpy)
            bone_vel_data = np.zeros_like(bone_data_numpy)

            joint_vel_data[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            joint_vel_data[:, -1] = 0

            bone_vel_data[:, :-1] = bone_data_numpy[:, 1:] - bone_data_numpy[:, :-1]
            bone_vel_data[:, -1] = 0

            # final_data = np.concatenate([data_numpy, bone_data_numpy, joint_vel_data, bone_vel_data], 0)
            final_data = np.concatenate([joint_vel_data, bone_vel_data], 0)

            return final_data, label, index

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
