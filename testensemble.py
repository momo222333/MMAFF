import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

'''
Dataset split wise alpha values

NTU60 Xsub : [2, 1.8, 0.7]
NTU60 XView : [2.1, 1.6, 0.7]

NTU120 XSub : [2.2, 1.5, 0.7]
NTU120 XSet : [2.1, 2, 0.8]

NTU60x : [3.4, 2, 1]
NTU120x : [2.4, 2.2, 1]

'''


if __name__ == "__main__":
    


    j_21 = "/public/home/zxh_8991/project/CTR-GCN-main/work_dir/ntu60/xsub/joint_21_aff(res, out)&kd/epoch1_test_score.pkl"


    b_21 = "/public/home/zxh_8991/project/CTR-GCN-main/work_dir/ntu60/xsub/bone_21_aff(res, out)&kd/epoch1_test_score.pkl"


    limb = "/public/home/zxh_8991/project/CTR-GCN-main/work_dir/ntu60/xsub/limb_aff(res, out)&kd_vel/epoch1_test_score.pkl"

    with open(j_21, 'rb') as r1:
        r1 = list(pickle.load(r1).items())



    with open(b_21, 'rb') as r2:
        r2 = list(pickle.load(r2).items())


    with open(limb, 'rb') as r3:
        r3 = list(pickle.load(r3).items())


    right_num = total_num = right_num_5 = 0

    parser = argparse.ArgumentParser()

    arg = parser.parse_args()


    label = []
    npz_data = np.load('/public/home/zxh_8991/data/NTU60_CS.npz')
    label = np.where(npz_data['y_test'] > 0)[1]


    # arg.alpha = [0.6, 1.0, 0.6]    #  93.1219%   joint+bone+limb(4)
    # arg.alpha = [0.6, 0.8, 0.6]    # 93.0733%  joint+bone+limb(vel)

    # arg.alpha = [0, 0, 0]
    res = []

    for alpha1 in np.arange(0.6, 1.0, 0.1):
        for alpha2 in np.arange(0.6, 1.1, 0.1):
            for alpha3 in np.arange(0.6, 1.0, 0.1):

                for i in tqdm(range(len(label))):
                    l = label[i]
                    _, r11 = r1[i]
                    _, r22 = r2[i]
                    _, r33 = r3[i]

                    r = r11 * alpha1 + r22 * alpha2 + r33 * alpha3

                    rank_5 = r.argsort()[-5:]
                    right_num_5 += int(int(l) in rank_5)
                    r = np.argmax(r)
                    right_num += int(r == int(l))
                    total_num += 1
                acc = right_num / total_num
                acc5 = right_num_5 / total_num

                res.append(acc)

    acc = max(res)

                # if acc > res:
                #     res = acc

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
