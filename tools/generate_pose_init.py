import numpy as np
import os
import math
from icecream import ic
import random


'''
将YCB数据集的Pose转换成满足DeepSFM pose 格式
'''
def eulerAngles2rotationMat(theta, format='rad'):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """
    if format is 'degree':
        theta = [i * math.pi / 180.0 for i in theta]

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def read_test(path):
    test = []
    with open(path, 'r') as fin:
        test = fin.readlines()
    return test

def get_random_T(rad_range, t_range):
    theta = np.array([random.uniform(-rad_range, rad_range),random.uniform(-rad_range, rad_range),random.uniform(-rad_range, rad_range)])
    t = np.array([random.uniform(-t_range, t_range),random.uniform(-t_range, t_range),random.uniform(-t_range, t_range)])
    T = np.identity(4)
    T[:3,3] = t
    T[:3,:3] = eulerAngles2rotationMat(theta, format='rad')
    return T

# use gt poses to generate noise pose!
if __name__ == '__main__':
    pose_init = 'gt'

    dataset_root = '/home/qk/Documents/2021/benchmark/DeepSFM/dataset/ycb'


    test_list = read_test(os.path.join(dataset_root, 'test.txt'))

    bottom = np.array([0,0,0,1]).astype(np.float32)
    for test in test_list:
        test = test[:-1]
        poses = np.genfromtxt(os.path.join(dataset_root, test, 'poses.txt'))

        T2 = np.vstack((poses[1].reshape(3,4), bottom))
        T2_noise = np.dot(get_random_T(0.2, 0.05), T2)

        poses[1] = T2_noise[:3].reshape(1,-1)
        np.savetxt(os.path.join(dataset_root, test, pose_init + '_poses.txt'), poses)
        print(" doing {}".format(test))
        # print('aaa')