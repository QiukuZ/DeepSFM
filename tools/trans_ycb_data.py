import numpy as np
import cv2
import argparse
import scipy.io as scio
import os

'''
将YCB数据集转换为满足DeepSFM test的数据格式
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ycb_data_root", required=True, help='ycb dataset root')
    parser.add_argument("--ycb_pair", required=True, help='ycb image pair path')
    parser.add_argument("--output_path", required=True, help='output file path')
    parser.add_argument("--output_file_name", required=True, help='output scene name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # cam = fx fy cx cy k1 k2 k3
    cam = [1066.778, 1067.487, 312.9869, 241.3109, 0.04112172, -0.4798174, 1.890084]

    ycb_data_root = '/home/qk/Videos/YCB_Video_Dataset/data'
    ycb_pair_path = '/home/qk/Videos/YCB_Video_Dataset/pairs/0050/pairs.txt'

    pairs = np.genfromtxt(ycb_pair_path).astype(np.int)

    output_path = "/home/qk/Documents/2021/benchmark/DeepSFM/dataset/ycb"
    output_name = "ycb_test_"

    data_seq_num = 50

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    test_list = []
    for i, pair in enumerate(pairs):
        image1_id = pair[0]
        image2_id = pair[1]

        output_id = i

        print('i = {} : i1 = {} i2 = {}'.format(i, image1_id, image2_id))
        image1_path = os.path.join(ycb_data_root, str(data_seq_num).zfill(4), str(image1_id).zfill(6) + '-color.png')
        depth1_path = os.path.join(ycb_data_root, str(data_seq_num).zfill(4), str(image1_id).zfill(6) + '-depth.png')
        meta1_path = os.path.join(ycb_data_root, str(data_seq_num).zfill(4), str(image1_id).zfill(6) + '-meta.mat')

        image2_path = os.path.join(ycb_data_root, str(data_seq_num).zfill(4), str(image2_id).zfill(6) + '-color.png')
        depth2_path = os.path.join(ycb_data_root, str(data_seq_num).zfill(4), str(image2_id).zfill(6) + '-depth.png')
        meta2_path = os.path.join(ycb_data_root, str(data_seq_num).zfill(4), str(image2_id).zfill(6) + '-meta.mat')

        meta1 = scio.loadmat(meta1_path)
        meta2 = scio.loadmat(meta2_path)

        pose1 = meta1['rotation_translation_matrix'].reshape((1,-1))
        pose2 = meta2['rotation_translation_matrix'].reshape((1,-1))


        output_path_i = os.path.join(output_path, output_name + str(output_id).zfill(5))
        test_list.append(output_name + str(output_id).zfill(5))

        if not os.path.exists(output_path_i):
            os.mkdir(output_path_i)

        # save image
        image1 = cv2.imread(image1_path)
        cv2.imwrite(os.path.join(output_path_i, '0000.jpg'), image1)
        image2 = cv2.imread(image2_path)
        cv2.imwrite(os.path.join(output_path_i, '0001.jpg'), image2)

        # save depth
        depth1 = cv2.imread(depth1_path, -1)
        depth2 = cv2.imread(depth2_path, -1)
        depth1 = depth1.astype(np.float32) / meta1['factor_depth'].astype(np.float32)
        depth2 = depth2.astype(np.float32) / meta2['factor_depth'].astype(np.float32)
        np.save(os.path.join(output_path_i, '0000.npy'), depth1)
        np.save(os.path.join(output_path_i, '0001.npy'), depth2)

        # save poses.txt
        np.savetxt(os.path.join(output_path_i, 'poses.txt'), np.vstack([pose1,pose2]), fmt='%.6e')

        # save cam.txt
        cam_K = np.array([
            [cam[0], 0., cam[2]],
            [0, cam[1], cam[3]],
            [0., 0., 1.]
        ])
        np.savetxt(os.path.join(output_path_i, 'cam.txt'), cam_K, fmt='%.8e')

    # wrtie test.txt
    with open(os.path.join(output_path, 'test.txt'), 'w') as fout:
        for name in test_list:
            fout.write(name+'\n')