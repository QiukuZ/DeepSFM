import numpy as np
import cv2
import torch
import time


'''
1.将npy深度图转换为点云
2.将npy深度+png图像转换为彩色点云
'''
def write_ply_color(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

def write_ply(vertices, filename):
    colors = np.ones((vertices.shape[0], 3))
    colors = np.float32(colors) * 255
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

def tensor_to_ply(depth_tensor, intrinsics, ply_path):
    if torch.is_tensor(intrinsics):
        pass
    else:
        intrinsics = torch.tensor(intrinsics)
    intrinsics_inv = torch.inverse(intrinsics)
    uv_tensor = torch.FloatTensor(h, w, 3)
    uv_tensor.fill_(1.0)

    for i in range(h):
        uv_tensor[i, :, 1].fill_(i)
    for i in range(w):
        uv_tensor[:, i, 0].fill_(i)

    uvs = torch.reshape(uv_tensor, (-1, 3))
    xyz_norm = torch.matmul(intrinsics_inv, torch.transpose(uvs, 0, 1))
    xyz_norm = torch.transpose(xyz_norm, 0, 1)
    z = torch.reshape(depth_tensor, (-1, 1))
    xyz = xyz_norm.mul(z)
    xyz_mask = xyz[:, 2] > 0
    xyz = xyz[xyz_mask]

    write_ply(xyz.numpy(), ply_path)

def tensor_to_rgb_ply(depth_tensor, rgb_tensor, intrinsics, ply_path):
    if torch.is_tensor(intrinsics):
        pass
    else:
        intrinsics = torch.tensor(intrinsics)
    intrinsics_inv = torch.inverse(intrinsics)
    rgb_tensor = torch.reshape(rgb_tensor,(-1,3))
    uv_tensor = torch.FloatTensor(h, w, 3)
    uv_tensor.fill_(1.0)

    for i in range(h):
        uv_tensor[i, :, 1].fill_(i)
    for i in range(w):
        uv_tensor[:, i, 0].fill_(i)

    uvs = torch.reshape(uv_tensor, (-1, 3))
    xyz_norm = torch.matmul(intrinsics_inv, torch.transpose(uvs, 0, 1))
    xyz_norm = torch.transpose(xyz_norm, 0, 1)
    z = torch.reshape(depth_tensor, (-1, 1))
    xyz = xyz_norm.mul(z)
    xyz_mask = xyz[:, 2] > 0
    xyz = xyz[xyz_mask]
    rgb = rgb_tensor[xyz_mask]

    write_ply(xyz.numpy(), ply_path)
    write_ply_color(xyz.numpy(), rgb.numpy(), ply_path)


if __name__ == '__main__':
    time1 = time.time()

    # cam = fx fy cx cy k1 k2 k3
    cam = [1066.778, 1067.487, 312.9869, 241.3109, 0.04112172, -0.4798174, 1.890084]

    intrinsics = torch.tensor( [ [cam[0], 0, cam[2]], [0, cam[1], cam[3]], [0,0,1]])
    intrinsics_inv = torch.inverse(intrinsics)


    depth_npy_pred_path = '/home/qk/Documents/2021/benchmark/DeepSFM/dataset/ycb/ycb_test_00000/0000_I0.npy'
    depth_npy_gt_path = '/home/qk/Documents/2021/benchmark/DeepSFM/dataset/ycb/ycb_test_00000/0000.npy'
    rgb_path = '/home/qk/Documents/2021/benchmark/DeepSFM/dataset/ycb/ycb_test_00000/0000.jpg'

    gt_ply_path = '/home/qk/Documents/2021/benchmark/DeepSFM/dataset/ycb/ycb_test_00000/gt_rgb.ply'
    pred_ply_path = '/home/qk/Documents/2021/benchmark/DeepSFM/dataset/ycb/ycb_test_00000/pred_rgb.ply'

    depth_gt = np.load(depth_npy_gt_path)
    depth_tensor_gt = torch.tensor(depth_gt)

    depth_pred = np.load(depth_npy_pred_path)
    depth_tensor_pred = torch.tensor(depth_pred)

    rgb_image = cv2.imread(rgb_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB, rgb_image)

    h, w = depth_pred.shape
    rgb_tensor = torch.tensor(rgb_image)

    tensor_to_rgb_ply(depth_tensor_gt, rgb_tensor, intrinsics, gt_ply_path)
    tensor_to_rgb_ply(depth_tensor_pred, rgb_tensor, intrinsics, pred_ply_path)

    time2 = time.time()
    print('Finish in {} s '.format(time2 - time1))