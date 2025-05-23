import numpy as np
from PIL import Image
import open3d as o3d
import os
from numba import jit, prange
from tqdm import tqdm

# فقط روی کلاس 01 اجرا می‌شود
linemod_cls_names = ['01']

# پارامترهای داخلی دوربین لینمود
linemod_K = np.array([[572.4114, 0., 325.2611],
                      [0., 573.57043, 242.04899],
                      [0., 0., 1.]])

# مسیرهای فایل‌ها
linemod_path = "/content/dataset/linemod/Linemod_preprocessed/data/"
original_linemod_path = "/content/dataset/linemod/Linemod_preprocessed/data/"

# تابع پروجکشن سه‌بعدی به صفحه تصویر
def project(xyz, K, RT):
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    actual_xyz = xyz
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy, actual_xyz

# تولید point cloud از depth
def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts, vs, us

# نوشتن داده‌ها در Radius3DMap
@jit(nopython=True, parallel=True)
def fast_for_map(yList, xList, xyz, distance_list, Radius3DMap):
    for i in prange(len(xList)):
        Radius3DMap[yList[i], xList[i]] = distance_list[i]
    return Radius3DMap

# خواندن فایل .dpt
def read_depth(path):
    if path.endswith('dpt'):
        with open(path, 'rb') as f:
            h, w = np.fromfile(f, dtype=np.uint32, count=2)
            data = np.fromfile(f, dtype=np.uint16, count=w*h)
            depth = data.reshape((h, w))
    else:
        depth = np.asarray(Image.open(path)).copy()
    return depth

# پردازش اصلی
if __name__ == '__main__':
    for class_name in tqdm(linemod_cls_names, desc="Classes"):
        # بارگذاری مش
        pcd_load = o3d.io.read_point_cloud(os.path.join(linemod_path, class_name, "mesh.ply"))
        xyz_load = np.asarray(pcd_load.points)  # در واحد میلی‌متر

        # بارگذاری نقاط کلیدی
        keypoints = np.load(os.path.join(linemod_path, class_name, "Outside9.npy"))  # واحد: میلی‌متر

        for i, keypoint in enumerate(keypoints):
            pt_idx = i  # 0 تا 8
            folder_name = f"Out_pt{pt_idx + 1}_dm"  # ساخت از 1 تا 9
            saveDict = os.path.join(original_linemod_path, class_name, folder_name)
            os.makedirs(saveDict, exist_ok=True)

            dataDict = os.path.join(original_linemod_path, class_name, "depth")

            for filename in os.listdir(dataDict):
                if not filename.endswith(".dpt"):
                    continue

                realdepth = read_depth(os.path.join(dataDict, filename))
                mask_path = os.path.join(linemod_path, class_name, "mask", os.path.splitext(filename)[0] + ".png")
                mask = np.asarray(Image.open(mask_path))[:, :, 0]
                realdepth[mask == 0] = 0

                Radius3DMap = np.zeros(mask.shape)
                RT = np.load(os.path.join(linemod_path, class_name, "pose", "pose" + os.path.splitext(filename)[0] + ".npy"))
                xyz, y, x = rgbd_to_point_cloud(linemod_K, realdepth)

                # نقطه کلیدی باید با RT ترنسفورم شود
                _, transformed_kpoint = project(np.array([keypoint]), linemod_K, RT)
                transformed_kpoint = transformed_kpoint[0]  # در میلی‌متر

                # محاسبه فاصله اقلیدسی با xyz در میلی‌متر
                distance_list = np.linalg.norm(xyz - transformed_kpoint, axis=1)

                print(f"📏 Out_pt{pt_idx + 1}_dm | {filename} | Max dist (mm):", np.max(distance_list))

                # نوشتن در RadiusMap
                Radius3DMap = fast_for_map(y, x, xyz, distance_list, Radius3DMap)
                out_path = os.path.join(saveDict, os.path.splitext(filename)[0].zfill(6) + '.npy')
                np.save(out_path, Radius3DMap)  # بدون ضرب در 10؛ چون در میلی‌متر است