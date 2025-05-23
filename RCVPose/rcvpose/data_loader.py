import torch
from rmap_dataset import RMapDataset
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

linemod_K = np.array([[572.4114, 0., 325.2611],
                      [0., 573.57043, 242.04899],
                      [0., 0., 1.]])

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    actual_xyz = xyz
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy, actual_xyz

def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts, vs, us

@jit(nopython=True, parallel=True)
def fast_for_map(yList, xList, xyz, distance_list, Radius3DMap):
    for i in prange(len(xList)):
        Radius3DMap[yList[i], xList[i]] = distance_list[i]
    return Radius3DMap

class RData(RMapDataset):
    def __init__(self, root, dataset, set='train', obj_name='ape', kpt_num='1'):
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.dataset = dataset
        super().__init__(
            root,
            dataset,
            set=set,
            obj_name=obj_name,
            kpt_num=kpt_num,
            transform=self.transform
        )

    def transform(self, img_id, img, depth, mask, gtpose, kpt):
        # تولید نقشهٔ فاصله سه‌بعدی
        Radius3DMap = np.zeros(mask.shape)
        depth[np.where(mask == 0)] = 0
        xyz_mm, y, x = rgbd_to_point_cloud(linemod_K, depth)
        xyz = xyz_mm / 1000.0

        # تبدیل gtpose به میلی‌متر
        gtpose_mm = gtpose.copy()
        gtpose_mm[:, 3:] = gtpose[:, 3:] * 1000.0

        # کلیدواژه به میلی‌متر
        kpt_mm = kpt * 1000.0
        _, transformed_kpt = project(
            np.array([kpt_mm]), linemod_K, gtpose_mm
        )
        transformed_kpt = transformed_kpt[0] / 1000.0

        # محاسبهٔ فهرست فاصله‌ها
        distance_list = np.sqrt(
            (xyz[:, 0] - transformed_kpt[0])**2 +
            (xyz[:, 1] - transformed_kpt[1])**2 +
            (xyz[:, 2] - transformed_kpt[2])**2
        )
        Radius3DMap = fast_for_map(y, x, xyz, distance_list, Radius3DMap)

        # نرمال‌سازی تصویر و برچسب
        img = np.array(img, dtype=np.float64) / 255.0
        lbl = Radius3DMap.astype(np.float64) * 10.0
        lbl = np.where(lbl > self.max_radii_dm, 0, lbl)

        if lbl.ndim == 2:
            lbl = lbl[np.newaxis, ...]

        img = (img - self.mean) / self.std

        # بردار باینری ماده/پس‌زمینه
        sem_lbl = np.where(lbl > 0, 1.0, -1.0)

        # فیلتر نویز برای ycb
        if self.dataset != 'lm':
            lbl = np.where(lbl >= 10.0, 0.0, lbl)

        # تبدیل به قالب PyTorch
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        sem_lbl = torch.from_numpy(sem_lbl).float()

        return img, lbl, sem_lbl

    def __len__(self):
        return len(self.ids)

def get_loader(opts):
    train_loader = data.DataLoader(
        RData(
            opts.root_dataset,
            opts.dataset,
            set='train',
            obj_name=opts.class_name,
            kpt_num=opts.kpt_num
        ),
        batch_size=int(opts.batch_size),
        shuffle=True,
        num_workers=1
    )
    val_loader = data.DataLoader(
        RData(
            opts.root_dataset,
            opts.dataset,
            set='val',
            obj_name=opts.class_name,
            kpt_num=opts.kpt_num
        ),
        batch_size=int(opts.batch_size),
        shuffle=False,
        num_workers=1
    )
    return train_loader, val_loader
