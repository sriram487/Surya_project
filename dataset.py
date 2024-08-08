import os
import cv2
import time
import json
import random
import trimesh
import numpy as np
import numpy.ma as ma
from PIL import Image
import scipy.io as scio 
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

Borderlist = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

class YCBDataset(data.Dataset):

    def __init__ (self, root, mode, add_noise=True, config_path="dataset/ycb_bop/dataset_config", obj_id=1):
        self.config_path = config_path
        self.Trainlist = os.path.join(config_path, 'training.txt')
        self.Testlist = os.path.join(config_path, 'validation.txt')
        self.Classlist = os.path.join(config_path, 'classes.txt')
        self.mode = mode
        self.root = root 
        self.add_noise = add_noise
        self.obj_id = obj_id
        self._init_config()
        self._init_data_info()
        self.model_points = self.load_points(self.root, self.classes) # {1:[[x1, y1, z1], [x2, y2, z2], ...], 2:[...], 3:[...], ...}

    def __len__(self):
        return len(self.all_data_dirs)
    
    def __getitem__(self, index):
        # load raw data - img
        assert index < len(self)

        img, img_name = self._load_image(index)
        depth = self._load_depth(index)
        meta = self._load_meta(index)
        cam_poses = self._load_gt(index)

        # set camera focus and center
        cam = np.reshape(np.array(meta[str(img_name)]["cam_K"]), newshape=(3,3))
        cam_scale = np.array(meta[str(img_name)]["depth_scale"])
        cam_cx, cam_cy, cam_fx, cam_fy = cam[0][2], cam[1][2], cam[0][0], cam[1][1]

        # color jitter (noise) for img
        if self.add_noise:
            img = self.trancolor(img)
        img = np.array(img) # shape: (480, 640, 3)

        # get ground truth rotation and translation
        for obj in cam_poses[str(img_name)]:
            if obj["obj_id"] == self.obj_id:
                R_gt = np.reshape(np.array(obj["cam_R_m2c"]), newshape=(3,3))  # (3, 3)
                T_gt = np.reshape(np.array(obj["cam_t_m2c"]), newshape=(1,3))  # (1, 3)
                label_id = cam_poses[str(img_name)].index(obj)
                break

        try:
            
            label = self._load_labels(index, label_id)
            
            mask_label = ma.getmaskarray(ma.masked_equal(label, 255))

            # get sample point (3D) on model
            model_sample_list = list(range(len(self.model_points[str(self.obj_id)])))
            model_sample_list = sorted(random.sample(model_sample_list, self.sample_model_pt_num))
            # sampled_model_pt = np.array(self.model_points[str(self.obj_id)][model_sample_list, :])
            sampled_model_pt = np.array([self.model_points[str(self.obj_id)][i] for i in model_sample_list])

            # get model points in the camera coordinate
            sampled_model_pt_camera = np.add(np.dot(sampled_model_pt, R_gt.T), T_gt) # (sample_model_pt_num, 3)

            # projection and getting bbox
            proj_x = sampled_model_pt_camera[:, 0] * cam_fx / sampled_model_pt_camera[:, 2] + cam_cx
            proj_y = sampled_model_pt_camera[:, 1] * cam_fy / sampled_model_pt_camera[:, 2] + cam_cy
            cmin, cmax = min(proj_x), max(proj_x)
            rmin, rmax = min(proj_y), max(proj_y)
            img_h, img_w = label.shape

            rmin, rmax, cmin, cmax = discretize_bbox(rmin, rmax, cmin, cmax, Borderlist, img_w, img_h)

            # img_plot = cv2.rectangle(img, (rmin, rmax), (cmin, cmax), (0, 255, 0), 2)
            # cv2.imshow("img_plot", img_plot)
            # cv2.waitKey(0)
        
            # set sample points (2D) on depth/point cloud
            sample2D = mask_label[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # non-zero positions on flattened mask, 1-D array

            if len(sample2D) >= self.sample_2d_pt_num:
                sample2D = np.array(sorted(np.random.choice(sample2D, self.sample_2d_pt_num))) # randomly choose pt_num points (idx)
            elif len(sample2D) == 0:
                sample2D = np.pad(sample2D, (0, self.sample_2d_pt_num - len(sample2D)), 'constant')
            else:
                sample2D = np.pad(sample2D, (0, self.sample_2d_pt_num - len(sample2D)), 'wrap')
            
            depth_crop = depth[rmin:rmax, cmin:cmax].flatten()[sample2D][:, np.newaxis].astype(np.float32) # (pt_num, )

            xmap_crop = self.xmap[rmin:rmax, cmin:cmax].flatten()[sample2D][:, np.newaxis].astype(np.float32) # (pt_num, ) store y for sample points
            ymap_crop = self.ymap[rmin:rmax, cmin:cmax].flatten()[sample2D][:, np.newaxis].astype(np.float32) # (pt_num, ) store x for sample points
        
            # get point cloud [[px, py, pz], ...]
            pz = depth_crop / cam_scale
            px = (xmap_crop - cam_cx) * pz / cam_fx
            py = (ymap_crop - cam_cy) * pz / cam_fy
            point_cloud = np.concatenate((px, py, pz), axis=1) # (pt_num, 3) store XYZ point cloud value for sample points

            img_0 = img[rmin:rmax, cmin:cmax, :]
            img_0 = np.transpose(img_0[:, :, :3], (2, 0, 1))
            img_crop = img_0
                                                
            # # get rgb image
            # img_crop = np.transpose(img[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax] # shape (3, H, W)

            _, H, W = img_crop.shape
            img_crop_cpy = img_crop.reshape(H, W, _)

            # vis_data(img, depth_crop, label, img_crop_cpy)

            return {"img": torch.from_numpy(img),
                    "label": torch.from_numpy(label),
                    "depth": torch.from_numpy(depth.astype(np.float32)),
                    # "img_crop": torch.from_numpy(img_crop.astype(np.float32)), \
                    "img_crop": self.norm(torch.from_numpy(img_crop.astype(np.float32))), \
                    "point_cloud": torch.from_numpy(point_cloud.astype(np.float32)), \
                    "sample_2d": torch.LongTensor(sample2D.astype(np.int32)), \
                    "sampled_model_pt_camera": torch.from_numpy(sampled_model_pt_camera.astype(np.float32)), \
                    "sampled_model_pt": torch.from_numpy(sampled_model_pt.astype(np.float32)), \
                    "obj_id": torch.LongTensor([self.obj_id]), \
                    "R": torch.from_numpy(R_gt.astype(np.float32)), \
                    "T": torch.from_numpy(T_gt.astype(np.float32))}
        
        except FileNotFoundError:
            print("Desired Obj ID not available")

    def _init_config(self):
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])

        self.noise_trans = 0.02
        self.minimum_px_num = 50
        self.symmetry_obj_cls = [12, 15, 18, 19, 20]
        self.sample_model_pt_num = 500
        self.sample_2d_pt_num = 1000 # pt_num
        self.num_obj = 1

    def _init_data_info(self):
        if self.mode == 'train':
            self.data_list_path = self.Trainlist
        elif self.mode == 'test':
            self.data_list_path = self.Testlist
        else:
            raise NotImplementedError
        self.class_path = self.Classlist

        with open(self.data_list_path, 'r', encoding='utf-8') as f:
            self.all_data_dirs = f.read().splitlines()
        with open(self.class_path, 'r', encoding='utf-8') as f:
            self.classes = f.read().splitlines()


    def load_points(self, root, classes):
        
        model_points = {} # {1:[[x1, y1, z1], [x2, y2, z2], ...], 2:[...], 3:[...], ...}
        
        for cls in classes:
            cls_name = int(cls.split('.')[0].split('_')[-1])
            cls_filepath = os.path.join(root, 'ycbv_models', 'models_eval', cls)
            
            # if cls_name == 1:
            #     if os.path.isfile(cls_filepath):
            #         mesh = trimesh.load('/home/ba1071/Desktop/DTTD2/dataset/ycb_bop/YCB_video_Dataset/ycbv_models/models_eval/can.ply')
            #         model_points[str(cls_name)] = mesh.vertices.tolist()

            if os.path.isfile(cls_filepath):
                mesh = trimesh.load(cls_filepath)
                model_points[str(cls_name)] = mesh.vertices.tolist()

        return model_points
    
    def _load_image(self, index):

        rgb_name = int(self.all_data_dirs[index].split('/')[-1].split('.')[0])
        rgb_img = Image.open(self.all_data_dirs[index])   # PIL, size: (640, 480)

        return rgb_img, rgb_name
    
    def _load_depth(self, index):

        depth_path = self.all_data_dirs[index].replace("rgb", "depth").replace("jpg", "png")
        depth_img = np.array(Image.open(depth_path))   # shape: (480, 640)
        return depth_img
 
    def _load_meta(self, index):

        json_path = '/'.join(self.all_data_dirs[index].split("/")[:-2]) + "/scene_camera.json"
        with open(json_path, 'r', encoding='utf-8') as fp:
            json_data = json.load(fp)
        return json_data

    def _load_gt(self, index):

        json_path = '/'.join(self.all_data_dirs[index].split("/")[:-2]) + "/scene_gt.json"
        with open(json_path, 'r', encoding='utf-8') as fp:
            json_data = json.load(fp)
        return json_data

    def _load_labels(self, index, label_id):

        label_name = self.all_data_dirs[index].split("/")[-1].split(".")[0]
        label_name = label_name + '_' + f"{label_id:06}" + '.png'
        label_path = '/'.join(self.all_data_dirs[index].split("/")[:-1]).replace("rgb", "mask") + "/" + label_name

        label_img = np.array(Image.open(label_path))     # shape: (480, 640)

        return label_img

    def get_object_num(self):
        """ return number of objects"""
        return self.num_obj

    def get_sym_list(self):
        """ return a list of object which are symmetry"""
        return self.symmetry_obj_cls

    def get_model_point_num(self):
        """ return no of sample points in 3d model"""
        return self.sample_model_pt_num

    def get_2d_sample_num(self):
        """ return sample 2d points"""
        return self.sample_2d_pt_num

    def get_models_xyz(self):
        """ return a desired obj id model points"""
        return self.model_points

def vis_data(img, depth, label, img_crp):

    """A function to visulaize the image and depth"""

    rows = 2
    columns = 2
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(img)
    plt.axis('off')
    plt.title("img")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2) 

    # showing image
    plt.imshow(depth)
    plt.axis('off')
    plt.title("depth")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)

    # showing image
    plt.imshow(label)
    plt.axis('off')
    plt.title("label")

    fig.add_subplot(rows, columns, 4)

    plt.imshow(img_crp)
    plt.axis('off')
    plt.title("img_crop")

    plt.show()
    plt.waitforbuttonpress()
    # plt.pause(5)
    plt.close()
    
def discretize_bbox(rmin, rmax, cmin, cmax, border_list, img_w, img_h):

    """ Retruning the descrete bounding boxes """

    rmax += 1
    cmax += 1
    r_b = border_list[binary_search(border_list, rmax - rmin)]
    c_b = border_list[binary_search(border_list, cmax - cmin)]
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_h:
        delt = rmax - img_h
        rmax = img_h
        rmin -= delt
    if cmax > img_w:
        delt = cmax - img_w
        cmax = img_w
        cmin -= delt

    return rmin, rmax, cmin, cmax

def binary_search(sorted_list, target):

    """Returning the apt value"""

    l = 0
    r = len(sorted_list)-1
    while l!=r:
        mid = (l+r)>>1
        if sorted_list[mid] > target:
            r = mid
        elif sorted_list[mid] < target:
            l = mid + 1
        else:
            return mid
    return l

if __name__ == "__main__":

    dataset = YCBDataset(root="/home/ba1071/Desktop/DTTD2/dataset/ycb_bop/YCB_video_Dataset",
                         mode="train",
                         config_path="/home/ba1071/Desktop/DTTD2/dataset/ycb_bop/dataset_config/")

    for i in range(0, 4000):
        dt = dataset[i]