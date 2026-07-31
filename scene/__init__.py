#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import generate_random_poses_llff, generate_random_poses_360

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0],spiral=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.source_path = args.source_path 
        print(f"args.source_path  is {args.source_path }")
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.source_path.find('llff') != -1:
                print("############ load llff ############")
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, rand_pcd=args.rand_pcd)
            elif args.source_path.find('mipnerf360') != -1:
                print("############ load mipnerf360 ############")
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, rand_pcd=args.rand_pcd)
            elif args.source_path.find('DTU') != -1:
                print("############ load DTU ############")
                scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, args.images, args.eval, args.n_views, rand_pcd=args.rand_pcd)
            if args.source_path.find('OMMO') != -1:
                print("############ load OMMO ############")
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, rand_pcd=args.rand_pcd)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views, rand_pcd=args.rand_pcd)
        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras", resolution_scale)
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras", resolution_scale)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            if spiral:
                self.video_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.video_cameras, resolution_scale, args)


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.init_point_cloud = scene_info.point_cloud

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getVideoCameras(self, scale=1.0):
        return self.video_cameras[scale]