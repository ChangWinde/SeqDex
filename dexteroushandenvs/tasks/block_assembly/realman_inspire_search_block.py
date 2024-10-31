import numpy as np
import os

from dexteroushandenvs.tasks.hand_base.base_task import BaseTask
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgym import gymtorch
import torch
import random
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from PIL import Image as Im
import time, datetime
from collections import deque
import cv2
import math
import pickle
import wandb
import warnings

class InspireSearchBlock(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=None, is_multi_agent=None):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        
        self.num_realman_dofs = 7
        self.num_inspire_dofs = 12
        self.actuated_dof_names = ["R_thumb_MCP_joint1", "R_thumb_MCP_joint2", "R_index_MCP_joint", "R_middle_MCP_joint", "R_ring_MCP_joint", "R_pinky_MCP_joint"]
        self.actuated_dof_indices = [7, 8, 11, 13, 15, 17]
        
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.arm_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.hand_reset_step = self.cfg["env"]["handResetStep"]
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.lego_curriculum_interval = self.cfg["env"]["legoCurriculumInterval"]
        self.enable_camera_sensors = self.cfg["env"]["enable_camera_sensors"]
        self.enable_lego_curriculum = self.cfg["env"]["enable_lego_curriculum"]
        self.enable_wandb = self.cfg["env"]["enable_wandb"]
        self.num_lego_suit = self.cfg["env"]["num_lego_suit"]
        self.num_max_suit = self.cfg["env"]["num_max_suit"]
        self.num_max_suit = self.num_max_suit if self.enable_lego_curriculum else self.num_lego_suit
        self.num_lego_block = self.num_max_suit*8 + 10*6
        # check
        self.cfg["env"]["numObservations"] = 175 - 24
        self.cfg["env"]["numStates"] = 175 - 24
        self.cfg["env"]["numActions"] = 19
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.up_axis = 'z'
        self.vel_obs_scale = 0.2
        self.hidden_height = 200
        self.record_completion_time = False
        # setup parameters above
        
        super().__init__(cfg=self.cfg, enable_camera_sensors=self.enable_camera_sensors)
        
        # setup viewer
        if self.viewer != None:
            # finger view
            # cam_pos = gymapi.Vec3(1, 0.0, 1.1)
            # cam_target = gymapi.Vec3(-0.7, 0.0, 0.5)
            cam_pos = gymapi.Vec3(1, -0.1, 1.5)
            cam_target = gymapi.Vec3(-0.7, -0.1, 0.5)
            # left
            # cam_pos = gymapi.Vec3(-0.26, -1.5, 1.2)
            # cam_target = gymapi.Vec3(-0.26,1 , 0.7)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "arm_hand"))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        # unit tensors
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        # arm_hand default dof pos
        self.arm_hand_default_dof_pos = torch.zeros(self.num_arm_hand_dofs, dtype=torch.float, device=self.device)
        self.arm_hand_default_dof_pos[:7] = to_torch([-.1, 0.45, 0.0, 1.78, 0.0, -0.5, -1.571] , dtype=torch.float, device=self.device)     
        self.arm_hand_default_dof_pos[7:] = to_torch([0.0, 0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0], dtype=torch.float, device=self.device)

        # arm_hand prepare dof pos
        self.arm_hand_prepare_dof_pos = torch.zeros(self.num_arm_hand_dofs, dtype=torch.float, device=self.device)
        self.arm_hand_prepare_dof_pos[:7] = to_torch([0.3, 0.45, 0.0, 1.78, 0.0, -0.5, -1.571] , dtype=torch.float, device=self.device)
        self.arm_hand_prepare_dof_pos[7:] = to_torch([0.0, 0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0], dtype=torch.float, device=self.device)

        # arm_hand dof state
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        # rigid_body state
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_rigid_body = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.E_prev = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_indices[0], "Link7", gymapi.DOMAIN_ENV)
        self.seg_start_pos = self.root_state_tensor[self.seg_indices, 0:3].clone()
        self.seg_start_rot = self.root_state_tensor[self.seg_indices, 3:7].clone()
        self.seg_pos = self.root_state_tensor[self.seg_indices, 0:3].clone()
        self.seg_rot = self.root_state_tensor[self.seg_indices, 3:7].clone()
        self.base_pos = self.rigid_body_states[:, 0, 0:3].clone()
        self.episodes = 0
        self.total_successes = 0
        self.total_resets = 0
        self.total_steps = 0
        
        # Print information of this environment
        print("Num envs: ", self.num_envs)
        print("Num bodies: ", self.num_rigid_body)
        print("Num dofs: ", self.num_dofs)
        print("Num arm hand dofs: ", self.num_arm_hand_dofs)
        print("Contact Tensor Dimension", self.contact_tensor.shape)
        print("hand_base_rigid_body_index: ", self.hand_base_rigid_body_index)
        
        self.extras = {'dist_reward': 0, 'action_penalty': 0, 'lego_up_reward': 0, "pose_reward": 0, "angle_reward": 0, "emergency_reward": 0,
                       'z_lift': 0, 'xy_move': 0, 'success_length': self.max_episode_length}
        self.ema_success = deque(maxlen=10)
        self._init_wandb()
    
    def _init_wandb(self):
        project = "DexterousHand"
        group = self.cfg["env"]["env_name"]
        name = datetime.datetime.strftime(datetime.datetime.now(), '%m%d') + f"_Search"
        monitor_config = flatten_dict(self.cfg)
        if self.enable_wandb:
            wandb.init(project=project, 
                    entity='tempholder', 
                    group=group, 
                    name=name,
                    config=monitor_config,
                    save_code=True)      
        
    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim_params.physx.max_gpu_contact_pairs = int(self.sim_params.physx.max_gpu_contact_pairs)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
    
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self, num_envs, env_spacing, num_envs_per_row):
        lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
        
        # =================== Set up the arm hand =================== #
        arm_hand_asset_file = "mjcf/realman_mjcf/realman_inspire_mjmodel.xml"
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.angular_damping = 0.01
        asset_options.thickness = 0.001
        asset_options.use_physx_armature = True if self.physics_engine == gymapi.SIM_PHYSX else False
        arm_hand_asset = self.gym.load_asset(self.sim, asset_root, arm_hand_asset_file, asset_options)
        self.num_arm_hand_shapes = self.gym.get_asset_rigid_shape_count(arm_hand_asset)
        self.num_arm_hand_bodies = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.num_arm_hand_dofs = self.gym.get_asset_dof_count(arm_hand_asset)
        self.num_arm_hand_actuators = self.num_arm_hand_dofs 
        print("num_arm_hand_shapes: ", self.num_arm_hand_shapes)
        print("num_arm_hand_bodies: ", self.num_arm_hand_bodies)
        print("Num dofs: ", self.num_arm_hand_dofs)
        
        self.arm_hand_dof_default_pos = []
        self.arm_hand_dof_default_vel = [0.0 for _ in range(self.num_arm_hand_dofs)]
        robot_lower_qpos = []
        robot_upper_qpos = []
        arm_hand_dof_props = self.gym.get_asset_dof_properties(arm_hand_asset)
        
        self.realman_inspire_dof_lower_limits = [-3.1, -2.268, -3.1, -2.355, -3.1, -2.233, -6.28, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.realman_inspire_dof_upper_limits = [3.1, 2.268, 3.1, 2.355, 3.1, 2.233, 6.28, 1.3, 0.5, 1.0, 1.2, 1.7, 1.6, 1.7, 1.6, 1.7, 1.6, 1.7, 1.6]
        realman_dof_stiffness_list = [200, 200, 100, 100, 50, 50, 50]
        realman_dof_damping_list = [20, 20, 10, 10, 10, 5, 5]
        realman_dof_effort_list = [60, 60, 30, 30, 10, 10, 10]
        realman_dof_velocity_list = [1, 1, 1, 1, 1, 1, 1]
        for dof_id in range(self.num_arm_hand_dofs):
            arm_hand_dof_props['driveMode'][dof_id] = gymapi.DOF_MODE_POS
            # setup arm
            if dof_id < self.num_realman_dofs:
                arm_hand_dof_props['stiffness'][dof_id] = realman_dof_stiffness_list[dof_id]
                arm_hand_dof_props['damping'][dof_id] = realman_dof_damping_list[dof_id]
                arm_hand_dof_props['effort'][dof_id] = realman_dof_effort_list[dof_id]
                arm_hand_dof_props['velocity'][dof_id] = realman_dof_velocity_list[dof_id]
            robot_lower_qpos.append(self.realman_inspire_dof_lower_limits[dof_id])
            robot_upper_qpos.append(self.realman_inspire_dof_upper_limits[dof_id])
            
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(robot_lower_qpos, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(robot_upper_qpos, device=self.device)
        self.arm_hand_dof_lower_qvel = to_torch(-arm_hand_dof_props["velocity"], device=self.device)
        self.arm_hand_dof_upper_qvel = to_torch(arm_hand_dof_props["velocity"], device=self.device)
        self.arm_hand_dof_default_pos = to_torch(self.arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)
        
        arm_hand_start_pose = gymapi.Transform()
        arm_hand_start_pose.p = gymapi.Vec3(-0.3, 0.0, 0.6)
        arm_hand_start_pose.r = gymapi.Quat(0, 0, 1, 0)
        # =================== the arm hand End =================== #
        
        
        # =================== Set up the table =================== #
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.thickness = 0.001
        table_size = gymapi.Vec3(1.5, 1.0, 0.6)

        table_asset = self.gym.create_box(self.sim, table_size.x, table_size.y, table_size.z, table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_size.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.0)
        # =================== the table End =================== #
        
        
        # =================== Set up the Box =================== #
        box_assets = []
        box_start_poses = []

        box_thin = 0.01
        box_xyz = [0.60, 0.416, 0.165]
        box_offset = [0.25, 0.19, 0]

        box_asset_options = gymapi.AssetOptions()
        box_asset_options.disable_gravity = False
        box_asset_options.fix_base_link = True
        box_asset_options.flip_visual_attachments = True
        box_asset_options.collapse_fixed_joints = True
        box_asset_options.disable_gravity = True
        box_asset_options.thickness = 0.001

        box_bottom_asset = self.gym.create_box(self.sim, box_xyz[0], box_xyz[1], box_thin, table_asset_options)
        box_left_asset = self.gym.create_box(self.sim, box_xyz[0], box_thin, box_xyz[2], table_asset_options)
        box_right_asset = self.gym.create_box(self.sim, box_xyz[0], box_thin, box_xyz[2], table_asset_options)
        box_former_asset = self.gym.create_box(self.sim, box_thin, box_xyz[1], box_xyz[2], table_asset_options)
        box_after_asset = self.gym.create_box(self.sim, box_thin, box_xyz[1], box_xyz[2], table_asset_options)

        box_bottom_start_pose = gymapi.Transform()
        box_bottom_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_thin) / 2)
        box_left_start_pose = gymapi.Transform()
        box_left_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], (box_xyz[1] - box_thin) / 2 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_right_start_pose = gymapi.Transform()
        box_right_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], -(box_xyz[1] - box_thin) / 2 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_former_start_pose = gymapi.Transform()
        box_former_start_pose.p = gymapi.Vec3((box_xyz[0] - box_thin) / 2 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_after_start_pose = gymapi.Transform()
        box_after_start_pose.p = gymapi.Vec3(-(box_xyz[0] - box_thin) / 2 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_xyz[2]) / 2)

        box_assets.append(box_bottom_asset)
        box_assets.append(box_left_asset)
        box_assets.append(box_right_asset)
        box_assets.append(box_former_asset)
        box_assets.append(box_after_asset)
        box_start_poses.append(box_bottom_start_pose)
        box_start_poses.append(box_left_start_pose)
        box_start_poses.append(box_right_start_pose)
        box_start_poses.append(box_former_start_pose)
        box_start_poses.append(box_after_start_pose)
        # =================== the box End =================== #
        
        
        # =================== Set up the Lego =================== #
        lego_path = "urdf/blender/urdf/"
        lego_files = ['1x2.urdf', '1x2_curve.urdf', '1x3_curve_soft.urdf', '1x3_curve.urdf', '1x1.urdf', '1x3.urdf', '1x4.urdf', '2x2_curve_soft.urdf']
        lego_assets = []
        lego_start_poses = []
        self.segmentation_id = 1

        # lego in box
        for iter_n in range(self.num_max_suit):
            for idx, lego_file_name in enumerate(lego_files):
                lego_asset = create_lego_asset(self.sim, self.gym, asset_root, lego_path, lego_file_name, fix_base_link=None)
                x_offset = 0.17 * (idx % 3)
                y_offset = 0.11 * (idx // 3)
                z_height = 0.68 + iter_n * 0.06
                
                if iter_n % 2 == 0:
                    x_pos = -0.17 + x_offset + 0.25
                    y_pos = -0.11 + y_offset + 0.19
                else:
                    x_pos = 0.17 - x_offset + 0.25
                    y_pos = 0.11 - y_offset + 0.19
                if iter_n >= self.num_lego_suit: z_height += self.hidden_height
                lego_start_pose = create_lego_pose(x_pos, y_pos, z_height, 0.0, 0.0, 0.785)
                lego_assets.append(lego_asset)
                lego_start_poses.append(lego_start_pose)
        # legos which are located on the bottom of box
        # Flat lego asset options
        flat_lego_asset_options = gymapi.AssetOptions()
        flat_lego_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        flat_lego_asset_options.disable_gravity = False
        flat_lego_asset_options.fix_base_link = True
        flat_lego_asset_options.thickness = 0.001

        ran_list = [0, 0, 0, 1, 2, 2]
        lego_list = [0, 5, 6]
        bianchang = [0.03, 0.045, 0.06]
        flat_lego_begin = len(lego_assets)
        for iter_n in range(10):
            random.shuffle(ran_list)
            lego_center = [0.254 - bianchang[ran_list[0]] + 0.25, 0.175 + 0.19 - 0.039 * iter_n, 0.63]
            for rank_idx in range(6):
                lego_file_name = lego_files[lego_list[ran_list[rank_idx]]]
                lego_asset = create_lego_asset(self.sim, self.gym, asset_root, lego_path, lego_file_name, fix_base_link=True, thickness=0.001)
                lego_start_pose = create_lego_pose(*lego_center)
                lego_assets.append(lego_asset)
                lego_start_poses.append(lego_start_pose)
                lego_center[0] -= bianchang[ran_list[rank_idx]] + (bianchang[ran_list[rank_idx+1]] if rank_idx < 5 else 0) + 0.006
        flat_lego_end = len(lego_assets)
        
        extra_lego_assets = []
        extra_lego_asset_options = gymapi.AssetOptions()
        extra_lego_asset_options.disable_gravity = False
        extra_lego_asset_options.fix_base_link = True

        # fake extra lego
        extra_lego_asset = self.gym.load_asset(self.sim, asset_root, "urdf/blender/assets_for_insertion/urdf/12x12x1_real.urdf", extra_lego_asset_options)
        extra_lego_assets.append(extra_lego_asset)

        extra_lego_start_pose = gymapi.Transform()
        extra_lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.0)
        extra_lego_start_pose.p = gymapi.Vec3(0.25, -0.35, 0.618)
        # =================== the Lego End =================== #
        
        self.hand_start_states = []
        self.seg_start_states = []
        self.lego_start_states = []
        self.extra_lego_start_states = []
        self.segmentation_id_list = []
        self.robot_indices, self.table_indices, self.lego_indices, self.extra_obj_indices, self.seg_indices = [], [], [], [], []
        self.actor_index = {}
        self.envs = []
        # something for camera
        self.cameras = []
        self.camera_tensors = []
        self.camera_seg_tensors = []
        self.camera_view_matrixs = []
        self.camera_proj_matrixs = []
        
        # compute aggregate size
        max_agg_bodies = self.num_arm_hand_bodies + 2 + 1 + len(lego_assets) + 5 + 10 
        max_agg_shapes = self.num_arm_hand_shapes + 2 + 1 + len(lego_assets) + 5 + 100 
        # =================== Create Actor =================== #
        for env_id in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_envs_per_row)
            
            if self.aggregate_mode >= 1: self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            
            # Arm hand
            arm_hand_actor = self.gym.create_actor(env_ptr, arm_hand_asset, arm_hand_start_pose, "arm_hand", env_id, 0, 0)
            hand_idx = self.gym.get_actor_index(env_ptr, arm_hand_actor, gymapi.DOMAIN_SIM)
            self.robot_indices.append(hand_idx)
            self.hand_start_states.append([arm_hand_start_pose.p.x,
                                           arm_hand_start_pose.p.y,
                                           arm_hand_start_pose.p.z,
                                           arm_hand_start_pose.r.x,
                                           arm_hand_start_pose.r.y,
                                           arm_hand_start_pose.r.z,
                                           arm_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, arm_hand_actor, arm_hand_dof_props)
            arm_hand_actor_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, arm_hand_actor)
            for _, arm_hand_actor_shape_prop in enumerate(arm_hand_actor_shape_props):
                arm_hand_actor_shape_prop.friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, arm_hand_actor, arm_hand_actor_shape_props)
            
            # Table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", env_id, -1, 0)
            table_idx = self.gym.get_actor_index(env_ptr, table_actor, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.9, 0.8))
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
            for object_shape_prop in table_shape_props:
                object_shape_prop.friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_shape_props)
            
            # Box
            for box_id, box_asset in enumerate(box_assets):
                box_handle = self.gym.create_actor(env_ptr, box_asset, box_start_poses[box_id], "box_{}".format(box_id), env_id, 0, 0)
                self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))

            # Lego Block
            color_map = [
                [0.80, 0.64, 0.20], [0.13, 0.54, 0.13], [0, 0.4, 0.8], [1.0, 0.54, 0.0], 
                [0.69, 0.13, 0.13], [0.69, 0.13, 0.13], [0, 0.4, 0.8], [0.8, 0.64, 0.2]
            ]
            self.segmentation_id = env_id % 8
            if self.segmentation_id in [3, 4, 7]:
                self.segmentation_id = 0

            lego_idx = []
            for lego_i, lego_asset in enumerate(lego_assets):
                lego_handle = self.gym.create_actor(env_ptr, lego_asset, lego_start_poses[lego_i], f"lego_{lego_i}", env_id, 0, lego_i + 1)
                self.lego_start_states.append([
                    lego_start_poses[lego_i].p.x, lego_start_poses[lego_i].p.y, lego_start_poses[lego_i].p.z,
                    lego_start_poses[lego_i].r.x, lego_start_poses[lego_i].r.y, lego_start_poses[lego_i].r.z, lego_start_poses[lego_i].r.w,
                    0, 0, 0, 0, 0, 0
                ])
                lego_block_idx = self.gym.get_actor_index(env_ptr, lego_handle, gymapi.DOMAIN_SIM)
                if lego_i == self.segmentation_id:
                    self.segmentation_id_list.append(lego_i + 1)
                    self.seg_indices.append(lego_block_idx)
                lego_idx.append(lego_block_idx)
                
                # setup lego mass
                lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, lego_handle)
                for lego_body_prop in lego_body_props:
                    if flat_lego_begin < lego_i < flat_lego_end:
                        lego_body_prop.mass *= 1
                self.gym.set_actor_rigid_body_properties(env_ptr, lego_handle, lego_body_props)

                # setup lego color
                color = color_map[lego_i % 8]
                if flat_lego_begin < lego_i < flat_lego_end:
                    color = color_map[random.randint(0, 7)]
                self.gym.set_rigid_body_color(env_ptr, lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))

            self.lego_indices.append(lego_idx)

            extra_lego_handle = self.gym.create_actor(env_ptr, extra_lego_assets[0], extra_lego_start_pose, "extra_lego", env_id, 0, 0)
            self.extra_lego_start_states.append([
                extra_lego_start_pose.p.x, extra_lego_start_pose.p.y, extra_lego_start_pose.p.z,
                extra_lego_start_pose.r.x, extra_lego_start_pose.r.y, extra_lego_start_pose.r.z, extra_lego_start_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            extra_object_idx = self.gym.get_actor_index(env_ptr, extra_lego_handle, gymapi.DOMAIN_SIM)
            extra_lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, extra_lego_handle)
            self.gym.set_rigid_body_color(env_ptr, extra_lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))
            self.extra_obj_indices.append(extra_object_idx)
            
            
            if self.enable_camera_sensors:
                # camera properties
                self.camera_props = gymapi.CameraProperties()
                self.camera_props.width = 128
                self.camera_props.height = 128
                self.camera_props.enable_tensors = True
                self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
                # self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
                # self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)
                # self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

                self.camera_offset_quat = gymapi.Quat().from_euler_zyx(0, - 3.141 + 0.5, 1.571)
                self.camera_offset_quat = to_torch([self.camera_offset_quat.x, self.camera_offset_quat.y, self.camera_offset_quat.z, self.camera_offset_quat.w], device=self.device)
                self.camera_offset_pos = to_torch([0.03, 0.107 - 0.098, 0.067 + 0.107], device=self.device)
                
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.35, 0.19, 1.0), gymapi.Vec3(0.2, 0.19, 0))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                camera_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
                torch_cam_seg_tensor = gymtorch.wrap_tensor(camera_seg_tensor)

                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)
                self.mount_rigid_body_index = self.gym.find_actor_rigid_body_index(env_ptr, arm_hand_actor, "Link7", gymapi.DOMAIN_ENV)
                
                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[env_id][0] = origin.x
                self.env_origin[env_id][1] = origin.y
                self.env_origin[env_id][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_seg_tensors.append(torch_cam_seg_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            
            self.envs.append(env_ptr)
        # =================== Actor End =================== #
        
        # =================== Varible  =================== #
        self.fingertip_names = ["R_thumb_distal", "R_index_distal", "R_middle_distal", "R_ring_distal", "R_pinky_distal"]
        self.fingertip_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, name) for name in self.fingertip_names]
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.lego_start_states = to_torch(self.lego_start_states, device=self.device).view(self.num_envs, len(lego_assets), 13)
        self.seg_start_states = to_torch(self.seg_start_states, device=self.device)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.robot_indices = to_torch(self.robot_indices, dtype=torch.long, device=self.device)
        self.lego_indices = to_torch(self.lego_indices, dtype=torch.long, device=self.device)
        self.seg_indices = to_torch(self.seg_indices, dtype=torch.long, device=self.device)
        # camera
        self.emergence_reward = torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)
        self.emergence_pixel = torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)
        self.last_emergence_pixel = torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)
        self.emergence_reward_buf = torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)     
    
    def compute_observation(self):
        # refresh the anything
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        
        # camera setup
        if self.enable_camera_sensors and self.progress_buf[0] >= self.max_episode_length - 1:
            pos = self.arm_hand_default_dof_pos
            self.arm_hand_dof_pos[:, :] = pos[:]
            self.arm_hand_dof_vel[:, :] = self.arm_hand_dof_default_vel
            self.prev_targets[:, :self.num_arm_hand_dofs] = pos
            self.cur_targets[:, :self.num_arm_hand_dofs] = pos

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(self.robot_indices.to(torch.int32)), self.num_envs)

            for i in range(1):
                self.render()
                self.gym.simulate(self.sim)

            self.render_for_camera()
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            self.compute_emergence_reward(self.camera_tensors, self.camera_seg_tensors, segmentation_id_list=self.segmentation_id_list)
            self.all_lego_brick_pos = self.root_state_tensor[self.lego_indices[:].view(-1), 0:3].clone().view(self.num_envs, -1, 3)
            # self.compute_heap_movement_penalty(self.all_lego_brick_pos)

            camera_rgba_image = camera_rgb_visulization(self.camera_tensors, env_id=0, is_depth_image=False)
            camera_seg_image = camera_segmentation_visulization(self.segmentation_id_list, self.camera_tensors, self.camera_seg_tensors, env_id=0, is_depth_image=False)

            cv2.namedWindow("DEBUG_RGB_VIS", 0)
            cv2.namedWindow("DEBUG_SEG_VIS", 0)

            cv2.imshow("DEBUG_RGB_VIS", camera_rgba_image)
            cv2.imshow("DEBUG_SEG_VIS", camera_seg_image)
            cv2.waitKey(1)
        
        # define the observation
        # hand
        self.hand_base_pose = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:7]
        self.hand_base_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3]
        self.hand_base_rot = self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7]
        self.hand_base_linvel = self.rigid_body_states[:, self.hand_base_rigid_body_index, 7:10]
        self.hand_base_angvel = self.rigid_body_states[:, self.hand_base_rigid_body_index, 10:13]
        
        #robot
        self.robot_base_pos = self.root_state_tensor[self.robot_indices, 0:3]
        self.robot_base_rot = self.root_state_tensor[self.robot_indices, 3:7]

        # lego_block
        self.seg_pose = self.root_state_tensor[self.seg_indices, 0:7]
        self.seg_pos = self.root_state_tensor[self.seg_indices, 0:3]
        self.seg_rot = self.root_state_tensor[self.seg_indices, 3:7]
        self.seg_linvel = self.root_state_tensor[self.seg_indices, 7:10]
        self.seg_angvel = self.root_state_tensor[self.seg_indices, 10:13]
        
        # five fingers
        self.finger_thumb_pos = self.rigid_body_states[:, self.fingertip_handles[0], 0:3]
        self.finger_thumb_rot = self.rigid_body_states[:, self.fingertip_handles[0], 3:7]
        self.finger_thumb_linvel = self.rigid_body_states[:, self.fingertip_handles[0], 7:10]
        self.finger_thumb_angvel = self.rigid_body_states[:, self.fingertip_handles[0], 10:13]
        self.finger_index_pos = self.rigid_body_states[:, self.fingertip_handles[1], 0:3]
        self.finger_index_rot = self.rigid_body_states[:, self.fingertip_handles[1], 3:7]
        self.finger_index_linvel = self.rigid_body_states[:, self.fingertip_handles[1], 7:10]
        self.finger_index_angvel = self.rigid_body_states[:, self.fingertip_handles[1], 10:13]
        self.finger_middle_pos = self.rigid_body_states[:, self.fingertip_handles[2], 0:3]
        self.finger_middle_rot = self.rigid_body_states[:, self.fingertip_handles[2], 3:7]
        self.finger_middle_linvel = self.rigid_body_states[:, self.fingertip_handles[2], 7:10]
        self.finger_middle_angvel = self.rigid_body_states[:, self.fingertip_handles[2], 10:13]
        self.finger_ring_pos = self.rigid_body_states[:, self.fingertip_handles[3], 0:3]
        self.finger_ring_rot = self.rigid_body_states[:, self.fingertip_handles[3], 3:7]
        self.finger_ring_linvel = self.rigid_body_states[:, self.fingertip_handles[3], 7:10]
        self.finger_ring_angvel = self.rigid_body_states[:, self.fingertip_handles[3], 10:13]
        self.finger_pinky_pos = self.rigid_body_states[:, self.fingertip_handles[4], 0:3]
        self.finger_pinky_rot = self.rigid_body_states[:, self.fingertip_handles[4], 3:7]
        self.finger_pinky_linvel = self.rigid_body_states[:, self.fingertip_handles[4], 7:10]
        self.finger_pinky_angvel = self.rigid_body_states[:, self.fingertip_handles[4], 10:13]
        
        self.finger_thumb_pos += quat_apply(self.finger_thumb_rot[:], to_torch([0, 0.5, 0.1], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.finger_index_pos += quat_apply(self.finger_index_rot[:], to_torch([0.18, 0.9, 0.1], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.finger_middle_pos+= quat_apply(self.finger_middle_rot[:], to_torch([0.15, 0.9, 0.1],device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.finger_ring_pos  += quat_apply(self.finger_ring_rot[:], to_torch([0.2, 0.9, 0.1],  device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.finger_pinky_pos += quat_apply(self.finger_pinky_rot[:], to_torch([0.2, 0.8, 0.1], device=self.device).repeat(self.num_envs, 1) * 0.04)
        
        # add ball to visualize the finger tip
        self.compute_contact_asymmetric_observations()
        
        # object 6d pose randomization
        if self.enable_camera_sensors:
            self.mount_pos = self.rigid_body_states[:, self.mount_rigid_body_index, 0:3]
            self.mount_rot = self.rigid_body_states[:, self.mount_rigid_body_index, 3:7]
            self.q_camera, self.p_camera = tf_combine(self.mount_rot, self.mount_pos, self.camera_offset_quat.repeat(self.num_envs, 1), self.camera_offset_pos.repeat(self.num_envs, 1))
            self.q_camera_inv, self.p_camera_inv = tf_inverse(self.q_camera, self.p_camera)
            self.camera_view_segmentation_target_rot, self.camera_view_segmentation_target_pos = tf_combine(self.q_camera_inv, self.p_camera_inv, self.seg_rot, self.seg_pos)
        
        if self.enable_camera_sensors and self.progress_buf[0] % self.hand_reset_step == 0 and self.progress_buf[0] != 0:
            self.gym.end_access_image_tensors(self.sim)
    
    def compute_contact_asymmetric_observations(self):
        # 0:19 => hand_arm_dof_pos 
        start_index = 0
        self.states_buf[:, :self.num_arm_hand_dofs] = unscale(
            self.arm_hand_dof_pos[:, :self.num_arm_hand_dofs],
            self.arm_hand_dof_lower_limits[:self.num_arm_hand_dofs],
            self.arm_hand_dof_upper_limits[:self.num_arm_hand_dofs]
        )
        start_index += self.num_arm_hand_dofs
        
        # 19:38 => hand_arm_dof_vel
        self.states_buf[:, start_index:start_index + self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel[:, :self.num_arm_hand_dofs]
        start_index += self.num_arm_hand_dofs

        # 38:53 =>  five fingers pos
        self.states_buf[:, start_index+0:start_index+3] = self.finger_thumb_pos - self.seg_pos 
        self.states_buf[:, start_index+3:start_index+6] = self.finger_index_pos - self.seg_pos
        self.states_buf[:, start_index+6:start_index+9] = self.finger_middle_pos - self.seg_pos 
        self.states_buf[:, start_index+9:start_index+12] = self.finger_ring_pos - self.seg_pos 
        self.states_buf[:, start_index+12:start_index+15] = self.finger_pinky_pos - self.seg_pos 
        self.middle_point = (self.finger_thumb_pos + self.finger_index_pos + self.finger_middle_pos) / 3
        self.states_buf[:, start_index+15:start_index+18] = self.middle_point - self.seg_pos
        start_index = start_index + 6 * 3
        
        # 53:72 => hand_arm action
        self.states_buf[:, start_index:start_index + self.num_arm_hand_dofs] = self.actions
        start_index += self.num_arm_hand_dofs
        
        # 72:86 => hand_base_pose & lego_pose
        self.states_buf[:, start_index:start_index+7] = self.hand_base_pose
        self.states_buf[:, start_index+7:start_index+14] = self.seg_pose
        start_index += 14

        # 86:100 => history hand pos
        # self.states_buf[:, start_index+0:start_index+3] = self.hand_pos_history_0
        # self.states_buf[:, start_index+3:start_index+6] = self.hand_pos_history_1
        # self.states_buf[:, start_index+6:start_index+9] = self.hand_pos_history_2
        # self.states_buf[:, start_index+9:start_index+12] = self.hand_pos_history_3
        # self.states_buf[:, start_index+12:start_index+15] = self.hand_pos_history_4
        # self.states_buf[:, start_index+15:start_index+18] = self.hand_pos_history_5
        # self.states_buf[:, start_index+18:start_index+21] = self.hand_pos_history_6
        # self.states_buf[:, start_index+21:start_index+24] = self.hand_pos_history_7
        # start_index += 8*3

        # 100:103 => segmentation_object
        # self.states_buf[:, start_index+0:start_index+1] = self.lego_center_point_x / 128
        # self.states_buf[:, start_index+1:start_index+2] = self.lego_center_point_y / 128
        # self.states_buf[:, start_index+2:start_index+3] = self.lego_point_num / 100
        # start_index += 3

        # 103:109 => hand_base linvel & angvel
        self.states_buf[:, start_index:start_index+3] = self.hand_base_linvel
        self.states_buf[:, start_index+3:start_index+6] = self.hand_base_angvel
        start_index += 6

        # 109:119 => 
        self.states_buf[:, start_index+0:start_index+4 ] = self.finger_thumb_rot  
        self.states_buf[:, start_index+4:start_index+7 ] = self.finger_thumb_linvel
        self.states_buf[:, start_index+7:start_index+10] = self.finger_thumb_angvel
        start_index += 10
        
        self.states_buf[:, start_index+0:start_index+4 ] = self.finger_index_rot  
        self.states_buf[:, start_index+4:start_index+7 ] = self.finger_index_linvel
        self.states_buf[:, start_index+7:start_index+10] = self.finger_index_angvel
        start_index += 10

        self.states_buf[:, start_index+0:start_index+4 ] = self.finger_middle_rot
        self.states_buf[:, start_index+4:start_index+7 ] = self.finger_middle_linvel
        self.states_buf[:, start_index+7:start_index+10] = self.finger_middle_angvel
        start_index += 10

        self.states_buf[:, start_index+0:start_index+4 ] = self.finger_ring_rot
        self.states_buf[:, start_index+4:start_index+7 ] = self.finger_ring_linvel
        self.states_buf[:, start_index+7:start_index+10] = self.finger_ring_angvel
        start_index += 10
        
        self.states_buf[:, start_index+0:start_index+4 ] = self.finger_pinky_rot
        self.states_buf[:, start_index+4:start_index+7 ] = self.finger_pinky_linvel
        self.states_buf[:, start_index+7:start_index+10] = self.finger_pinky_angvel
        start_index += 10

        self.states_buf[:, start_index+0:start_index+3] = self.seg_angvel
        self.states_buf[:, start_index+3:start_index+6] = self.seg_linvel
        
        self.obs_buf = self.states_buf

    def compute_emergence_reward(self, camera_tensors, camera_seg_tensors, segmentation_id_list=0):
        for i in range(self.num_envs):
            torch_seg_tensor = camera_seg_tensors[i]
            self.emergence_pixel[i] = torch_seg_tensor[torch_seg_tensor == segmentation_id_list[i]].shape[0]
        self.emergence_reward = (self.emergence_pixel - self.last_emergence_pixel) / 100
        # print("emergence_reward: ", self.emergence_reward[0])
        self.last_emergence_pixel = self.emergence_pixel.clone()
    
    def compute_reward(self):
        # define distance between fingers and lego
        fingers_pos = [self.finger_thumb_pos, self.finger_index_pos]
        finger_dist = sum([torch.norm(self.seg_pos - finger_pos, p=2, dim=-1) for finger_pos in fingers_pos])
        distance_reward = 1.0 * torch.exp(- 5 * torch.clamp(finger_dist - 0.15, 0, None))
        
        # define grasp pose reward 
        grasp_fingers_pos = [
            self.middle_point
        ]
        pose_dist = sum([tolerance(point_, self.seg_pos, 0.016, 0.01) for point_ in grasp_fingers_pos]) / len(grasp_fingers_pos)
        pose_reward = pose_dist * 6
        
        # define angle reward
        angle_finger = [self.finger_index_pos, self.finger_middle_pos]
        angle_dist  = sum([compute_angle_line_plane(self.finger_thumb_pos, finger, self.z_unit_tensor) for finger  in angle_finger]) / len(angle_finger)
        angle_reward = distance_reward * torch.exp(-1.0 * torch.abs(angle_dist)) * 0.5
        
        # define grasp reward
        target_lift_height = 0.3
        z_lift = torch.abs(self.seg_start_pos[:, 2] + target_lift_height - self.seg_pos[:, 2])
        xy_move = torch.norm(self.seg_start_pos[:, 0:2] - self.seg_pos[:, 0:2], p=2, dim=-1)
        target_pos = self.seg_start_pos.clone() + torch.tensor([0, 0, target_lift_height]).repeat(self.num_envs, 1).to(self.device)
        goal_dist = torch.norm(self.seg_pos - target_pos, p=2, dim=-1)
        lift_reward = pose_dist * 400 * torch.clamp((target_lift_height- goal_dist), -0.05, None)
        
        # define emergency reward
        emergency_reward = self.emergence_reward
        # curr_pixel_diff = torch.cat((emergency_reward, torch.zeros((self.num_envs, 2))), dim=1)
        # emergency_reward = tolerance(curr_pixel_diff, torch.zeros_like(self.emergence_reward), 0.1, 0.1)
        
        # Penalize actions
        action_penalty = 0.001 * torch.sum(self.actions ** 2, dim=-1)
        
        total_reward = (distance_reward + pose_reward + lift_reward + angle_reward + emergency_reward - self.E_prev) - action_penalty
        
        self.extras['dist_reward'] += distance_reward[0]
        self.extras['pose_reward'] += pose_reward[0]
        self.extras['lego_up_reward'] += lift_reward[0]
        self.extras['angle_reward'] += angle_reward[0]
        self.extras['action_penalty'] += action_penalty[0]
        self.extras['emergency_reward'] += emergency_reward[0]
        self.extras['z_lift'] = z_lift[0]
        self.extras['xy_move'] = xy_move[0]
        self.extras['success'] = 100.0 if self.seg_pos[0, 2] > self.seg_start_pos[0, 2] + 0.8*target_lift_height else 0.0
        # self.extras['success'] = 100.0 if self.seg_pos[0, 2] > self.seg_start_pos[0, 2] + 0.5*target_lift_height else 0.0
        # self.extras['success'] = 100.0 if self.seg_pos[0, 2] > 0.9 else 0.0
        if self.extras['success'] > 0 and self.extras['success_length'] == self.max_episode_length:
            self.extras['success_length'] = self.progress_buf[0]

        self.E_prev = distance_reward + pose_reward + lift_reward + angle_reward + emergency_reward
        
        resets = torch.where(finger_dist <= -1, torch.ones_like(self.reset_buf), self.reset_buf)
        resets = torch.where(self.progress_buf[:] > self.max_episode_length, torch.ones_like(resets), resets)
        self.rew_buf[:], self.reset_buf[:] = total_reward, resets
        
    def reset_envs(self, env_ids):
        if self.enable_lego_curriculum and self.episodes % self.lego_curriculum_interval == 0 and self.episodes > 0 and self.num_lego_suit < self.num_max_suit:
            print("Increase the lego blocks, current lego suit: ", (self.num_lego_suit+1)*8)
            self.lego_start_states[:, self.num_lego_suit*8: (self.num_lego_suit+1)*8, 2] -= self.hidden_height
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.num_lego_suit += 1
        
        # record the trajectory
        if self.record_completion_time:
            self.end_time = time.time()
            self.complete_time_list.append(self.end_time - self.last_start_time)
            self.last_start_time = self.end_time
            print("complete_time_mean: ", np.array(self.complete_time_list).mean())
            print("complete_time_std: ", np.array(self.complete_time_list).std())
            if len(self.complete_time_list) == 25:
                with open("output_video/search_complete_time.pkl", "wb") as f:
                    pickle.dump(self.complete_time_list, f)
                exit()

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        segmentation_object_success_threshold = [20, 20, 15, 20, 20, 30, 30, 20]
        # if self.total_steps > 0:
        #     self.record_8_type = [0, 0, 0, 0, 0, 0, 0, 0]
        #     for i in range(self.num_envs):
        #         object_idx = i % 8
        #         if self.segmentation_object_point_num[i] > segmentation_object_success_threshold[object_idx]:
        #             self.record_8_type[object_idx] += 1 

        #     for i in range(8):
        #         self.record_8_type[i] /= (self.num_envs / 8)
        #     print("insert_success_rate_index: ", self.record_8_type)
        #     print("insert_success_rate: ", sum(self.record_8_type) / 8)

        # save the terminal state
        if self.total_steps < -1:
            axis1 = quat_apply(self.segmentation_target_rot, self.z_unit_tensor)
            axis2 = self.z_unit_tensor
            dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
            lego_z_align_reward = (torch.sign(dot1) * dot1 ** 2)

            self.saved_searching_ternimal_state = self.root_state_tensor.clone()[self.lego_indices.view(-1), :].view(self.num_envs, 132, 13)
            self.saved_searching_hand_ternimal_state = self.dof_state.clone().view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
            for i in range(self.num_envs):
                object_i = i % 8
                if self.segmentation_object_point_num[i] > segmentation_object_success_threshold[object_i]:
                    if lego_z_align_reward[i] < 10.6:
                        if self.save_hdf5:
                            if self.use_temporal_tvalue:
                                self.succ_grp.create_dataset("{}th_success_data".format(self.success_v_count), data=self.t_value_obs_buf[i].cpu().numpy())
                                
                            self.success_v_count += 1

                        self.saved_searching_ternimal_states_list[object_i][self.saved_searching_ternimal_states_index_list[object_i]:self.saved_searching_ternimal_states_index_list[object_i] + 1] = self.saved_searching_ternimal_state[i]
                        self.saved_searching_hand_ternimal_states_list[object_i][self.saved_searching_ternimal_states_index_list[object_i]:self.saved_searching_ternimal_states_index_list[object_i] + 1] = self.saved_searching_hand_ternimal_state[i]

                        self.saved_searching_ternimal_states_index_list[object_i] += 1
                        if self.saved_searching_ternimal_states_index_list[object_i] > 10000:
                            self.saved_searching_ternimal_states_index_list[object_i] = 0

                    else:
                        if self.save_hdf5:
                            if self.use_temporal_tvalue:
                                self.fail_grp.create_dataset("{}th_failure_data".format(self.failure_v_count), data=self.t_value_obs_buf[i].cpu().numpy())
                            else:
                                self.fail_grp.create_dataset("{}th_failure_data".format(self.failure_v_count), data=self.camera_view_segmentation_target_rot[i].cpu().numpy())
                            self.failure_v_count += 1
                else:
                    if self.save_hdf5:
                        if self.use_temporal_tvalue:
                            self.fail_grp.create_dataset("{}th_failure_data".format(self.failure_v_count), data=self.t_value_obs_buf[i].cpu().numpy())
                        else:
                            self.fail_grp.create_dataset("{}th_failure_data".format(self.failure_v_count), data=self.camera_view_segmentation_target_rot[i].cpu().numpy())
                        self.failure_v_count += 1

            for j in range(8):
                print("saved_searching_ternimal_states_index_{}: ".format(j), self.saved_searching_ternimal_states_index_list[j])

            if all([i > 5000 for i in self.saved_searching_ternimal_states_index_list]):
                with open("intermediate_state/saved_searching_ternimal_states_medium_mo_tvalue.pkl", "wb") as f:
                    pickle.dump(self.saved_searching_ternimal_states_list, f)
                with open("intermediate_state/saved_searching_hand_ternimal_states_medium_mo_tvalue.pkl", "wb") as f:
                    pickle.dump(self.saved_searching_hand_ternimal_states_list, f)

                print("RECORD SUCCESS!")
                exit()

        # Randomize the start state
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_hand_dofs * 2 + 5), device=self.device)
        lego_init_rand_floats = torch_rand_float(-1.0, 1.0, (self.num_envs * self.num_lego_block, 3), device=self.device)
        lego_init_rand_floats.view(self.num_envs, self.num_lego_block, 3)[:, 72:, :] = 0
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 0:7] = self.lego_start_states[env_ids].view(-1, 13)[:, 0:7].clone()
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 7:13] = torch.zeros_like(self.root_state_tensor[self.lego_indices[env_ids].view(-1), 7:13])
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 0:1] = self.lego_start_states[env_ids].view(-1, 13)[:, 0:1].clone() + lego_init_rand_floats[:, 0].unsqueeze(-1) * 0.02
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 1:2] = self.lego_start_states[env_ids].view(-1, 13)[:, 1:2].clone() + lego_init_rand_floats[:, 1].unsqueeze(-1) * 0.02

        # reset segmentation object start pos
        reset_lego_index = self.seg_indices[env_ids].view(-1)
        self.root_state_tensor[reset_lego_index, 0] = 0.25 + rand_floats[env_ids, 0] * 0.05
        self.root_state_tensor[reset_lego_index, 1] = 0.19 + rand_floats[env_ids, 0] * 0.05
        self.root_state_tensor[reset_lego_index, 2] = 0.9
        object_indices = torch.unique(self.lego_indices[env_ids].view(-1)).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),gymtorch.unwrap_tensor(object_indices), len(object_indices))
        
        # reset arm hand
        self.arm_hand_dof_pos[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_default_dof_pos
        self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel
        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_default_dof_pos
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_default_dof_pos
        hand_indices = self.robot_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # reset the potential function
        self.E_prev[env_ids] = to_torch([0.0], device=self.device).repeat(len(env_ids))
        
        self.post_reset(env_ids, hand_indices)

        if 0 in env_ids and self.total_steps > 0:
            self.ema_success.append(self.extras['success'])
            reward_items = ['dist_reward', 'pose_reward', 'lego_up_reward', 'action_penalty', 'angle_reward', 'emergency_reward']
            for item in reward_items:
                self.extras[item] = self.extras[item].to('cpu').numpy()
            total_reward = sum([abs(self.extras[item]) for item in reward_items])
            print("\n")
            print("Current lego blocks: ", self.num_lego_suit*8)
            print("#" * 17, " Statistics", "#" * 17)
            print(f"dist_reward:      {self.extras['dist_reward']:.2f} ({(abs(self.extras['dist_reward']) / total_reward * 100):.2f}%)")
            print(f"angle_reward:     {self.extras['angle_reward']:.2f} ({(abs(self.extras['angle_reward']) / total_reward * 100):.2f}%)")
            print(f"pose_reward:      {self.extras['pose_reward']:.2f} ({(abs(self.extras['pose_reward']) / total_reward * 100):.2f}%)")
            print(f"emergency_reward: {self.extras['emergency_reward']:.2f} ({(abs(self.extras['emergency_reward']) / total_reward * 100):.2f}%)")
            print(f"lego_up_reward:   {self.extras['lego_up_reward']:.2f} ({(abs(self.extras['lego_up_reward']) / total_reward * 100):.2f}%)")
            print(f"action_penalty:   {self.extras['action_penalty']:.2f} ({(abs(self.extras['action_penalty']) / total_reward * 100):.2f}%)")
            print(f"EMA Success rate: {np.mean(self.ema_success):.2f}%")
            print(f"Success length:   {self.extras['success_length']}")
            print("#" * 15, "Statistics End", "#" * 15,"\n")
            
            # upload wandb
            if self.enable_wandb:
                wandb.log({'reward/dist_reward': self.extras['dist_reward'],
                        "reward/pose_reward": self.extras['pose_reward'],
                        'reward/lego_up_reward': self.extras['lego_up_reward'], 
                        'reward/action_penalty': self.extras['action_penalty'],
                        'reward/angle_reward': self.extras['angle_reward'],
                        'reward/emergency_reward': self.extras['emergency_reward'],
                        'dist/z_lift': self.extras['z_lift'],
                        'dist/xy_move': self.extras['xy_move'],
                        'stats/total_reward': sum([_ for _ in self.extras.values()]),
                        'stats/success_rate': np.mean(self.ema_success),
                        'stats/success_length': self.extras['success_length'],
                        }, step=self.total_steps, commit=False)
            
            self.extras = {'dist_reward': 0, 'action_penalty': 0, 'lego_up_reward': 0, "pose_reward": 0, 'angle_reward': 0, 'emergency_reward': 0, 'success_length': self.max_episode_length}
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0 
        self.episodes += 1
              
    def post_reset(self, env_ids, hand_indices):
        self.gym.clear_lines(self.viewer)
        # step physics and render each frame
        for _ in range(60):
            self.render()
            self.gym.simulate(self.sim)
        
        self.gym.fetch_results(self.sim, True)
        if self.enable_camera_sensors:
            self.render_for_camera()
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            camera_rgba_image = camera_rgb_visulization(self.camera_tensors, env_id=0, is_depth_image=False)
            camera_seg_image = camera_segmentation_visulization(self.segmentation_id_list, self.camera_tensors, self.camera_seg_tensors, env_id=0, is_depth_image=False)
            self.compute_emergence_reward(self.camera_tensors, self.camera_seg_tensors, segmentation_id_list=self.segmentation_id_list)
            cv2.namedWindow("DEBUG_RGB_VIS", 0)
            cv2.namedWindow("DEBUG_SEG_VIS", 0)
            cv2.imshow("DEBUG_RGB_VIS", camera_rgba_image)
            cv2.imshow("DEBUG_SEG_VIS", camera_seg_image)
            cv2.waitKey(1)
            self.gym.end_access_image_tensors(self.sim)

        self.arm_hand_dof_pos[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_pos
        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_pos
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_pos
        self.gym.set_dof_position_target_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        
        # reset target block init pose
        self.seg_start_pos[env_ids] = self.root_state_tensor[self.seg_indices[env_ids], 0:3].clone()
        self.seg_start_rot[env_ids] = self.root_state_tensor[self.seg_indices[env_ids], 3:7].clone()
        
        print("Post Reset finish!!")
               
    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_envs(env_ids)
        
        self.actions = actions.clone().to(self.device)
        # ============ control inspire hand ============ 
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions[:, self.actuated_dof_indices],
            self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
            self.arm_hand_dof_upper_limits[self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
            self.cur_targets[:, self.actuated_dof_indices],
            self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
            self.arm_hand_dof_upper_limits[self.actuated_dof_indices]
        )
        # ============ control realman arm ============
        RL_CONTROL = True
        if RL_CONTROL:
            targets = self.arm_hand_dof_pos[:, :self.num_realman_dofs] + self.arm_hand_dof_speed_scale  * self.dt * self.actions[:, :self.num_realman_dofs]
            self.cur_targets[:, :self.num_realman_dofs] = tensor_clamp(
                targets, 
                self.arm_hand_dof_lower_limits[:self.num_realman_dofs], 
                self.arm_hand_dof_upper_limits[:self.num_realman_dofs]
            )
        else:
            pos_err = self.seg_pos - self.rigid_body_states[:, self.hand_base_rigid_body_index, :3]
            pos_err[:, 2] += 0.07
            pos_err[:, 0] -= 0.16
            
            target_rot = to_torch([0.5, -0.5,  0.5, -0.5], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
            rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone())
            dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, dpose, self.num_envs)
            self.cur_targets[:, :self.num_realman_dofs] = self.arm_hand_dof_pos[:, :self.num_realman_dofs] + delta[:, :self.num_realman_dofs]
        
        # apply the targets to the simulation
        self.cur_targets[:, :] = tensor_clamp(
            self.cur_targets[:, :],
            self.arm_hand_dof_lower_limits[:],
            self.arm_hand_dof_upper_limits[:]
        )
        
        self.prev_targets[:, :] = self.cur_targets[:, :]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
    
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        # update observation and reward
        self.compute_observation()
        self.compute_reward()
        self.total_steps += 1
        
        # info
        if self.print_success_stat:
            print("Total steps = {}".format(self.total_steps))
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
        
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.add_debug_lines(self.envs[0], middle_pos[0], self.finger_thumb_rot[0])
        self.add_debug_lines(self.envs[0], self.seg_pos[0], self.seg_rot[0])
        draw_point(self.gym, self.viewer, self.envs[0], self.seg_pos[0], self.seg_rot[0], color=(1, 0, 0), radius=0.03)
        # draw_point(self.gym, self.viewer, self.envs[0], self.seg_pos[0], self.lego_rot[0], color=(1, 0, 0), radius=0.015)
        # draw_point(self.gym, self.viewer, self.envs[0], self.middle_point[0], self.finger_thumb_rot[0], color=(1, 0, 0), radius=0.016)
        # draw_point(self.gym, self.viewer, self.envs[0], ik_point[0], self.hand_base_rot[0], color=(0, 0, 1), radius=0.02)
         
    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

# pre_physics_step
# reset_envs
# post_reset
# post_physics_step
# compute_observation
# compute_reward
###########################################################
##############  inner function tools ####################
###########################################################
def create_lego_asset(sim, gym, asset_root, lego_path, lego_file_name, disable_gravity=False, fix_base_link=False, thickness=0.00001, dof_mode=gymapi.DOF_MODE_NONE):
    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = disable_gravity
    asset_options.fix_base_link = fix_base_link
    asset_options.thickness = thickness
    asset_options.default_dof_drive_mode = dof_mode
    return gym.load_asset(sim, asset_root, lego_path + lego_file_name, asset_options)

def create_lego_pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(x, y, z)
    pose.r = gymapi.Quat().from_euler_zyx(roll, pitch, yaw)
    return pose

###########################################################
##############  Helpful function tools ####################
###########################################################
def control_ik(j_eef, device, dpose, num_envs):
	# Set controller parameters
	# IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u

def orientation_error(desired, current):
	cc = quat_conjugate(current)
	q_r = quat_mul(desired, cc)
	return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.

    :param d: The dictionary to flatten.
    :param parent_key: The base key for the current level of recursion.
    :param sep: The separator between parent and child keys.
    :return: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

def draw_point(gym, viewer, env, center, rotation, ax="xyz", radius=0.02, num_segments=32, color=(1, 0, 0),):
    rotation = rotation.cpu().numpy()
    center = center.cpu().numpy()
    rot_matrix = R.from_quat(rotation).as_matrix()

    for ax in list(ax):
        if ax.lower() == "x":
            plane_axes = [1, 2]  # yz平面
        elif ax.lower() == "y":
            plane_axes = [0, 2]  # xz平面
        else:
            plane_axes = [0, 1]  # xy平面

        points = []
        for i in range(num_segments + 1):
            angle = 2 * math.pi * i / num_segments
            # 在选定的平面上计算点
            local_point = np.zeros(3)
            local_point[plane_axes[0]] = radius * math.cos(angle)
            local_point[plane_axes[1]] = radius * math.sin(angle)

            # 将局部坐标转换为全局坐标
            global_point = center + rot_matrix @ local_point
            points.append(global_point)

        for i in range(num_segments):
            start = points[i]
            end = points[i + 1]
            gym.add_lines(viewer, env, 1, [*start, *end], color)

def camera_rgb_visulization(camera_tensors, env_id=0, is_depth_image=False):
        torch_rgba_tensor = camera_tensors[env_id].clone()
        camera_image = torch_rgba_tensor.cpu().numpy()
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
        
        return camera_image

def camera_segmentation_visulization(segmentation_id_list, camera_tensors, camera_seg_tensors, segmentation_id=0, env_id=0, is_depth_image=False):
    torch_rgba_tensor = camera_tensors[env_id].clone()
    torch_seg_tensor = camera_seg_tensors[env_id].clone()
    torch_rgba_tensor[torch_seg_tensor != segmentation_id_list[env_id]] = 0

    camera_image = torch_rgba_tensor.cpu().numpy()
    camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

    return camera_image

_DEFAULT_VALUE_AT_MARGIN = 0.1

def _sigmoids(x, value_at_1, sigmoid):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
        x: A scalar or PyTorch tensor of shape (batch_size, 1).
        value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
        sigmoid: String, choice of sigmoid type.

    Returns:
        A PyTorch tensor with values between 0.0 and 1.0.

    Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
          `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                             'got {}.'.format(value_at_1))
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                             'got {}.'.format(value_at_1))

    if sigmoid == 'gaussian':
        scale = torch.sqrt(-2 * torch.log(torch.tensor(value_at_1)))
        return torch.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == 'hyperbolic':
        scale = torch.acosh(1 / torch.tensor(value_at_1))
        return 1 / torch.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = torch.sqrt(1 / torch.tensor(value_at_1) - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / torch.tensor(value_at_1) - 1
        return 1 / (torch.abs(x) * scale + 1)

    elif sigmoid == 'cosine':
        scale = torch.acos(2 * torch.tensor(value_at_1) - 1) / torch.pi
        scaled_x = x * scale
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore', message='invalid value encountered in cos')
            cos_pi_scaled_x = torch.cos(torch.pi * scaled_x)
        return torch.where(torch.abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, torch.tensor(0.0))

    elif sigmoid == 'linear':
        scale = 1 - torch.tensor(value_at_1)
        scaled_x = x * scale
        return torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x, torch.tensor(0.0))

    elif sigmoid == 'quadratic':
        scale = torch.sqrt(1 - torch.tensor(value_at_1))
        scaled_x = x * scale
        return torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x ** 2, torch.tensor(0.0))

    elif sigmoid == 'tanh_squared':
        scale = torch.atanh(torch.sqrt(1 - torch.tensor(value_at_1)))
        return 1 - torch.tanh(x * scale) ** 2

    else:
        raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))

def tolerance(x, y, r, margin=0.0, sigmoid='gaussian', value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    """Returns 1 when `x` falls inside the circle centered at `p` with radius `r`, between 0 and 1 otherwise.

    Args:
        x: A batch_size x 3 numpy array representing the points to check.
        y: A length-3 numpy array representing the center of the circle.
        r: Float. The radius of the circle.
        margin: Float. Parameter that controls how steeply the output decreases as `x` moves out-of-bounds.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian', 'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when the distance from `x` to the nearest bound is equal to `margin`. Ignored if `margin == 0`.

    Returns:
        A numpy array with values between 0.0 and 1.0 for each point in the batch.

    Raises:
        ValueError: If `margin` is negative.
    """
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    # Calculate the Euclidean distance from each point in x to p
    distance = torch.norm(x - y, p=2, dim=-1)

    in_bounds = distance <= r
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        d = (distance - r) / margin
        
        value = torch.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return value

def quaternion_multiply(q1, q2):
    """Multiply two batches of quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack((w, x, y, z), dim=-1)

def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion."""
    q_conj = q.clone()
    q_conj[:, 1:] = -q_conj[:, 1:]
    return q_conj

def compute_relative_rotation(q1, q2):
    """Compute the relative rotation quaternion from q1 to q2 for a batch of quaternions."""
    q1_conj = quaternion_conjugate(q1)
    relative_rotation = quaternion_multiply(q1_conj, q2)
    return relative_rotation

def quaternion_angle(q):
    """Compute the angle of a batch of quaternions."""
    q = q / q.norm(dim=-1, keepdim=True)  # Ensure the quaternions are normalized
    w = q[..., 0].clamp(-1.0, 1.0)
    angle = 2 * torch.acos(w)
    return angle

def fingers_alignment_err(q1, q2, tolerance_angle=torch.pi/12):
    """Compute the alignment error for a batch of quaternions."""
    # Compute the relative rotation
    relative_rotation = compute_relative_rotation(q1, q2)
    
    # Compute the angle difference
    angle_diff = quaternion_angle(relative_rotation)
    
    # Calculate the alignment error
    # The optimal alignment is when the angle difference is π (180 degrees)
    alignment_err = torch.abs(angle_diff - torch.pi)
    
    # Normalizing the alignment error based on the tolerance angle
    alignment_err = torch.clamp(alignment_err / tolerance_angle, max=1.0)
    
    return alignment_err.unsqueeze(-1)

def compute_angle_line_plane(p1, p2, plane_normal):
    # Compute the direction vector of the line
    line_direction = p2 - p1  # (batch, 3)
    
    # Normalize the line direction and the plane normal
    line_direction_normalized = line_direction / torch.norm(line_direction, dim=-1, keepdim=True)  # (batch, 3)
    plane_normal_normalized = plane_normal / torch.norm(plane_normal, dim=-1, keepdim=True)  # (batch, 3)
    
    # Compute the dot product between the line direction and the plane normal
    dot_product = torch.bmm(line_direction_normalized.unsqueeze(1), plane_normal_normalized.unsqueeze(2)).squeeze()  # (batch)
    
    # Clamp the dot product to avoid numerical issues with acos
    dot_product_clamped = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute the angle between the line direction and the plane normal
    angle_with_normal = torch.acos(dot_product_clamped)  # (batch)
    
    # Compute the angle between the line and the plane
    angle_line_plane = torch.pi / 2 - angle_with_normal  # (batch)
    
    return angle_line_plane