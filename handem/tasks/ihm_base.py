import os

# import pickle

import numpy as np
import torch
import random
from isaacgym import gymapi, gymtorch

# pylint: disable = wildcard-import, unused-wildcard-import
from isaacgym.torch_utils import *

from termcolor import cprint

from handem.tasks.base.vec_task import VecTask

# from datetime import datetime

# from isaacgymenvs.tasks.base.vec_task import VecTask
from handem.utils.torch_jit_utils import tensor_clamp, my_quat_rotate
from handem.utils.torch_utils import to_torch
from scipy import interpolate
import pickle
from datetime import datetime

class IHMBase(VecTask):
    """This is a base class for all in-hand manipulation tasks such as sampling grasps and learning
    finger-gaiting.

    The net contact force which will have as a part of the state, obs is in the global frame.
    The net contact force tensor contains the net contact forces experienced by each rigid body
    during the last simulation step, with the forces expressed as 3D vectors. It is a read-only
    tensor with shape (num_rigid_bodies, 3). You can index individual rigid bodies in the same way
    as the rigid body state tensor. Directly, using these numbers in the observation is fine.
    However, when we want to add simulated noise we should be a little bit more careful.

    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self._setup_states_obs_actions_dims()
        self.up_axis = "z"
        # load parameters
        self._setup_object_params()
        self._setup_hand_params()
        self._setup_ur5e_params()

        super().__init__(
            config=self.cfg,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
        )
        self.debug_viz = self.cfg["env"].get("enableDebugVis", False)
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        # viewer
        if self.viewer is not None:
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
            cam_target = gymapi.Vec3(0.125, 0.5, 0.25)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        self.vis_obj_axes = self.cfg["env"]["visualization"].get("visObjAxes", False)
        self.vis_contact_vec = self.cfg["env"]["visualization"].get("visContactVec", False)

        # logging
        self.enable_log = self.cfg["env"]["logging"]["enableLog"]
        if self.enable_log:
            self._setup_logging_config()
        
        self._setup_default_state()

        self.saved_grasp_states = None
        self.states_buf = None
        self.out_of_bounds = torch.zeros_like(self.reset_buf, device=self.device)
        self.contactBoolForceThreshold = cfg["env"]["contactBoolForceThreshold"]
        self.action_scale = to_torch(cfg["env"]["actionScale"], device=self.device)

        self.create_tensor_views()
        self.gym.simulate(self.sim)
        self._refresh_tensors()

    def _setup_states_obs_actions_dims(self):
        dims = {
            "hand_joint_pos": 15,
            "hand_joint_vel": 15,
            "hand_joint_target": 15,
            "ftip_contact_force": 15,
            "object_pos": 3,
            "object_orientation": 4,
            "object_lin_vel": 3,
            "object_ang_vel": 3,
            "hand_joint_torque": 15,
            "ftip_contact_bool": 5,
            "ftip_contact_pos": 15,
        }
        self.cfg["env"]["numStates"] = sum([dims[key] for key in self.cfg["env"]["feedbackState"]])
        self.cfg["env"]["numObservations"] = sum([dims[key] for key in self.cfg["env"]["feedbackObs"]])
        self.cfg["env"]["numActions"] = 15

    def _setup_object_params(self):
        self.object_mass = self.cfg["env"]["object_params"]["mass"]
        self.object_com = list(self.cfg["env"]["object_params"]["com"])
        self.object_friction = self.cfg["env"]["object_params"]["friction"]
        self.object_scale = self.cfg["env"]["object_params"]["scale"]

    def _setup_hand_params(self):
        # actuator parameters
        self.hand_stiffness = self.cfg["env"]["hand_params"]["stiffness"]
        self.hand_damping = self.cfg["env"]["hand_params"]["damping"]
        self.hand_velocity_limit = self.cfg["env"]["hand_params"]["velocityLimit"]
        self.hand_effort_limit = self.cfg["env"]["hand_params"]["effortLimit"]
        self.hand_joint_friction = self.cfg["env"]["hand_params"]["jointFriction"]

    def _setup_ur5e_params(self):
        # actuator parameters
        self.ur5e_stiffness = self.cfg["env"]["ur5e_params"]["stiffness"]
        self.ur5e_damping = self.cfg["env"]["ur5e_params"]["damping"]
        self.ur5e_velocity_limit = self.cfg["env"]["ur5e_params"]["velocityLimit"]
        self.ur5e_effort_limit = self.cfg["env"]["ur5e_params"]["effortLimit"]

    def _setup_default_state(self):
        self.default_hand_joint_pos = to_torch(self.cfg["env"]["default_hand_joint_pos"], device=self.device)
        self.default_object_pos = to_torch(self.cfg["env"]["default_object_pos"], device=self.device)
        self.default_object_vel = to_torch([0] * 6, device=self.device)
        self.default_ur5e_joint_pos = to_torch(self.cfg["env"]["default_ur5e_joint_pos"], device=self.device).unsqueeze(0)
        self.default_ur5e_joint_vel = 0 * to_torch(self.cfg["env"]["default_ur5e_joint_pos"], device=self.device).unsqueeze(0)
        init_obj_euler = self.cfg["env"]["default_object_euler"]
        init_obj_quat = quat_from_euler_xyz(
            torch.tensor(init_obj_euler[0]),
            torch.tensor(init_obj_euler[1]),
            torch.tensor(init_obj_euler[2]),
        )
        self.default_object_quat = to_torch(init_obj_quat, device=self.device)

    def _setup_logging_config(self):
        self.log_dir = self.cfg["env"]["logging"]["logDir"]
        self.max_log_len = self.cfg["env"]["logging"]["maxLogLen"]
        self.log_count = 0
        self.log = {}
        self.log_obs = self.cfg["env"]["logging"]["logObs"]
        self.log_pose = self.cfg["env"]["logging"]["logPose"]
        self.log_contact = self.cfg["env"]["logging"]["logContact"]

    def create_sim(self):
        self.up_axis_idx = 2 if self.up_axis == "z" else 1  # index of up axis: Y=1, Z=2
        super().create_sim()
        self._create_ground_plane()

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        robot = "ur5e_hand"

        asset_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"))
        robot_asset_file = os.path.join(f"{robot}", "urdf", f"{robot}.urdf")

        # robot asset options
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        # asset_options.thickness = 0.001
        asset_options.angular_damping = 0.1
        asset_options.vhacd_enabled = False
        asset_options.vhacd_params.resolution = 1000000
        # asset_options.vhacd_params.max_num_vertices_per_ch = 256
        # Resolution of fingertip collision gemetry is not sufficient and must to be dealt with!

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = False

        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_actuators = self.gym.get_asset_actuator_count(robot_asset)
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)


        # dof limits
        self.ur5e_dof_lower_limits = to_torch(
            list(self.cfg["env"]["ur5e_params"]["ur5e_joint_lower_lim"]), 
            device=self.device
        )
        self.ur5e_dof_upper_limits = to_torch(
            list(self.cfg["env"]["ur5e_params"]["ur5e_joint_upper_lim"]), 
            device=self.device
        )
        self.hand_dof_lower_limits = to_torch(
            list(self.cfg["env"]["hand_params"]["hand_joint_lower_lim"]) * 5, 
            device=self.device
        )
        self.hand_dof_upper_limits = to_torch(
            list(self.cfg["env"]["hand_params"]["hand_joint_upper_lim"]) * 5, 
            device=self.device
        )

        self.robot_dof_lower_limits = to_torch(
            self.ur5e_dof_lower_limits.tolist() + self.hand_dof_lower_limits.tolist(), 
            device=self.device
        )
        self.robot_dof_upper_limits = to_torch(
            self.ur5e_dof_upper_limits.tolist() + self.hand_dof_upper_limits.tolist(), 
            device=self.device
        )

        self.obj_xyz_lower_lim = to_torch(list(self.cfg["env"]["reset"]["obj_xyz_lower_lim"]), device=self.device)
        self.obj_xyz_upper_lim = to_torch(list(self.cfg["env"]["reset"]["obj_xyz_upper_lim"]), device=self.device)

        # hand dof
        ur5e_dofs = range(6)
        hand_dofs = range(6, 21)

        for i in ur5e_dofs:
            robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            robot_dof_props["stiffness"][i] = self.ur5e_stiffness
            robot_dof_props["damping"][i] = self.ur5e_damping
            robot_dof_props["effort"][i] = self.ur5e_effort_limit
            robot_dof_props["velocity"][i] = self.ur5e_velocity_limit
            robot_dof_props["armature"][i] = 0.01
            robot_dof_props["lower"][i] = self.robot_dof_lower_limits[i]
            robot_dof_props["upper"][i] = self.robot_dof_upper_limits[i]

        for i in hand_dofs:
            robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            robot_dof_props["stiffness"][i] = self.hand_stiffness
            robot_dof_props["damping"][i] = self.hand_damping
            robot_dof_props["effort"][i] = self.hand_effort_limit
            robot_dof_props["velocity"][i] = self.hand_velocity_limit
            robot_dof_props["armature"][i] = 0.01
            robot_dof_props["friction"][i] = self.hand_joint_friction
            robot_dof_props["lower"][i] = self.robot_dof_lower_limits[i]
            robot_dof_props["upper"][i] = self.robot_dof_upper_limits[i]

        # Create table asset
        table_pos = [0.125, 0.5, 0.15]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[0.5, 0.5, table_thickness], table_opts)

        # Create table stand asset
        table_stand_pos = [table_pos[0], table_pos[1], table_pos[2]/2 - table_thickness / 4]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.1, 0.1, table_pos[2] - table_thickness / 2], table_opts)

        # start pose for robot
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for object
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(table_pos[0], table_pos[1], table_pos[2] + 0.1)
        object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # object asset options
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.angular_damping = 0.01
        asset_options.vhacd_enabled = self.cfg["env"].get("enableObjConvexDecomp", False)
        asset_options.vhacd_params.resolution = 400000

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = False
        self.object = self.cfg["env"]["object"]
        if isinstance(self.object, str):
            self.object_list = [self.object]
        else:
            self.object_list = list(self.object)

        object_asset_list = []
        for obj in self.object_list:
            object_asset_file = os.path.join("objects", f"{obj}.urdf")
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)
            object_asset_list.append(object_asset)

        self.actor_handles = {"robot": [], "object": [], "table": [], "table_stand": []}

        self.envs = []
        self.object_indices = []

        # store object name for each environment (for obj agnostic training)
        self.object_name_array = []

        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            # create robot actor
            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            self.actor_handles["robot"].append(robot_actor)

            # Enable DOF force sensors as they are disabled by default
            self.gym.enable_actor_dof_force_sensors(env_ptr, robot_actor)

            # Create table and table stand
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 0, 0)
            self.actor_handles["table"].append(table_actor)
            self.actor_handles["table_stand"].append(table_stand_actor)
            
            idx = random.choice(range(len(self.object_list)))
            self.object_name_array.append(self.object_list[idx])
            object_asset = object_asset_list[idx]
            
            object_actor = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.gym.set_rigid_body_color(
                env_ptr,
                object_actor,
                0,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(0.2, 0.6, 1.0),
            )
            self.actor_handles["object"].append(object_actor)
            object_idx = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            
            ############ object scale ############
            obj_scale = self.object_scale
            self.gym.set_actor_scale(env_ptr, object_actor, obj_scale)
            ############ object scale ############
    
        
            #### object rigid body properties ####
            prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_actor)
            obj_mass = self.object_mass
            for p in prop:
                p.mass = obj_mass

            obj_com = self.object_com
            prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
            self.gym.set_actor_rigid_body_properties(env_ptr, object_actor, prop)
            #### object rigid body properties ####


            #### object rigid shape properties ####
            robot_props = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_actor)
            object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_actor)

            friction = self.object_friction             
            for p in robot_props:
                p.friction = friction
            for p in object_props:
                p.friction = friction
            
            self.gym.set_actor_rigid_shape_properties(env_ptr, robot_actor, robot_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_actor, object_props)
            #### object rigid shape properties ####

        # Rigid body handles used later to compute object pose and contact forces
        self.rigid_body_handles = {}
        self.rigid_body_handles["object"] = self.gym.find_actor_rigid_body_handle(env_ptr, self.actor_handles["object"][0], "object")

        for i in range(5):
            ftip = f"finger{i + 1}_distal"
            self.rigid_body_handles[ftip] = self.gym.find_actor_rigid_body_handle(env_ptr, self.actor_handles["robot"][0], ftip)

        # Get rigid body indices for all fingertips
        for i in range(5):
            ftip = f"finger{i + 1}_distal_tip"
            self.rigid_body_handles[ftip] = self.gym.find_actor_rigid_body_handle(env_ptr, self.actor_handles["robot"][0], ftip)

        num_actors_per_env = 4 # robot, table, stand, object            
        self.actor_idx = torch.arange(self.num_envs * num_actors_per_env, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.robot_idx = self.actor_idx[:, 0]
        self.table_idx = self.actor_idx[:, 1]
        self.table_stand_idx = self.actor_idx[:, 2]
        self.object_idx = self.actor_idx[:, 3]
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_tensor_views(self):
        """Create tensor views to useful data in simulation. These tensors contain garbage until
        the first call to step() and _refresh_tensors().
        """

        # Create tensor views of simulation data
        root_state_tensor_ = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(root_state_tensor_).view(-1, 13)
        dof_state_ = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_).view(self.num_envs, -1, 2)

        rigid_body_state_ = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_).view(self.num_envs, -1, 13)

        jacobian_ = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        self.jacobian = gymtorch.wrap_tensor(jacobian_)

        # DOF Forces
        forces_ = self.gym.acquire_dof_force_tensor(self.sim)
        
        torques = gymtorch.wrap_tensor(forces_).view(self.num_envs, -1)

        self.ur5e_torques = torques[:, :6]
        self.hand_torques = torques[:, 6:]

        # Rigid Body Force Sensors
        # Force sensors can be attached to rigid bodies to measure forces and torques experienced
        # at user-specified reference frames.
        # _fsdata = self. gym.acquire_force_sensor_tensor(self.sim)
        # self.fsdata = gymtorch.wrap_tensor(_fsdata)
        # Error: "*** Can't create empty tensor"

        contact_force_ = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_force = gymtorch.wrap_tensor(contact_force_).view(self.num_envs, -1, 3)

        # More views for convenience
        
        # ur5e dof state
        self.ur5e_dof_pos = self.dof_state[:, 0:6, 0]
        self.ur5e_dof_vel = self.dof_state[:, 0:6, 1]

        # hand dof state
        self.hand_dof_pos = self.dof_state[:, 6:, 0]
        self.hand_dof_vel = self.dof_state[:, 6:, 1]

        self.object_state = self.rigid_body_state[:, self.rigid_body_handles["object"], :]
        self.object_state_prev = self.object_state.detach().clone()
        self.object_pose = self.object_state[:, :7]
        self.object_pos = self.object_state[:, :3]
        self.object_orientation = self.object_state[:, 3:7]
        self.object_lin_vel = self.object_state[:, 7:10]
        self.object_ang_vel = self.object_state[:, 10:]
        self.ftip_contact_force = [self.contact_force[:, self.rigid_body_handles[f"finger{i + 1}_distal"], :] for i in range(5)]
        self.ftip_pos = [self.rigid_body_state[:, self.rigid_body_handles[f"finger{i + 1}_distal_tip"], :3] for i in range(5)]
        self.ftip_orientation = [self.rigid_body_state[:, self.rigid_body_handles[f"finger{i + 1}_distal_tip"], 3:7] for i in range(5)]

        # perturbation
        self.force_tensor = torch.zeros(
            self.rigid_body_state.shape[0], self.rigid_body_state.shape[1], 3, device=self.device
        )
        self.torque_tensor = torch.zeros(
            self.rigid_body_state.shape[0], self.rigid_body_state.shape[1], 3, device=self.device
        )

        # action buffers
        self.target_hand_joint_pos = self.hand_dof_pos.clone().detach()
        self.target_ur5e_joint_pos = self.default_ur5e_joint_pos.repeat(self.num_envs, 1)

    def _allocate_task_buffer(self, num_envs):
        self.prop_hist_len = self.cfg["env"]["adaptation"]["propHistoryLen"]
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, self.num_obs), device=self.device, dtype=torch.float)

    def _refresh_tensors(self):
        """Updates data in tensor views created in create_tensor_views(). To be used judicioulsy as
        it involves copying data.
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """
        actions = actions.to(self.device)
        self.actions = actions
        self.target_hand_joint_pos += actions * self.action_scale
        self.target_hand_joint_pos = tensor_clamp(
            self.target_hand_joint_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
        )

        target_robot_joint_pos = torch.cat((self.target_ur5e_joint_pos, self.target_hand_joint_pos), dim=1)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target_robot_joint_pos))

    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_tensors()
        self.compute_reward()
        self.check_reset()
        env_idx = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_idx) > 0:
            self.reset_idx(env_idx)
        self.compute_observation()
        self.object_state_prev = self.object_state.detach().clone()

    def compute_observation(self):

        self.ftip_contact_pos, contact_vectors = self.compute_ftip_contact_pos_vec()
        self.contact_bool_tensor = self.compute_contact_bool()
        self.ncon = torch.sum(self.contact_bool_tensor, dim=1)

        # flip object_pos sign if first element is negative (to keep consistent with quat generation)
        flip_indices = torch.argwhere(self.object_orientation[:, 0] < 0).squeeze(-1)
        object_orientation = self.object_orientation.clone()
        object_orientation[flip_indices, :] = -object_orientation[flip_indices, :]

        feeback = {
            "hand_joint_pos": self.hand_dof_pos,
            "hand_joint_vel": self.hand_dof_vel,
            "hand_joint_target": self.target_hand_joint_pos,
            "object_pos": self.object_pos,
            "object_orientation": object_orientation,
            "object_lin_vel": self.object_lin_vel,
            "object_ang_vel": self.object_ang_vel,
            "ftip_contact_force": torch.cat(self.ftip_contact_force, dim=1),
            "hand_joint_torque": self.hand_torques,
            "ftip_contact_bool": self.contact_bool_tensor,
            "ftip_contact_pos": torch.cat(self.ftip_contact_pos, dim=1),
        }

        if self.enable_log and self.headless == False and self.log_count < self.max_log_len:
            self.log_data(feeback)

        states = {key: feeback[key] for key in self.cfg["env"]["feedbackState"]}
        obs = {key: feeback[key] for key in self.cfg["env"]["feedbackObs"]}
        self.states_buf = torch.cat(list(states.values()), dim=-1)

        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()

        cur_obs_buf = torch.cat(list(obs.values()), dim=-1)
        cur_state_buf = torch.cat(list(states.values()), dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf.unsqueeze(1)], dim=1)

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids] = cur_obs_buf[at_reset_env_ids].unsqueeze(1)
        # pulls the last obs_hist_len observations from the history buffer
        t_buf = (self.obs_buf_lag_history[:, -1:].reshape(self.num_envs, -1)).clone()

        self.obs_buf[:, : t_buf.shape[1]] = t_buf
        self.at_reset_buf[at_reset_env_ids] = 0

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len :].clone()

    def log_data(self, feedback):
        self.log_count += 1
        # log data
        if self.log_obs:
            obs_dict = {key: feedback[key] for key in set(self.cfg["env"]["feedbackObs"]) - {"object_orientation"}}
            obs = torch.cat(list(obs_dict.values()), dim=-1)
            self.logval(
                "obs",
                obs[0, :].detach().cpu().numpy(),
            )
        if self.log_pose:
            self.logval(
                "obj_pose",
                self.object_pose[0, :].detach().cpu().numpy(),
            )
        if self.log_contact:
            ftip_contact_force = torch.cat(self.ftip_contact_force, dim=1)
            self.logval(
                "contact",
                ftip_contact_force[0, :].detach().cpu().numpy(),
            )
        # if logging period is over, dump data
        if self.log_count >= self.max_log_len:
            self.dump_log()

    def dump_log(self):
        for key, log in self.log.items():
            self.log[key] = np.vstack(log)
            logfile = self.log_dir + self.cfg["env"]["object"] + datetime.now().strftime("_%m_%d_%H_%M.pkl")
        cprint(f"Saving {logfile}")
        with open(logfile, "wb") as f:
            pickle.dump(self.log, f)

    def logval(self, key, val):
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(val)

    def visualize(self, contact_vectors):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        for i in range(self.num_envs):
            if self.vis_obj_axes:
                # render object axes
                objectx = (self.object_pos[i] + quat_apply(self.object_orientation[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_orientation[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_orientation[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])
            if self.vis_contact_vec:
                # render estimated contact normals
                for j in range(5):
                    if self.contact_bool_tensor[i, j]:
                        ftip_pos = self.ftip_pos[j][i].cpu().numpy()
                        viz_vector = contact_vectors[j][i].cpu().numpy()
                        self.gym.add_lines(self.viewer, self.envs[i], 1, [ftip_pos[0], ftip_pos[1], ftip_pos[2], viz_vector[0], viz_vector[1], viz_vector[2]], \
                                        [0, 0, 0])

    def reset(self):
        super().reset()
        self.obs_dict["proprio_hist"] = self.proprio_hist_buf.to(self.rl_device)
        if self.states_buf is not None:
            self.obs_dict["state"] = self.states_buf.clone()
        return self.obs_dict

    def step(self, actions):
        super().step(actions)
        self.obs_dict["proprio_hist"] = self.proprio_hist_buf.to(self.rl_device)
        if self.states_buf is not None:
            self.obs_dict["state"] = self.states_buf.clone()
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def compute_reward(self, actions=None):
        pass

    def check_obj_out_of_bounds(self):
        """Check if object is out of bounds"""
        obj_out_of_bounds = torch.zeros_like(self.reset_buf)
        obj_out_of_bounds = torch.where(self.object_state[:, 0] > self.obj_xyz_upper_lim[0], torch.ones_like(self.reset_buf), obj_out_of_bounds)
        obj_out_of_bounds = torch.where(self.object_state[:, 0] < self.obj_xyz_lower_lim[0], torch.ones_like(self.reset_buf), obj_out_of_bounds)
        obj_out_of_bounds = torch.where(self.object_state[:, 1] > self.obj_xyz_upper_lim[1], torch.ones_like(self.reset_buf), obj_out_of_bounds)
        obj_out_of_bounds = torch.where(self.object_state[:, 1] < self.obj_xyz_lower_lim[1], torch.ones_like(self.reset_buf), obj_out_of_bounds)
        obj_out_of_bounds = torch.where(self.object_state[:, 2] > self.obj_xyz_upper_lim[2], torch.ones_like(self.reset_buf), obj_out_of_bounds)
        obj_out_of_bounds = torch.where(self.object_state[:, 2] < self.obj_xyz_lower_lim[2], torch.ones_like(self.reset_buf), obj_out_of_bounds)
        return obj_out_of_bounds

    def check_reset(self):
        # if object falls out of the box
        self.out_of_bounds = self.check_obj_out_of_bounds()
        # store oob buf separately
        reset = self.out_of_bounds.clone()
        # if end of episode
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset)
        self.reset_buf[:] = reset

    def load_grasps(self):
        self.saved_grasp_states = {}
        cache_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../cache"))
        # loop through all objects
        for obj in self.object_list:
            self.saved_grasp_states[obj] = {}
            path = self.cfg["env"]["saved_grasps"][obj]
            grasp_file = os.path.join(cache_root, path)
            cprint(f"Loading grasps from {grasp_file}", "blue", attrs=["bold"])
            self.saved_grasp_states[obj] = torch.from_numpy(np.load(grasp_file)).to(self.device)

    def sample_grasp(self, env_idx, objects):
        """Sample new grasp pose"""
        if self.saved_grasp_states is None:
            return self.sample_default_grasp(env_idx)
        target_hand_joint_pos = torch.zeros((len(env_idx), self.hand_dof_pos.size(-1)), device=self.device)
        hand_joint_pos = torch.zeros((len(env_idx), self.hand_dof_pos.size(-1)), device=self.device)
        object_pos = torch.zeros((len(env_idx), 3), device=self.device)
        object_rot = torch.zeros((len(env_idx), 4), device=self.device)
        object_vel = torch.zeros((len(env_idx), 6), device=self.device)
        for obj in self.object_list:
            # which environments have been assigned to this object
            idx_o = [i for i, x in enumerate(objects) if x == obj]
            n = len(idx_so)
            # sample a grasp for each environment assigned to this scale
            if n == 0:
                continue
            grasp_idx = torch.randint(len(self.saved_grasp_states[obj]), (n,))
            hand_joint_pos[idx_so, :]= self.saved_grasp_states[obj][grasp_idx][:, :15]
            target_hand_joint_pos[idx_so, :] = self.saved_grasp_states[obj][grasp_idx][:, 15:30]
            object_pos[idx_so, :] = self.saved_grasp_states[obj][grasp_idx][:, 30:33]
            object_rot[idx_so, :] = self.saved_grasp_states[obj][grasp_idx][:, 33:37]
            object_vel[idx_so, :] = self.saved_grasp_states[obj][grasp_idx][:, 37:]
        return hand_joint_pos, target_hand_joint_pos, object_pos, object_rot, object_vel

    def sample_default_grasp(self, env_idx):
        hand_joint_pos = to_torch(self.default_hand_joint_pos, device=self.device).repeat(len(env_idx), 1)
        hand_joint_pos = tensor_clamp(hand_joint_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        object_pos = to_torch(self.default_object_pos, device=self.device).repeat(len(env_idx), 1)
        object_rot = to_torch(self.default_object_quat, device=self.device).repeat(len(env_idx), 1)
        object_vel = to_torch(self.default_object_vel, device=self.device).repeat(len(env_idx), 1)
        return hand_joint_pos, hand_joint_pos, object_pos, object_rot, object_vel

    def reset_idx(self, env_idx):
        objects = [self.object_name_array[idx] for idx in env_idx]

        hand_joint_pos, target_hand_joint_pos, object_pos, object_rot, object_vel = self.sample_grasp(env_idx, objects)
        self.hand_dof_pos[env_idx, :] = 0.0
        self.hand_dof_vel[env_idx, :] = 0.0
        self.ur5e_dof_pos[env_idx, :] = self.default_ur5e_joint_pos
        self.ur5e_dof_vel[env_idx, :] = 0.0
        self.target_hand_joint_pos[env_idx, :] = target_hand_joint_pos
        self.root_state_tensor[self.object_indices[env_idx], :3] = object_pos
        self.root_state_tensor[self.object_indices[env_idx], 3:7] = object_rot
        self.root_state_tensor[self.object_indices[env_idx], 7:13] = object_vel

        # position
        target_hand_joint_pos = self.target_hand_joint_pos.clone().detach()
        target_ur5e_joint_pos = self.target_ur5e_joint_pos.clone().detach()
        target_robot_joint_pos = torch.cat((target_ur5e_joint_pos, target_hand_joint_pos), dim=1)

        # velocity
        target_hand_joint_vel = 0 * self.target_hand_joint_pos.clone().detach()
        target_ur5e_joint_vel = 0 * self.target_ur5e_joint_pos.clone().detach()
        target_robot_joint_vel = torch.cat((target_ur5e_joint_vel, target_hand_joint_vel), dim=1)

        # Set the new states in the simulator
        # API: Indexed versions of the tensor API need global indices of the actors involved.
        # Next, we store these actor indices in a separate variable to so that the python garabage
        # collector will not release it and cause segmentation fault. See IsaacGym documentation
        # for more details.
        robot_idx = self.robot_idx[env_idx].flatten()
        dof_state = self.dof_state.clone().detach()

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(dof_state),
            gymtorch.unwrap_tensor(robot_idx),
            len(robot_idx),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(target_robot_joint_pos),
            gymtorch.unwrap_tensor(robot_idx),
            len(robot_idx),
        )
        self.gym.set_dof_velocity_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(target_robot_joint_vel),
            gymtorch.unwrap_tensor(robot_idx),
            len(robot_idx),
        )
        
        changed_root_indices = self.object_idx[env_idx].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(changed_root_indices),
            len(changed_root_indices),
        )
        
        self.progress_buf[env_idx] = 0
        self.obs_buf[env_idx] = 0

        self.object_state_prev[env_idx, :3] = object_pos
        self.object_state_prev[env_idx, 3:7] = object_rot
        self.object_state_prev[env_idx, 7:13] = 0.0

        self.proprio_hist_buf[env_idx] = 0
        self.at_reset_buf[env_idx] = 1

    def compute_contact_bool(self, distal_finger_force_threshold=0):
        # TODO: Wherever this is used, it should have two modes. One for the computing the accurate "state" and
        # another for computing the "obs"
        force_scalar = [torch.linalg.norm(force, dim=1) for force in self.ftip_contact_force]
        contact = [(force > self.contactBoolForceThreshold).long() for force in force_scalar]
        contact_bool_tensor = torch.transpose(torch.stack(contact), 0, 1)
        return contact_bool_tensor

    def compute_ftip_contact_pos_vec(self):
        """Compute approximate ftip contact position from ftip positions and net contact force"""
        ftip_radius = 0.0185
        # store contact positions
        contact_pos_cuda = []
        # also create exaggerated flipped contact vector for viz
        contact_viz_vecs = []
        for ftip_pos, contact_force in zip(self.ftip_pos, self.ftip_contact_force):
            approx_contact_force_normal = contact_force / torch.linalg.norm(contact_force, dim=1, keepdim=True)
            contact_pos_cuda.append(ftip_pos + approx_contact_force_normal * ftip_radius)
            contact_viz_vecs.append(ftip_pos + approx_contact_force_normal * 0.05)
        contact_pos_cuda = [torch.nan_to_num(contact_position) for contact_position in contact_pos_cuda]
        contact_viz_vecs = [torch.nan_to_num(contact_viz_vec) for contact_viz_vec in contact_viz_vecs]
        return contact_pos_cuda, contact_viz_vecs


    def random_actions(self) -> torch.Tensor:
        """Returns a buffer with random actions drawn from normal distribution

        Returns:
            torch.Tensor: A buffer of random actions torch actions
        """
        mean = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.device)
        std = torch.ones([self.num_envs, self.num_actions], dtype=torch.float32, device=self.device)
        actions = torch.normal(mean, std)

        return actions
