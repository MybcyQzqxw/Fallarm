# 观测空间 (13维):
#     [0:4]   arm_dof_pos - default  (shoulder_pitch/roll/yaw + elbow)
#     [4:8]   arm_dof_vel            (shoulder_pitch/roll/yaw + elbow)
#     [8:12]  arm_actions            (shoulder_pitch/roll/yaw + elbow)
#     [12:13] action_rescale         (动作缩放系数, 带小噪声)

# 动作空间 (4维, 仅手臂关节):
#     [0] shoulder_pitch 目标角偏移
#     [1] shoulder_roll  目标角偏移
#     [2] shoulder_yaw   目标角偏移
#     [3] elbow          目标角偏移

import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.fall_arm.fall_arm_config import FallArmCfg
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    tolerance,
    torch_rand_float,
)


class FallArm(BaseTask):

    # ========== 新增：肩根最小高度奖励相关状态初始化 ==========
    def _init_min_shoulder_root_height_reward(self):
        self.min_shoulder_root_height_reward_unlocked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_shoulder_root_height = torch.zeros(self.num_envs, device=self.device)

    def __init__(self, cfg: FallArmCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.debug_viz = True
        self.init_done = False  # 初始化flag
        self._parse_cfg(self.cfg)
        self.num_dofs = cfg.env.num_dofs
        self.num_real_dofs = cfg.env.num_real_dofs

        # 单步观测维度
        self.num_one_step_obs = self.cfg.env.num_one_step_observations  # if not self.cfg.env.add_force else self.cfg.env.num_one_step_observations + 1
        # 历史观测长度
        self.actor_history_length = self.cfg.env.num_actor_history
        # 总观测数 = 单步观测维度 * 历史观测长度
        self.actor_proprioceptive_obs_length = self.num_one_step_obs * self.actor_history_length

        # 初始化中包含一句 self.create_sim()
        # self.create_sim() 中包含 self._create_envs()
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._init_min_shoulder_root_height_reward()
        self._prepare_reward_function()
        self.init_done = True  # 初始化flag

    # =====================================================================
    #                         核心仿真流程
    # =====================================================================

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        # 课程辅助力准备 (力值在 step 期间不变, 只需准备一次)
        if self.cfg.curriculum.use_curriculum:
            self.curriculum_forces[:] = 0
            self.curriculum_forces[:, self.shoulder_root_index, 2] = self.force.squeeze(1)

        # 每步开始前清零子步累积量
        self.ee_in_contact[:] = False
        self.max_shoulder_root_acc_in_one_step[:] = 0.
        prev_slider_vel = self.dof_vel[:, self.slider_dof_idx].clone()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # 课程辅助力: 竖直向上施加在 shoulder_root 上
            if self.cfg.curriculum.use_curriculum:
                self.gym.apply_rigid_body_force_tensors(
                    self.sim,
                    gymtorch.unwrap_tensor(self.curriculum_forces.view(-1, 3)),
                    None, gymapi.ENV_SPACE
                )
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

            # 每个子步刷新接触力, 累积峰值和接触状态
            self.gym.refresh_net_contact_force_tensor(self.sim)
            substep_force = torch.norm(self.contact_forces[:, self.end_idx, :], dim=-1)
            # accumulate per-substep contact into a per-step boolean flag (in-place to keep tensor identity)
            self.ee_in_contact[:] |= (substep_force > 0.1)
            # 滑块加速度: 子步间速度差 / 子步时间步长
            current_slider_vel = self.dof_vel[:, self.slider_dof_idx]
            substep_slider_acc = torch.abs(current_slider_vel - prev_slider_vel) / self.sim_params.dt
            self.max_shoulder_root_acc = torch.maximum(self.max_shoulder_root_acc, substep_slider_acc)
            self.max_shoulder_root_acc_in_one_step = torch.maximum(self.max_shoulder_root_acc_in_one_step, substep_slider_acc)
            prev_slider_vel = current_slider_vel.clone()
            self.max_shoulder_pitch_torque = torch.maximum(self.max_shoulder_pitch_torque, torch.abs(self.torques[:, self.shoulder_pitch_dof_idx]))
            # use DOF-index for accessing torques (self.torques is DOF-indexed)
            self.max_elbow_torque = torch.maximum(self.max_elbow_torque, torch.abs(self.torques[:, self.elbow_dof_idx]))
            self.min_shoulder_root_height = torch.minimum(self.min_shoulder_root_height, self.dof_pos[:, self.slider_dof_idx])

        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        if torch.isnan(self.dof_pos).any():
            print("PHYSICS EXPLODED: dof_pos NaN")
        if torch.isnan(self.dof_vel).any():
            print("PHYSICS EXPLODED: dof_vel NaN")
        # 检查是否需要终止 self.check_termination()
        # 计算奖励 self.compute_reward()
        # 重置需要终止的环境 self.reset_idx(env_ids)
        # 计算观测值 self.compute_observations()
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.real_episode_length_buf += 1

        # 连杆位置
        self.shoulder_pitch_pos[:] = self.rigid_body_states[:, self.shoulder_pitch_idx, :3]
        self.shoulder_roll_pos[:] = self.rigid_body_states[:, self.shoulder_roll_idx, :3]
        self.shoulder_yaw_pos[:] = self.rigid_body_states[:, self.shoulder_yaw_idx, :3]
        self.elbow_pos[:] = self.rigid_body_states[:, self.elbow_idx, :3]
        self.end_effector_pos[:] = self.rigid_body_states[:, self.end_idx, :3]

        # 接触检测 & 回合级统计量已在 decimation 子步循环中完成累积

        # 当前步 shoulder_root 高度 (用于每步奖励函数)
        self.shoulder_root_height[:] = self.dof_pos[:, self.slider_dof_idx]

        # 更新连续无接触计数器及离地候选标志
        contact_mask = self.ee_in_contact
        no_contact_mask = ~contact_mask
        # 若本步有接触：计数清零，并清除候选（再次接触取消此前的离地候选）
        if contact_mask.any():
            self.ee_no_contact_counter[contact_mask] = 0
            self.ee_left_candidate[contact_mask] = False
            self.ee_ever_contacted[contact_mask] = True
        # 若本步无接触：计数自增
        if no_contact_mask.any():
            self.ee_no_contact_counter[no_contact_mask] += 1
        # 在刚从接触变为无接触的帧（即 prev_ee_in_contact 为 True 且 本帧无接触）标记为离地候选
        start_candidate = self.prev_ee_in_contact & no_contact_mask
        if start_candidate.any():
            self.ee_left_candidate[start_candidate] = True

        self.check_termination()
        self.compute_reward()

        # 记录历史 (在 reset 之前更新, 避免 reset 清零被覆盖)
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.prev_shoulder_root_vel[:] = self.rigid_body_states[:, self.shoulder_root_index, 7:10]
        self.prev_ee_vel[:] = self.rigid_body_states[:, self.end_idx, 7:10]

        # ========== 新增：肩根高度第一次升高事件检测 ==========
        locked_mask = ~self.min_shoulder_root_height_reward_unlocked
        curr_height = self.shoulder_root_height
        prev_height = self._prev_shoulder_root_height
        unlocked_now = (curr_height > prev_height) & locked_mask
        self.min_shoulder_root_height_reward_unlocked[unlocked_now] = True
        self._prev_shoulder_root_height = curr_height.clone()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # =====================================================================
    #                       仿真 / 环境创建
    # =====================================================================

    def create_sim(self):
        """创建仿真、地面与所有环境实例"""
        self.up_axis_idx = 2  # z-up
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == 'plane':
            self._create_ground_plane()
        self._create_envs()

    def _create_ground_plane(self):
        """添加平面地面"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        print("Bodies:", self.gym.get_asset_rigid_body_count(robot_asset))
        print("DOFs:", self.gym.get_asset_dof_count(robot_asset))
        print("Joints:", self.gym.get_asset_joint_count(robot_asset))
        # Also print names for easier debugging (fall back safely if not available)
        try:
            body_names_dbg = self.gym.get_asset_rigid_body_names(robot_asset)
            print("Body names:", body_names_dbg)
        except Exception as e:
            print("Body names: unavailable -", e)
        try:
            dof_names_dbg = self.gym.get_asset_dof_names(robot_asset)
            print("DOF names:", dof_names_dbg)
        except Exception as e:
            print("DOF names: unavailable -", e)
        # joint names API may not exist on all gym versions
        if hasattr(self.gym, 'get_asset_joint_names'):
            try:
                joint_names_dbg = self.gym.get_asset_joint_names(robot_asset)
                print("Joint names:", joint_names_dbg)
            except Exception as e:
                print("Joint names: unavailable -", e)
        else:
            print("Joint names: API not available in this gym build")

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # expose body names for shape processing
        self.body_names = body_names
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        self.slider_dof_idx = next(i for i, n in enumerate(self.dof_names) if 'shoulder_root' in n)
        self.arm_dof_indices = [i for i in range(self.num_dofs) if i != self.slider_dof_idx]

        # 惩罚和终止条件
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # 初始状态
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.cfg.init_state.pos)

        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
        self.shoulder_root_index = next(i for i, name in enumerate(body_names) if 'shoulder_root' in name)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.envs = []
        self.actor_handles = []

        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            # xyz 方向上放大倍数
            self.com_displacement[:, 0] = self.com_displacement[:, 0] * 4
            self.com_displacement[:, 1] = self.com_displacement[:, 1] * 4
            self.com_displacement[:, 2] = self.com_displacement[:, 2] * 2

        for i in range(self.num_envs):
            # env handle 创建
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            # actor 初始位置设置
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            # 摩擦系数和恢复系数随机化
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            # actor handle 创建
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)

            # 关节属性处理
            dof_props = self._process_dof_props(dof_props_asset, i)

            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

            if i == 0:
                self.default_com_shoulder_root = copy.deepcopy(body_props[self.shoulder_root_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass

            # 负载质量、质心偏移、连杆质量随机化
            body_props = self._process_rigid_body_props(body_props, i)

            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            # 存储在 envs 和 actor_handles 中
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # 下面开始录入各个连杆的索引
        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        base_names = [s for s in body_names if self.cfg.asset.base_name in s]
        self.base_indices = torch.zeros(len(base_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(base_names)):
            self.base_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], base_names[i]
            )

        shoulder_root_names = [s for s in body_names if self.cfg.asset.shoulder_root_name in s]
        self.shoulder_root_indices = torch.zeros(len(shoulder_root_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(shoulder_root_names)):
            self.shoulder_root_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], shoulder_root_names[i]
            )

        shoulder_pitch_names = [s for s in body_names if self.cfg.asset.shoulder_pitch_name in s]
        self.shoulder_pitch_indices = torch.zeros(len(shoulder_pitch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(shoulder_pitch_names)):
            self.shoulder_pitch_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], shoulder_pitch_names[i]
            )
        self.shoulder_pitch_idx = self.shoulder_pitch_indices[0].item()
        print(f"Identified shoulder_pitch body index: {self.shoulder_pitch_idx} (name: {body_names[self.shoulder_pitch_idx]})")

        shoulder_roll_names = [s for s in body_names if self.cfg.asset.shoulder_roll_name in s]
        self.shoulder_roll_indices = torch.zeros(len(shoulder_roll_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(shoulder_roll_names)):
            self.shoulder_roll_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], shoulder_roll_names[i]
            )
        self.shoulder_roll_idx = self.shoulder_roll_indices[0].item()
        print(f"Identified shoulder_roll body index: {self.shoulder_roll_idx} (name: {body_names[self.shoulder_roll_idx]})")

        shoulder_yaw_names = [s for s in body_names if self.cfg.asset.shoulder_yaw_name in s]
        self.shoulder_yaw_indices = torch.zeros(len(shoulder_yaw_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(shoulder_yaw_names)):
            self.shoulder_yaw_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], shoulder_yaw_names[i]
            )
        self.shoulder_yaw_idx = self.shoulder_yaw_indices[0].item()
        print(f"Identified shoulder_yaw body index: {self.shoulder_yaw_idx} (name: {body_names[self.shoulder_yaw_idx]})")

        elbow_names = [s for s in body_names if self.cfg.asset.elbow_name in s]
        self.elbow_indices = torch.zeros(len(elbow_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(elbow_names)):
            self.elbow_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], elbow_names[i]
            )
        self.elbow_idx = self.elbow_indices[0].item()
        print(f"Identified elbow body index: {self.elbow_idx} (name: {body_names[self.elbow_idx]})")

        end_names = [s for s in body_names if self.cfg.asset.end_name in s]
        self.end_indices = torch.zeros(len(end_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(end_names)):
            self.end_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], end_names[i]
            )
        self.end_idx = self.end_indices[0].item()
        print(f"Identified end-effector body index: {self.end_idx} (name: {body_names[self.end_idx]})")

        self.slider_dof_idx = next(i for i, n in enumerate(self.dof_names) if 'shoulder_root' in n)
        print(f"Identified slider DOF index: {self.slider_dof_idx} (name: {self.dof_names[self.slider_dof_idx]})")
        self.arm_dof_indices = [i for i in range(self.num_dofs) if i != self.slider_dof_idx]
        print(f"Identified arm DOF indices: {self.arm_dof_indices} (names: {[self.dof_names[i] for i in self.arm_dof_indices]})")
        self.shoulder_pitch_dof_idx = next(i for i, n in enumerate(self.dof_names) if 'shoulder_pitch' in n)
        print(f"Identified shoulder_pitch DOF index: {self.shoulder_pitch_dof_idx} (name: {self.dof_names[self.shoulder_pitch_dof_idx]})")
        self.shoulder_roll_dof_idx = next(i for i, n in enumerate(self.dof_names) if 'shoulder_roll' in n)
        print(f"Identified shoulder_roll DOF index: {self.shoulder_roll_dof_idx} (name: {self.dof_names[self.shoulder_roll_dof_idx]})")
        self.shoulder_yaw_dof_idx = next(i for i, n in enumerate(self.dof_names) if 'shoulder_yaw' in n)
        print(f"Identified shoulder_yaw DOF index: {self.shoulder_yaw_dof_idx} (name: {self.dof_names[self.shoulder_yaw_dof_idx]})")
        self.elbow_dof_idx = next(i for i, n in enumerate(self.dof_names) if 'elbow' in n)
        print(f"Identified elbow DOF index: {self.elbow_dof_idx} (name: {self.dof_names[self.elbow_dof_idx]})")

    def _get_env_origins(self):
        # 为每个机器人实例分配一个不重叠的初始位置
        self.custom_origins = False
        # env_origins (num_envs, 3) 存储每个环境的原点位置
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.0

    # =====================================================================
    #                     配置解析
    # =====================================================================

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.constraint_scales = class_to_dict(self.cfg.constraints.scales)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    # =====================================================================
    #                     资产属性回调 (创建环境时调用)
    # =====================================================================

    def _process_rigid_shape_props(self, props, env_id):
        # allow randomizing friction/restitution as before
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs, 1), device=self.device)
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs, 1), device=self.device)
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            print("DOF limits:")
            for i, name in enumerate(self.dof_names):
                lower = props['lower'][i].item()
                upper = props['upper'][i].item()
                print(name, lower, upper)
            # 位置限制: 所有 DOF
            self.dof_pos_limits = torch.zeros(len(props), 2, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                if i == self.slider_dof_idx:
                    # slider: 硬限制 (不乘 soft 系数)
                    self.dof_pos_limits[i, 0] = props['lower'][i].item()
                    self.dof_pos_limits[i, 1] = props['upper'][i].item()
                else:
                    # 手臂关节: 软限制
                    self.dof_pos_limits[i, 0] = props['lower'][i].item() * self.cfg.limitation.soft_dof_pos_limit
                    self.dof_pos_limits[i, 1] = props['upper'][i].item() * self.cfg.limitation.soft_dof_pos_limit

            # 速度 / 力矩限制: 仅手臂 DOF (slider 为被动自由度, 不需要)
            self.dof_vel_limits = torch.zeros(len(self.arm_dof_indices), dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(len(self.arm_dof_indices), dtype=torch.float, device=self.device, requires_grad=False)
            for j, i in enumerate(self.arm_dof_indices):
                self.dof_vel_limits[j] = props['velocity'][i].item()
                self.torque_limits[j] = props['effort'][i].item()
        return props

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.shoulder_root_index].mass = self.default_rigid_body_mass[self.shoulder_root_index] + self.payload[env_id, 0]

        if self.cfg.domain_rand.randomize_com_displacement:
            props[self.shoulder_root_index].com = self.default_com_shoulder_root + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(0, len(props)):
                if i == self.shoulder_root_index:
                    continue
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props

    # =====================================================================
    #                     Buffer / 张量初始化
    # =====================================================================

    def _init_buffers(self):
        # ---------- GPU 状态张量 ----------
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # ---------- 包装为 PyTorch 张量 ----------
        self.dof_state = gymtorch.wrap_tensor(dof_state)
        self.dof_states = self.dof_state.view(self.num_envs, self.num_dofs, 2)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, 13
        )

        # ---------- 坠落高度范围 ----------
        if hasattr(self.cfg.init_state, 'drop_height_range'):
            self.drop_height_min = self.cfg.init_state.drop_height_range[0]
            self.drop_height_max = self.cfg.init_state.drop_height_range[1]

        # ---------- 观测量 ----------
        self.obs_buf = torch.zeros(
            self.num_envs,
            self.actor_proprioceptive_obs_length,
            device=self.device,
            dtype=torch.float
        )

        # ---------- 通用 buffer ----------
        # 仿真步数计数器
        self.common_step_counter = 0
        # 额外信息
        self.extras = {}
        # 噪声
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # 各关节力矩 (num_dofs 维, 覆盖所有 DOF 包含 slider)
        self.torques = torch.zeros(
            self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False
        )

        # PD 增益 (num_actions 维)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # 动作 actions（当前、上一个、上上个）
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )

        # ---------- 末端执行器 & 接触 buffer ----------
        self.shoulder_pitch_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.shoulder_roll_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.shoulder_yaw_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.elbow_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.end_effector_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.ee_in_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.prev_ee_in_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.ee_no_contact_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.ee_left_candidate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.ee_ever_contacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.max_shoulder_root_acc = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.max_shoulder_root_acc_in_one_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.max_shoulder_pitch_torque = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.max_elbow_torque = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.min_shoulder_root_height = torch.full(
            (self.num_envs,), float('inf'), dtype=torch.float, device=self.device
        )
        self.shoulder_root_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.prev_shoulder_root_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.prev_ee_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)

        # 动作缩放
        self.action_rescale = torch.full(
            (self.num_envs, 1),
            self.cfg.control.action_scale,
            device=self.device
        )
        # 模拟延迟
        self.delay_buffer = torch.zeros(self.cfg.domain_rand.max_delay_timesteps, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # ---------- 默认关节角 & PD 增益 ----------
        self.default_dof_pos = torch.zeros(
            self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

        # PD 增益仅赋值给手臂关节 (num_actions 维)
        for arm_j, dof_i in enumerate(self.arm_dof_indices):
            name = self.dof_names[dof_i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[arm_j] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[arm_j] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[arm_j] = 0.0
                self.d_gains[arm_j] = 0.0
                if self.cfg.control.control_type in ['P', 'V']:
                    print(f'PD gain of joint {name} were not defined, setting them to zero')
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        print({self.default_dof_pos})
        print("Default DOF positions:", {self.dof_names[i]: self.default_dof_pos[0, i].item() for i in range(self.num_dofs)})

        # 随机化 kp、kd、actuation_offset、motor_strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_actions), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.delay:
            self.delay_idx = torch.randint(low=0, high=self.cfg.domain_rand.max_delay_timesteps, size=(self.num_envs,), device=self.device)

        # ---------- 课程学习 buffer ----------
        if self.cfg.curriculum.use_curriculum:
            self.force = self.cfg.curriculum.force_initial * torch.ones(
                self.num_envs, 1, dtype=torch.float, device=self.device
            )
            self.curriculum_forces = torch.zeros(
                self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device
            )

    # =====================================================================
    #                     奖励系统
    # =====================================================================

    def _prepare_reward_function(self):
        # 移除零 scale, 非零 scale 乘 dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= 1

        for key in list(self.constraint_scales.keys()):
            scale = self.constraint_scales[key]
            if scale == 0:
                self.constraint_scales.pop(key)
            else:
                self.constraint_scales[key] *= self.dt  # constraints 权重乘以时间步长

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == 'termination':
                continue
            self.reward_names.append(name)
            name = '_reward_' + '_'.join(name.split('_')[1:])
            self.reward_functions.append(getattr(self, name))
        self.constraint_functions = []
        self.constraint_names = []
        for name, scale in self.constraint_scales.items():
            self.constraint_names.append(name)
            name = '_reward_' + '_'.join(name.split('_')[1:])
            self.constraint_functions.append(getattr(self, name))

        self.episode_sums = {}
        for name in self.reward_scales.keys():
            self.episode_sums[name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        for name in self.constraint_scales.keys():
            self.episode_sums[name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        for rg in self.reward_groups:
            self.episode_sums[rg] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def compute_reward(self):
        self.rew_buf[:, :] = 0
        task_group_index = self.reward_groups.index('task')
        self.rew_buf[:, task_group_index] = 1

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if len(rew.shape) == 2 and rew.shape[1] == 1:
                rew = rew.squeeze(1)
            self.rew_buf[:, task_group_index] *= rew
            self.episode_sums[name] += rew

        # 异常终止 → task 组乘零归零 + 施加惩罚
        abnormal_termination = self._reward_termination()  # 异常终止: bool mask
        # 将布尔掩码转换为浮点掩码; 若为 True 表示异常终止 -> 生存 mask 为 0.0
        survival_mask = (~abnormal_termination).float()
        self.rew_buf[:, task_group_index] *= survival_mask
        # termination scale 控制异常终止的惩罚力度 (负值 → 惩罚)
        termination_rew = abnormal_termination.float() * self.reward_scales['termination']
        self.rew_buf[:, task_group_index] += termination_rew
        self.episode_sums['termination'] += termination_rew

        for i in range(len(self.constraint_functions)):
            name = self.constraint_names[i]
            reward_group_name = name.split('_')[0]  # 提取前缀: regu/style/target
            rew = self.constraint_functions[i]() * self.constraint_scales[name]
            group_index = self.reward_groups.index(reward_group_name)
            self.rew_buf[:, group_index] += rew
            self.episode_sums[name] += rew

        # reward_groups = ['task', 'regu', 'style', 'target'] 四类汇总
        for rg in self.reward_groups:
            idx = self.reward_groups.index(rg)
            self.episode_sums[rg] += self.rew_buf[:, idx]

    # =====================================================================
    #                     PD 控制器
    # =====================================================================

    def _compute_torques(self, actions):
        # PD 控制器: 在 4-dim 手臂空间完成全部计算, 最后组装 5-dim DOF 力矩
        # actions: (num_envs, num_actions=4) 仅手臂关节

        # ---- 1. 动作缩放与延迟 (4-dim 手臂空间) ----
        actions_scaled = actions * self.action_rescale  # (num_envs, 4)

        if self.cfg.domain_rand.delay:
            self.delay_buffer = torch.concat((self.delay_buffer[1:], actions_scaled.unsqueeze(0)), dim=0)
            delayed_actions = self.delay_buffer[self.delay_idx, torch.arange(len(self.delay_idx)), :]
        else:
            delayed_actions = actions_scaled

        # ---- 2. PD 计算 (4-dim 手臂空间) ----
        arm_dof_pos = self.dof_pos[:, self.arm_dof_indices]
        arm_dof_vel = self.dof_vel[:, self.arm_dof_indices]

        self.joint_pos_target = arm_dof_pos + delayed_actions

        control_type = self.cfg.control.control_type
        if control_type == 'P':
            arm_torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - arm_dof_pos) - self.d_gains * self.Kd_factors * arm_dof_vel
        elif control_type == 'V':
            arm_torques = self.p_gains * (delayed_actions - arm_dof_vel) - self.d_gains * (arm_dof_vel - self.last_dof_vel[:, self.arm_dof_indices]) / self.sim_params.dt
        elif control_type == 'T':
            arm_torques = delayed_actions
        else:
            raise NameError(f'Unknown controller type: {control_type}')

        # ---- 3. 驱动随机化 (4-dim 手臂空间) ----
        arm_torques = self.motor_strength * arm_torques + self.actuation_offset

        # ---- 4. 力矩限幅 (4-dim 手臂空间) ----
        arm_torques = torch.clip(arm_torques, -self.torque_limits, self.torque_limits)

        # ---- 5. 组装 5-dim DOF 力矩 ----
        torques = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        torques[:, self.arm_dof_indices] = arm_torques

        # 导轨滑块摩擦力: 粘性摩擦 + 库仑摩擦 (不受 motor_strength / clip 影响)
        slider_vel = self.dof_vel[:, self.slider_dof_idx]
        torques[:, self.slider_dof_idx] = -self.cfg.control.slider_viscous_friction * slider_vel - self.cfg.control.slider_coulomb_friction * torch.sign(slider_vel)

        return torques

    # =====================================================================
    #                           观测计算
    # =====================================================================

    def compute_observations(self):
        if torch.isnan(self.dof_pos).any():
            print("NaN in dof_pos")
        if torch.isnan(self.dof_vel).any():
            print("NaN in dof_vel")
        if torch.isnan(self.actions).any():
            print("NaN in actions")

        arm_dof_pos = (self.dof_pos[:, self.arm_dof_indices] - self.default_dof_pos[:, self.arm_dof_indices]) * self.obs_scales.dof_pos
        arm_dof_vel = self.dof_vel[:, self.arm_dof_indices] * self.obs_scales.dof_vel
        arm_actions = self.actions

        current_obs = torch.cat([
            arm_dof_pos, arm_dof_vel, arm_actions,
            self.action_rescale + (torch.rand_like(self.action_rescale) - 0.5) * 0.05,
        ], dim=-1)

        if torch.isnan(current_obs).any():
            print("NaN in current_obs")

        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec

        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_proprioceptive_obs_length], current_obs), dim=-1)
        self.obs_buf = torch.nan_to_num(self.obs_buf, 0.0)

        self.prev_ee_in_contact = self.ee_in_contact.clone()

    def _get_noise_scale_vec(self, cfg):
        # 构建与观测维度一致的噪声缩放向量 (13维)
        noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        idx = 0

        # arm_dof_pos (4)
        noise_vec[idx:idx + self.num_real_dofs] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        idx += self.num_real_dofs
        # arm_dof_vel (4)
        noise_vec[idx:idx + self.num_real_dofs] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        idx += self.num_real_dofs
        # arm_actions (4) - 策略输出, 非传感器量测, 无噪声
        noise_vec[idx:idx + self.num_real_dofs] = 0.0
        idx += self.num_real_dofs
        # action_rescale (1)
        noise_vec[idx:idx + 1] = 0.0
        idx += 1
        return noise_vec

    # =====================================================================
    #                           终止条件
    # =====================================================================

    def check_termination(self):
        # 非法部位触地
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            ) > 1., dim=1,
        )
        # 超时
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
        # 超速
        if hasattr(self.cfg, 'limitation'):
            dof_vel_exceeded = torch.any(
                torch.abs(self.dof_vel[:, self.arm_dof_indices]) > self.cfg.limitation.dof_vel_limit, dim=1,
            )
            slider_vel_exceeded = (
                torch.abs(self.dof_vel[:, self.slider_dof_idx]) > self.cfg.limitation.slider_vel_limit
            )
            self.reset_buf |= dof_vel_exceeded
            self.reset_buf |= slider_vel_exceeded
            # slider 位置低于阈值时终止
            if hasattr(self.cfg.limitation, 'slider_pos_min') and self.cfg.limitation.slider_pos_min is not None:
                slider_pos_below = (self.dof_pos[:, self.slider_dof_idx] < float(self.cfg.limitation.slider_pos_min))
                self.reset_buf |= slider_pos_below

    # =====================================================================
    #                           环境重置
    # =====================================================================

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self.extras['episode'] = {}
        self._reset_dofs(env_ids)

        # 难度增加
        self._update_force_curriculum(env_ids)

        # 清零 buffer
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.prev_shoulder_root_vel[env_ids] = 0.
        self.prev_ee_vel[env_ids] = 0.
        self.obs_buf[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.real_episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # 清理接触/离地相关状态，避免 reset 导致下一帧误判
        self.ee_in_contact[env_ids] = False
        self.prev_ee_in_contact[env_ids] = False
        self.ee_no_contact_counter[env_ids] = 0
        self.ee_left_candidate[env_ids] = False
        self.ee_ever_contacted[env_ids] = False

        # ========== 新增：肩根最小高度奖励相关状态重置 ==========
        self.min_shoulder_root_height_reward_unlocked[env_ids] = False
        self._prev_shoulder_root_height[env_ids] = self.shoulder_root_height[env_ids]

        # 回合奖励统计
        for key in self.episode_sums.keys():
            self.extras['episode']['rew_' + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.

        if self.cfg.env.send_timeouts:
            self.extras['time_outs'] = self.time_out_buf

        # 回合级统计量日志
        if len(env_ids) > 0:
            self.extras['episode']['max_shoulder_root_acc'] = torch.mean(self.max_shoulder_root_acc[env_ids])
            self.extras['episode']['max_shoulder_pitch_torque'] = torch.mean(self.max_shoulder_pitch_torque[env_ids])
            self.extras['episode']['max_elbow_torque'] = torch.mean(self.max_elbow_torque[env_ids])
            self.extras['episode']['min_shoulder_root_height'] = torch.mean(self.min_shoulder_root_height[env_ids])
            self.extras['episode']['final_shoulder_root_height'] = torch.mean(self.shoulder_root_height[env_ids])
        self.max_shoulder_root_acc[env_ids] = 0.
        self.max_shoulder_pitch_torque[env_ids] = 0.
        self.max_elbow_torque[env_ids] = 0.
        self.min_shoulder_root_height[env_ids] = float('inf')

        self.extras['episode']['curriculum_force'] = torch.mean(self.force[env_ids])
        self.extras['episode']['curriculum_action_rescale'] = torch.mean(self.action_rescale[env_ids])

    def _reset_dofs(self, env_ids):
        # ---- 滑块关节: 仅偏移, 从 drop_height_range 直接采样 (不乘缩放) ----
        slider_pos = torch_rand_float(
            self.drop_height_min, self.drop_height_max,
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # ---- 超限检测 ----
        slider_lower = self.dof_pos_limits[self.slider_dof_idx, 0]
        slider_upper = self.dof_pos_limits[self.slider_dof_idx, 1]
        if torch.any(slider_pos < slider_lower) or torch.any(slider_pos > slider_upper):
            print("WARNING: slider initial pos exceeds limits")
            print("slider_pos:", slider_pos[:5])
            print("limit:", slider_lower.item(), slider_upper.item())
        # ---- 强制 clamp ----
        slider_pos = torch.clip(slider_pos, slider_lower, slider_upper)
        self.dof_pos[env_ids, self.slider_dof_idx] = slider_pos

        # ---- 手臂关节: 缩放 × default + 偏移 ----
        arm_default = self.default_dof_pos[:, self.arm_dof_indices]  # (1, num_real_dofs)
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_arm_pos = arm_default * torch_rand_float(
                self.cfg.domain_rand.initial_joint_pos_scale[0],
                self.cfg.domain_rand.initial_joint_pos_scale[1],
                (len(env_ids), self.num_real_dofs), device=self.device
            )
            init_arm_pos += torch_rand_float(
                self.cfg.domain_rand.initial_joint_pos_offset[0],
                self.cfg.domain_rand.initial_joint_pos_offset[1],
                (len(env_ids), self.num_real_dofs), device=self.device
            )
            arm_lower = self.dof_pos_limits[self.arm_dof_indices, 0]
            arm_upper = self.dof_pos_limits[self.arm_dof_indices, 1]
            init_arm_pos = torch.clip(init_arm_pos, arm_lower, arm_upper)
            if torch.isnan(init_arm_pos).any():
                print("NaN in init_arm_pos")
            if torch.any(init_arm_pos < arm_lower) or torch.any(init_arm_pos > arm_upper):
                print("WARNING: arm pos exceeds limits")
        else:
            init_arm_pos = arm_default * torch_rand_float(
                0.9, 1.1, (len(env_ids), self.num_real_dofs), device=self.device
            )

        self.dof_pos[env_ids.unsqueeze(1), self.arm_dof_indices] = init_arm_pos
        # ---- 最终安全检查 ----
        if torch.isnan(self.dof_pos[env_ids]).any():
            print("NaN detected in dof_pos after reset")
        lower = self.dof_pos_limits[:, 0]
        upper = self.dof_pos_limits[:, 1]
        if torch.any(self.dof_pos[env_ids] < lower) or torch.any(self.dof_pos[env_ids] > upper):
            print("WARNING: dof_pos exceeds limits after reset")

        # 速度清零
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.refresh_dof_state_tensor(self.sim)

    def _update_force_curriculum(self, env_ids):
        # 课程学习: 根据回合表现逐步降低辅助, 增加任务难度
        # 整体思路:
        #   训练初期给 shoulder_root 施加较大的向上辅助力 (接近抵消重力),
        #   使策略先在"简单模式"下学会基本的手臂控制和着陆姿态;
        #   当策略足够好 (回合内始终维持一定高度) 时, 逐步减小辅助力,
        #   最终 force=0, 策略必须在完全自由落体下完成软着陆.
        #   action_rescale 同步递减, 逐步收紧动作幅度, 要求更精细的控制.

        if not self.cfg.curriculum.use_curriculum:
            return

        # 判断哪些环境"通过"了本回合的考核:
        #   min_shoulder_root_height[i] 记录了环境 i 整个回合中 shoulder_root 的最低高度
        passed = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        if self.cfg.curriculum.check_min_shoulder_root_height:
            passed &= (self.min_shoulder_root_height[env_ids] > self.cfg.curriculum.min_shoulder_root_height_lower_threshold) & \
                      (self.min_shoulder_root_height[env_ids] < self.cfg.curriculum.min_shoulder_root_height_upper_threshold)
        if self.cfg.curriculum.check_final_shoulder_root_height:
            passed &= (self.shoulder_root_height[env_ids] > self.cfg.curriculum.final_shoulder_root_height_lower_threshold) & \
                      (self.shoulder_root_height[env_ids] < self.cfg.curriculum.final_shoulder_root_height_upper_threshold)

        # 只对通过考核的环境增加难度 (未通过的保持当前难度继续练)
        passed_ids = env_ids[passed]

        if len(passed_ids) > 0:
            # 减小辅助力, 但不低于 force_min (通常为 0)
            self.force[passed_ids] = (
                self.force[passed_ids] - self.cfg.curriculum.force_decrement
            ).clamp(min=self.cfg.curriculum.force_min)
            # 减小动作缩放, 但不低于 action_rescale_min
            self.action_rescale[passed_ids] = (
                self.action_rescale[passed_ids] - self.cfg.curriculum.action_rescale_decrement
            ).clamp(min=self.cfg.curriculum.action_rescale_min)

    # =====================================================================
    #                         相机与可视化
    # =====================================================================

    def _draw_debug_vis(self):
        return

    # =====================================================================
    #                         奖励函数
    # =====================================================================

    # 异常终止

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    # task reward

    def _reward_shoulder_root_height(self):
        height_reward = tolerance(
            self.shoulder_root_height,
            bounds=(self.cfg.constraints.shoulder_root_height_threshold, np.inf),
            margin=self.cfg.constraints.shoulder_root_height_margin,
            value_at_margin=self.cfg.constraints.shoulder_root_height_value_at_margin,
        )
        return height_reward

    def _reward_arm_pose(self):
        arm_pos = self.dof_pos[:, self.arm_dof_indices]
        arm_default = self.default_dof_pos[:, self.arm_dof_indices]
        deviation = torch.sum(torch.square(arm_pos - arm_default), dim=1)
        # arm_pose_reward = tolerance(
        #     deviation,
        #     bounds=(-np.inf, self.cfg.constraints.arm_pose_threshold),
        #     margin=self.cfg.constraints.arm_pose_margin,
        #     value_at_margin=self.cfg.constraints.arm_pose_value_at_margin,
        # )
        arm_pose_reward = torch.exp(-deviation / self.cfg.constraints.arm_pose_sigma)
        return arm_pose_reward

    def _reward_all_dof_pos(self):
        shoulder_pitch_dof_pos = self.dof_pos[:, self.shoulder_pitch_dof_idx]
        shoulder_pitch_dof_good = shoulder_pitch_dof_pos > self.cfg.constraints.shoulder_pitch_dof_pos_threshold

        shoulder_roll_dof_pos = self.dof_pos[:, self.shoulder_roll_dof_idx]
        shoulder_roll_dof_default = self.default_dof_pos[:, self.shoulder_roll_dof_idx]
        shoulder_roll_dof_deviation = torch.abs(shoulder_roll_dof_pos - shoulder_roll_dof_default)
        shoulder_roll_dof_good = shoulder_roll_dof_deviation < 0.1

        shoulder_yaw_dof_pos = self.dof_pos[:, self.shoulder_yaw_dof_idx]
        shoulder_yaw_dof_default = self.default_dof_pos[:, self.shoulder_yaw_dof_idx]
        shoulder_yaw_dof_deviation = torch.abs(shoulder_yaw_dof_pos - shoulder_yaw_dof_default)
        shoulder_yaw_dof_good = shoulder_yaw_dof_deviation < 0.1

        elbow_dof_pos = self.dof_pos[:, self.elbow_dof_idx]
        elbow_dof_good = elbow_dof_pos > self.cfg.constraints.elbow_dof_pos_threshold
        return shoulder_pitch_dof_good.float() * shoulder_roll_dof_good.float() * shoulder_yaw_dof_good.float() * elbow_dof_good.float()

    # regularization reward

    def _reward_dof_acc(self):
        arm_vel = self.dof_vel[:, self.arm_dof_indices]
        arm_last_vel = self.last_dof_vel[:, self.arm_dof_indices]
        return torch.sum(torch.square((arm_last_vel - arm_vel) / self.dt), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel[:, self.arm_dof_indices]), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_jerk(self):
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques[:, self.arm_dof_indices]), dim=1)

    def _reward_joint_power(self):
        arm_vel = self.dof_vel[:, self.arm_dof_indices]
        arm_torques = self.torques[:, self.arm_dof_indices]
        return torch.sum(torch.abs(arm_vel) * torch.abs(arm_torques), dim=1)

    def _reward_dof_pos_limits(self):
        arm_pos = self.dof_pos[:, self.arm_dof_indices]
        arm_limits = self.dof_pos_limits[self.arm_dof_indices]
        out_of_limits = -(arm_pos - arm_limits[:, 0]).clip(max=0.0)
        out_of_limits += (arm_pos - arm_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        return torch.sum(
            (torch.abs(self.dof_vel[:, self.arm_dof_indices]) - self.dof_vel_limits * self.cfg.limitation.soft_dof_vel_limit).clip(min=0.0, max=1.0),
            dim=1,
        )

    # behavior reward

    def _reward_penalised_contact(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1,
            dim=1,
        ).float()

    def _reward_encourage_contact(self):
        after_land = self.shoulder_root_height < self.cfg.constraints.land_height
        reward = (2 * self.ee_in_contact.float() - 1.0) * after_land.float()
        return reward

    def _reward_no_releave_after_contact(self):
        threshold = int(self.cfg.constraints.no_releave_after_contact_threshold)
        # 接触后离地且超过阈值的环境，按离地时长累积惩罚（离地越久惩罚越大）
        penalize = self.ee_left_candidate & (self.ee_no_contact_counter >= threshold)
        penalty = (self.ee_no_contact_counter - threshold).clamp(min=0).float() * penalize.float()
        return penalty

    def _reward_shoulder_pitch_dof_pos(self):
        shoulder_pitch_dof_pos = self.dof_pos[:, self.shoulder_pitch_dof_idx]
        straight = shoulder_pitch_dof_pos < self.cfg.constraints.shoulder_pitch_dof_pos_threshold
        return straight.float()

    def _reward_shoulder_roll_dof_pos(self):
        shoulder_roll_dof_pos = self.dof_pos[:, self.shoulder_roll_dof_idx]
        shoulder_roll_dof_default = self.default_dof_pos[:, self.shoulder_roll_dof_idx]
        deviation = torch.abs(shoulder_roll_dof_pos - shoulder_roll_dof_default)
        return deviation

    def _reward_shoulder_yaw_dof_pos(self):
        shoulder_yaw_dof_pos = self.dof_pos[:, self.shoulder_yaw_dof_idx]
        shoulder_yaw_dof_default = self.default_dof_pos[:, self.shoulder_yaw_dof_idx]
        deviation = torch.abs(shoulder_yaw_dof_pos - shoulder_yaw_dof_default)
        return deviation

    def _reward_elbow_dof_pos(self):
        elbow_dof_pos = self.dof_pos[:, self.elbow_dof_idx]
        straight = elbow_dof_pos < self.cfg.constraints.elbow_dof_pos_threshold
        return straight.float()

    def _reward_ee_distance(self):
        roll_pos = self.shoulder_roll_pos  # (num_envs, 3)
        ee_pos = self.end_effector_pos  # (num_envs, 3)
        diff_xy = ee_pos[:, :2] - roll_pos[:, :2]
        distance = torch.norm(diff_xy, dim=1)
        ee_distance_reward = tolerance(
            distance,
            bounds=(0.0, self.cfg.constraints.ee_distance_threshold),
            margin=self.cfg.constraints.ee_distance_margin,
            value_at_margin=self.cfg.constraints.ee_distance_value_at_margin,
        )
        return ee_distance_reward

    def _reward_ee_vel(self):
        before_land = self.shoulder_root_height > self.cfg.constraints.land_height
        ee_vel = self.rigid_body_states[:, self.end_idx, 7:10]  # 末端线速度 (3,)
        shoulder_root_vel = self.rigid_body_states[:, self.shoulder_root_index, 7:10]  # shoulder_root 线速度 (3,)
        relative_speed = torch.norm(ee_vel - shoulder_root_vel, dim=-1)  # 相对速度标量
        return before_land.float() * relative_speed
        # speed = torch.norm(ee_vel, dim=-1)  # 绝对速度标量
        # return after_land.float() * speed

    def _reward_high_shoulder_root_height(self):
        threshold = self.cfg.constraints.high_shoulder_root_height_threshold
        height_reward = (self.shoulder_root_height - threshold).clamp(max=0.0)
        return height_reward

    def _reward_low_shoulder_root_height(self):
        threshold = self.cfg.constraints.low_shoulder_root_height_threshold
        excess = (self.shoulder_root_height - threshold).clamp(min=0.0)
        return self.ee_ever_contacted.float() * excess

    def _reward_min_shoulder_root_height(self):
        lower = self.cfg.constraints.min_shoulder_root_height_lower_threshold
        upper = self.cfg.constraints.min_shoulder_root_height_upper_threshold
        unlocked = self.min_shoulder_root_height_reward_unlocked
        reward = tolerance(
            self.shoulder_root_height,
            bounds=(lower, upper),
            margin=self.cfg.constraints.min_shoulder_root_height_margin,
            value_at_margin=self.cfg.constraints.min_shoulder_root_height_value_at_margin,
        )
        reward = (unlocked.float() * reward)
        return reward

    # effort reward

    def _reward_low_max_shoulder_pitch_torque(self):
        return torch.exp(-self.max_shoulder_pitch_torque / self.cfg.constraints.low_max_shoulder_pitch_torque_sigma)

    def _reward_low_max_elbow_torque(self):
        return torch.exp(-self.max_elbow_torque / self.cfg.constraints.low_max_elbow_torque_sigma)

    def _reward_low_max_shoulder_root_acc(self):
        return tolerance(
            self.max_shoulder_root_acc,
            bounds=(0.0, self.cfg.constraints.low_max_shoulder_root_acc_threshold),
            margin=self.cfg.constraints.low_max_shoulder_root_acc_margin,
            value_at_margin=self.cfg.constraints.low_max_shoulder_root_acc_value_at_margin,
        )

    # stabilization reward

    def _reward_target_ee_vel(self):
        after_land = self.shoulder_root_height < self.cfg.constraints.land_height
        ee_vel = self.rigid_body_states[:, self.end_idx, 7:10]  # 末端线速度 (3,)
        # shoulder_root_vel = self.rigid_body_states[:, self.shoulder_root_index, 7:10]  # shoulder_root 线速度 (3,)
        # relative_speed = torch.norm(ee_vel - shoulder_root_vel, dim=-1)  # 相对速度标量
        # return after_land.float() * relative_speed
        speed = torch.norm(ee_vel, dim=-1)  # 绝对速度标量
        return after_land.float() * speed

    def _reward_target_shoulder_root_height(self):
        after_land = self.shoulder_root_height < self.cfg.constraints.land_height
        threshold = self.cfg.constraints.target_shoulder_root_height_threshold
        sigma = self.cfg.constraints.target_shoulder_root_height_sigma
        # height_reward = torch.exp(-((self.shoulder_root_height - threshold) / sigma)**2)
        height_reward = 1 - torch.abs(self.shoulder_root_height - threshold) / threshold
        return after_land.float() * height_reward
