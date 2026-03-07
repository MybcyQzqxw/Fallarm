"""
FallArm 环境类 — 直接继承 BaseTask，不依赖 LeggedRobot

任务描述:
    一条4自由度手臂（肩部3DOF + 肘部1DOF）根部挂载在垂直导轨上，
    从一定高度自由坠落。目标是训练缓冲着地控制策略:
    - 末端执行器先着地
    - 通过肘关节的主动顺应控制吸收冲击能量
    - 既不硬着陆 (冲击力过大) 也不软着陆 (肘关节折叠到限位)

DOF 布局 (共5个, 与URDF一致):
    [0] slider_joint           - prismatic, z轴, 被动 (PD增益=0)
    [1] shoulder_pitch_joint   - revolute, 肩俯仰
    [2] shoulder_roll_joint    - revolute, 肩横滚
    [3] shoulder_yaw_joint     - revolute, 肩偏航
    [4] elbow_joint            - revolute, 肘弯曲

观测空间 (24维):
    [0:5]   dof_pos - default  (含slider高度偏差)
    [5:10]  dof_vel            (含slider下落速度)
    [10:15] 上一步动作
    [15:18] 重力方向 (基座坐标系下)
    [18:19] 末端执行器高度
    [19:22] 末端执行器速度(xyz)
    [22:23] 归一化的回合时间
    [23:24] 末端是否触地 (contact phase)

动作空间 (5维):
    [0] slider (无效, 增益为0, step中显式清零)
    [1] shoulder_pitch 目标角偏移
    [2] shoulder_roll  目标角偏移
    [3] shoulder_yaw   目标角偏移
    [4] elbow          目标角偏移
"""

import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.fall_arm.fall_arm_config import FallArmCfg
from legged_gym.utils.helpers import class_to_dict


class FallArm(BaseTask):
    """落臂缓冲控制任务环境 — 直接继承 BaseTask"""

    cfg: FallArmCfg

    def __init__(self, cfg: FallArmCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    # =====================================================================
    #                       仿真 / 环境创建
    # =====================================================================

    def create_sim(self):
        """创建仿真、地面与所有环境实例"""
        self.up_axis_idx = 2  # z-up
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params
        )
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
        """加载 URDF 资产, 为每个环境创建 actor"""
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
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # 刚体 / 关节名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        # 末端执行器 / 惩罚 / 终止 体名称
        ee_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # 初始状态
        base_init_state_list = (self.cfg.init_state.pos + self.cfg.init_state.rot +
                                self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel)
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # 网格布局
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose,
                self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # 刚体索引
        self.ee_indices = torch.zeros(len(ee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(ee_names)):
            self.ee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], ee_names[i]
            )

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

    def _get_env_origins(self):
        """以网格布局放置所有环境"""
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    # =====================================================================
    #                     资产属性回调 (创建环境时调用)
    # =====================================================================

    def _process_rigid_shape_props(self, props, env_id):
        """可选: 摩擦力随机化"""
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device='cpu'
                )
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """读取 URDF 中定义的关节限位, 计算软限位"""
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # 软限位
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        """可选: 基座质量随机化"""
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    # =====================================================================
    #                     Buffer / 张量初始化
    # =====================================================================

    def _init_buffers(self):
        """获取 GPU 状态张量, 初始化所有运行时 buffer"""
        # ---------- GPU 状态张量 ----------
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # ---------- 包装为 PyTorch 张量 ----------
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, 13
        )

        # ---------- 末端执行器索引 ----------
        self.end_effector_idx = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], self.cfg.asset.foot_name
        )

        # ---------- DOF 索引快捷方式 ----------
        self.slider_dof_idx = 0
        self.arm_dof_indices = list(range(1, self.num_dof))
        self.elbow_dof_idx = self.num_dof - 1

        # ---------- 坠落高度范围 ----------
        if hasattr(self.cfg.init_state, 'drop_height_range'):
            self.drop_height_min = self.cfg.init_state.drop_height_range[0]
            self.drop_height_max = self.cfg.init_state.drop_height_range[1]
        else:
            default_h = self.cfg.init_state.default_joint_angles.get('slider_joint', 1.5)
            self.drop_height_min = default_h - 0.3
            self.drop_height_max = default_h + 0.3

        # ---------- 通用 buffer ----------
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))

        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # ---------- 末端执行器专用 buffer ----------
        self.end_effector_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.end_effector_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.end_effector_contact_force = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.ee_in_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.max_impact_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # ---------- 默认关节角 & PD 增益 ----------
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    # =====================================================================
    #                     配置解析
    # =====================================================================

    def _parse_cfg(self, cfg):
        """从配置中提取常用量"""
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    # =====================================================================
    #                     奖励系统
    # =====================================================================

    def _prepare_reward_function(self):
        """扫描所有非零奖励 scale, 建立奖励函数列表"""
        # 移除零 scale, 非零 scale 乘 dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # 按名称查找 _reward_<name> 方法
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, '_reward_' + name))
        # 回合累计
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def compute_reward(self):
        """计算总奖励"""
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # termination 奖励在 clip 之后加入
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    # =====================================================================
    #                     PD 控制器
    # =====================================================================

    def _compute_torques(self, actions):
        """PD 控制器: 将动作 → 力矩"""
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                       - self.d_gains * self.dof_vel)
        elif control_type == "V":
            torques = (self.p_gains * (actions_scaled - self.dof_vel)
                       - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt)
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # =====================================================================
    #                         核心仿真流程
    # =====================================================================

    def step(self, actions):
        """执行一步: clip 动作 → slider 清零 → decimation 次物理仿真 → post_physics_step"""
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # slider 维度强制清零 (PD增益也是0, 但保持动作张量维度一致)
        self.actions[:, self.slider_dof_idx] = 0.

        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """刷新状态张量 → 计算末端状态 → 终止 → 奖励 → 重置 → 观测"""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 基座状态 (fixed base, 仍用于 projected_gravity)
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 末端执行器状态
        self.end_effector_pos[:] = self.rigid_body_states[:, self.end_effector_idx, :3]
        self.end_effector_vel[:] = self.rigid_body_states[:, self.end_effector_idx, 7:10]
        self.end_effector_contact_force[:] = self.contact_forces[:, self.end_effector_idx, :]

        # 接触阶段
        force_mag = torch.norm(self.end_effector_contact_force, dim=-1)
        self.ee_in_contact = (force_mag > 0.1).float()
        self.max_impact_force = torch.maximum(self.max_impact_force, force_mag)

        # 终止 → 奖励 → 重置 → 观测
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        # 记录历史 (last_last 在 last 之前更新)
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # =====================================================================
    #                           观测计算
    # =====================================================================

    def compute_observations(self):
        """
        构建 24 维观测向量:
            [0:5]   关节位置偏差 (含滑块高度)
            [5:10]  关节速度 (含滑块下落速度)
            [10:15] 上一步动作
            [15:18] 重力方向
            [18:19] 末端执行器高度
            [19:22] 末端执行器线速度
            [22:23] 归一化回合时间
            [23:24] 末端是否触地 (contact phase)
        """
        ee_height = self.end_effector_pos[:, 2:3]
        time_ratio = (self.episode_length_buf.float() / self.max_episode_length).unsqueeze(1)

        self.obs_buf = torch.cat([
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,    # (5)
            self.dof_vel * self.obs_scales.dof_vel,                              # (5)
            self.actions,                                                         # (5)
            self.projected_gravity,                                               # (3)
            ee_height * self.obs_scales.end_effector_height,                      # (1)
            self.end_effector_vel * self.obs_scales.end_effector_vel,             # (3)
            time_ratio,                                                           # (1)
            self.ee_in_contact.unsqueeze(1),                                      # (1)
        ], dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """构建与观测维度一致的噪声缩放向量 (24维)"""
        noise_vec = torch.zeros(self.num_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        idx = 0

        # dof_pos (5)
        noise_vec[idx:idx + self.num_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        idx += self.num_dof
        # dof_vel (5)
        noise_vec[idx:idx + self.num_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        idx += self.num_dof
        # previous actions (5) — 无噪声
        noise_vec[idx:idx + self.num_actions] = 0.
        idx += self.num_actions
        # projected gravity (3)
        noise_vec[idx:idx + 3] = noise_scales.gravity * noise_level
        idx += 3
        # end_effector_height (1)
        noise_vec[idx:idx + 1] = 0.02 * noise_level
        idx += 1
        # end_effector_vel (3)
        noise_vec[idx:idx + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.end_effector_vel
        idx += 3
        # time_ratio (1) — 无噪声
        noise_vec[idx:idx + 1] = 0.
        idx += 1
        # contact_phase (1) — 无噪声
        noise_vec[idx:idx + 1] = 0.
        return noise_vec

    # =====================================================================
    #                           终止条件
    # =====================================================================

    def check_termination(self):
        """
        终止条件:
            1. 非末端刚体接触地面 (如 slider_link 触地 → 失败)
            2. 关节速度超过安全限制 (防止仿真爆炸)
            3. 回合超时
        """
        # 条件1: 非法部位触地
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            ) > 1., dim=1,
        )

        # 条件2: 安全限制
        if hasattr(self.cfg, 'limitation'):
            dof_vel_exceeded = torch.any(
                torch.abs(self.dof_vel[:, 1:]) > self.cfg.limitation.dof_vel_limit, dim=1,
            )
            slider_vel_exceeded = (
                torch.abs(self.dof_vel[:, self.slider_dof_idx]) > self.cfg.limitation.slider_vel_limit
            )
            self.reset_buf |= dof_vel_exceeded
            self.reset_buf |= slider_vel_exceeded

        # 条件3: 超时
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    # =====================================================================
    #                           环境重置
    # =====================================================================

    def reset_idx(self, env_ids):
        """重置指定环境并记录回合统计"""
        if len(env_ids) == 0:
            return

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # 清零 buffer
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # 回合奖励统计
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.

        # 最大冲击力日志
        if len(env_ids) > 0:
            self.extras["episode"]["max_impact_force"] = torch.mean(self.max_impact_force[env_ids])
        self.max_impact_force[env_ids] = 0.

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        """重置 DOF: 滑块高度随机, 手臂关节加小扰动, 速度清零"""
        self.dof_pos[env_ids] = self.default_dof_pos.clone()

        # 手臂关节加小扰动
        arm_noise = torch_rand_float(
            -0.1, 0.1, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_pos[env_ids] += arm_noise

        # 滑块高度随机化
        slider_height = torch_rand_float(
            self.drop_height_min, self.drop_height_max,
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.dof_pos[env_ids, self.slider_dof_idx] = slider_height

        # 速度清零
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """基座 fix_base_link, root state 重置保持张量一致性"""
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    # =====================================================================
    #                         相机与可视化
    # =====================================================================

    def lookat(self, i):
        """重写跟随相机: 追踪滑块高度而非固定基座"""
        slider_height = self.dof_pos[i, self.slider_dof_idx].item()
        look_at_pos = self.env_origins[i].clone()
        look_at_pos[2] = slider_height
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)

    def _draw_debug_vis(self):
        """可视化末端执行器位置 (绿色球体)"""
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 8, 8, None, color=(0, 1, 0))
        for i in range(min(self.num_envs, 10)):
            ee_pos = self.end_effector_pos[i].cpu().numpy()
            sphere_pose = gymapi.Transform(
                gymapi.Vec3(ee_pos[0], ee_pos[1], ee_pos[2]), r=None
            )
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    # =====================================================================
    #                         奖励函数
    # =====================================================================
    # 命名规则: _reward_<name> 对应 rewards.scales.<name>

    def _reward_soft_landing(self):
        """软着陆奖励: 接触力越小得分越高, 仅在触地时生效"""
        force_mag = torch.norm(self.end_effector_contact_force, dim=-1)
        reward = torch.exp(-force_mag / self.cfg.rewards.cushioning_sigma)
        return reward * self.ee_in_contact

    def _reward_end_effector_contact(self):
        """末端触地奖励: 鼓励末端执行器接触地面"""
        return self.ee_in_contact

    def _reward_elbow_cushion(self):
        """肘关节缓冲奖励 (触地时): 距机械限位越远 → 奖励越高"""
        elbow_pos = self.dof_pos[:, self.elbow_dof_idx]
        elbow_lower = self.dof_pos_limits[self.elbow_dof_idx, 0]
        elbow_upper = self.dof_pos_limits[self.elbow_dof_idx, 1]
        range_size = elbow_upper - elbow_lower + 1e-6
        dist_from_lower = (elbow_pos - elbow_lower) / range_size
        dist_from_upper = (elbow_upper - elbow_pos) / range_size
        min_dist = torch.minimum(dist_from_lower, dist_from_upper)
        return min_dist * self.ee_in_contact

    def _reward_impact_deceleration(self):
        """平缓减速奖励 (触地时): 滑块加速度越小 → 缓冲效果越好"""
        slider_acc = torch.abs(
            self.dof_vel[:, self.slider_dof_idx] - self.last_dof_vel[:, self.slider_dof_idx]
        ) / self.dt
        reward = torch.exp(-slider_acc / self.cfg.rewards.deceleration_sigma)
        return reward * self.ee_in_contact

    def _reward_smoothness(self):
        """二阶动作平滑惩罚: penalize |a_t - 2*a_{t-1} + a_{t-2}|^2"""
        diff2 = self.actions - 2 * self.last_actions + self.last_last_actions
        return torch.sum(torch.square(diff2), dim=1)

    def _reward_collision(self):
        """惩罚非末端刚体的地面接触 (如大臂/小臂碰地)"""
        return torch.sum(
            1. * (torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
            ) > 0.1), dim=1,
        )

    def _reward_arm_extension(self):
        """空中阶段手臂前伸准备奖励: 末端执行器越靠近地面 → 奖励越高"""
        ee_height = self.end_effector_pos[:, 2]
        reward = torch.exp(-ee_height * 2.0)
        in_air = (1.0 - self.ee_in_contact)
        return reward * in_air

    def _reward_torques(self):
        """惩罚力矩 (跳过滑块DOF, 只计算手臂关节)"""
        return torch.sum(torch.square(self.torques[:, 1:]), dim=1)

    def _reward_dof_vel(self):
        """惩罚关节速度过大"""
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """惩罚关节加速度"""
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )

    def _reward_action_rate(self):
        """惩罚动作变化率"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_pos_limits(self):
        """惩罚关节位置超出软限位"""
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_termination(self):
        """异常终止惩罚 (非超时终止)"""
        return self.reset_buf * ~self.time_out_buf
