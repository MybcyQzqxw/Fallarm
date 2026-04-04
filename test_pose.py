"""
Fall Arm 机器人姿态测试脚本
在 Isaac Gym 中观察机器人的指定关节角度姿态

使用方法:
    python test_pose.py

按键控制:
    - ESC: 退出
    - V: 切换相机视角
    - R: 重置机器人姿态
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'legged_gym'))

from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
import torch
import time


class FallArmPoseViewer:
    def __init__(self):
        # 目标关节角度配置
        # 滑块关节 (prismatic) 单位为米, 手臂关节 (revolute) 单位为度
        # 注意：肘关节 (elbow) 有限位约束，最小约10度
        self.target_joint_angles_deg = {
            'left_shoulder_pitch_joint': 45,    # 肩部俯仰 [deg]
            'left_shoulder_roll_joint': 0,     # 肩部横滚 [deg]
            'left_shoulder_yaw_joint': 0,      # 肩部偏航 [deg]
            'left_elbow_joint': 90,            # 肘关节 [deg]
        }

        # 滑块关节单独配置 (单位: 米, 控制初始坠落高度)
        self.slider_joint_pos = {
            'left_shoulder_root_joint': 0.45,     # 初始高度 [m]
        }

        # 基座姿态配置 (fall_arm 基座固定在原点)
        self.base_position = [0.0, 0.0, 0.0]    # x, y, z [m]
        self.base_orientation = [0, 0, 0, 1]     # 四元数 [x, y, z, w]

        # 转换手臂关节角度为弧度
        self.target_joint_angles = {
            k: np.deg2rad(v) for k, v in self.target_joint_angles_deg.items()
        }
        # 合并滑块关节 (已经是米, 无需转换)
        self.target_joint_angles.update(self.slider_joint_pos)
        # 初始化 gym
        self.gym = gymapi.acquire_gym()
        # 创建仿真
        self._create_sim()
        # 加载机器人
        self._load_robot()
        # 创建环境
        self._create_env()
        # 创建viewer
        self._create_viewer()
        # 设置关节角度
        self._set_joint_angles()
        # 初始化上次打印时间，用于每秒一次的终端输出
        self._last_print_time = time.time()

    def _create_sim(self):
        """创建仿真环境"""
        # 仿真参数
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        # PhysX 参数
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.5
        sim_params.physx.max_depenetration_velocity = 1.0
        sim_params.physx.use_gpu = True
        # 创建sim
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("Failed to create sim")
            sys.exit(1)
        # 创建地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def _load_robot(self):
        """加载机器人URDF"""
        asset_root = os.path.join(os.path.dirname(__file__), 'legged_gym/resources/robots/fall_arm')
        asset_file = "fall_arm.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True  # 固定基座，保持机器人绝对静止
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 1000.0
        asset_options.max_linear_velocity = 1000.0
        asset_options.armature = 0.01
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if self.robot_asset is None:
            print(f"Failed to load asset from {os.path.join(asset_root, asset_file)}")
            sys.exit(1)
        # 获取关节数量和名称
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        # 获取关节名称
        self.dof_names = []
        for i in range(self.num_dof):
            name = self.gym.get_asset_dof_name(self.robot_asset, i)
            self.dof_names.append(name)
        print(f"Robot DOFs ({self.num_dof}): {self.dof_names}")
        # 记录导轨（滑块）关节索引，便于运行时读取位置
        try:
            self.left_shoulder_root_idx = self.dof_names.index('left_shoulder_root_joint')
        except ValueError:
            self.left_shoulder_root_idx = None

    def _create_env(self):
        """创建环境"""
        # 环境参数
        env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
        env_upper = gymapi.Vec3(1.0, 1.0, 2.0)
        # 创建环境
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        # 设置初始位姿（使用前部配置）
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_position)
        start_pose.r = gymapi.Quat(*self.base_orientation)
        # 创建actor
        self.robot_handle = self.gym.create_actor(
            self.env, self.robot_asset, start_pose, "fall_arm", 0, 0
        )
        # 设置关节驱动属性 (高刚度PD控制保持姿态)
        dof_props = self.gym.get_actor_dof_properties(self.env, self.robot_handle)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            if name in self.slider_joint_pos:
                # 滑块关节: 锁定在指定位置 (将上下限设为相同值)
                dof_props['driveMode'][i] = gymapi.DOF_MODE_NONE
                dof_props['stiffness'][i] = 0.0
                dof_props['damping'][i] = 0.0
                dof_props['lower'][i] = self.slider_joint_pos[name]
                dof_props['upper'][i] = self.slider_joint_pos[name]
            else:
                dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                dof_props['stiffness'][i] = 500.0  # 高刚度
                dof_props['damping'][i] = 50.0     # 适当阻尼
        self.gym.set_actor_dof_properties(self.env, self.robot_handle, dof_props)

    def _create_viewer(self):
        """创建可视化窗口"""
        viewer_props = gymapi.CameraProperties()
        viewer_props.width = 1280
        viewer_props.height = 720
        viewer_props.horizontal_fov = 75.0
        self.viewer = self.gym.create_viewer(self.sim, viewer_props)
        if self.viewer is None:
            print("Failed to create viewer")
            sys.exit(1)
        # 设置相机位置
        cam_pos = gymapi.Vec3(1.0, 3.0, 2.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _set_joint_angles(self):
        """设置目标关节角度"""
        # 准备目标位置数组
        dof_targets = np.zeros(self.num_dof, dtype=np.float32)
        print("\n设置关节角度:")
        # 根据关节名称设置目标角度
        for i, name in enumerate(self.dof_names):
            if name in self.target_joint_angles:
                dof_targets[i] = self.target_joint_angles[name]
                if name in self.slider_joint_pos:
                    print(f"  [{i}] {name}: {dof_targets[i]:.3f} m (滑块)")
                else:
                    deg_val = self.target_joint_angles_deg[name]
                    rad_val = self.target_joint_angles[name]
                    print(f"  [{i}] {name}: {deg_val:.1f}° ({rad_val:.3f} rad)")
            else:
                dof_targets[i] = 0.0
                print(f"  [{i}] {name}: 0.0 (未在配置中找到，使用默认值)")
        # 设置目标位置
        self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, dof_targets)
        # 强制设置所有关节的当前状态
        dof_state = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
        for i in range(self.num_dof):
            dof_state['pos'][i] = dof_targets[i]
            dof_state['vel'][i] = 0.0
        self.gym.set_actor_dof_states(self.env, self.robot_handle, dof_state, gymapi.STATE_ALL)

    def run(self):
        """运行仿真循环"""
        print("\n" + "=" * 50)
        print("Fall Arm 姿态测试")
        print("=" * 50)
        print("按 ESC 退出")
        print("按 V 切换相机视角")
        print("按 R 重置机器人姿态")
        print("=" * 50 + "\n")
        while not self.gym.query_viewer_has_closed(self.viewer):
            # 处理键盘事件
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "reset" and evt.value > 0:
                    self._reset_robot()
            # 步进仿真
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            # 更新viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            # 同步
            self.gym.sync_frame_time(self.sim)
            # 每隔 1 秒在终端输出 left_shoulder_root_joint 的位置
            if getattr(self, 'left_shoulder_root_idx', None) is not None:
                now = time.time()
                if now - self._last_print_time >= 1.0:
                    dof_state = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
                    pos = float(dof_state['pos'][self.left_shoulder_root_idx])
                    print(f"left_shoulder_root_joint position: {pos:.4f} m")
                    self._last_print_time = now
        # 清理
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def _reset_robot(self):
        """重置机器人状态"""
        print("重置机器人姿态...")
        # 重置关节角度
        self._set_joint_angles()


def main():
    print("正在启动 Fall Arm 姿态测试...")
    viewer = FallArmPoseViewer()
    viewer.run()


if __name__ == "__main__":
    main()
