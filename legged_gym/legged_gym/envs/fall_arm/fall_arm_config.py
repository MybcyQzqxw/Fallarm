"""
FallArm 环境配置文件

任务描述:
    一条根部固定在垂直导轨（1自由度平移关节）上的4自由度手臂
    （肩部3自由度 + 肘部1自由度）携带配重从一定高度自由坠落，
    要求小臂末端着地，并训练出缓冲控制策略。

    目标：既不像硬位置控制那样产生极大冲击加速度，
    也不因力矩不足导致肘关节折叠到机械限位。

URDF 关节结构（共5个DOF）:
    - slider_joint:           棱柱关节(prismatic), z轴, 被动（无驱动）
    - shoulder_pitch_joint:   旋转关节, 肩部俯仰
    - shoulder_roll_joint:    旋转关节, 肩部横滚
    - shoulder_yaw_joint:     旋转关节, 肩部偏航
    - elbow_joint:            旋转关节, 肘部弯曲
"""

from legged_gym.envs.base.base_config import BaseConfig


class FallArmCfg(BaseConfig):
    """落臂缓冲任务的环境配置 — 直接继承 BaseConfig，与运动任务彻底解耦"""

    class env:
        num_envs = 4096
        # 观测维度: dof_pos(5) + dof_vel(5) + actions(5) + gravity(3)
        #          + ee_height(1) + ee_vel(3) + time_ratio(1) + contact_phase(1) = 24
        num_observations = 24
        num_privileged_obs = None
        # 动作维度: slider(被动,0增益) + 3肩关节 + 1肘关节 = 5
        num_actions = 5
        episode_length_s = 3.0          # 坠落+缓冲只需约3秒
        env_spacing = 2.0
        send_timeouts = True

    class terrain:
        mesh_type = 'plane'             # 仅需平地
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class init_state:
        pos = [0.0, 0.0, 0.0]          # 基座固定在原点
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

        default_joint_angles = {
            'slider_joint': 1.5,                # 初始高度 1.5m
            'shoulder_pitch_joint': 0.0,        # 肩部俯仰: 中立位
            'shoulder_roll_joint': 0.0,         # 肩部横滚: 中立位
            'shoulder_yaw_joint': 0.0,          # 肩部偏航: 中立位
            'elbow_joint': 0.5,                 # 肘关节: 微弯(准备着地)
        }

        # 坠落高度随机范围 [min, max] (m), 用于 reset 时随机化 slider_joint 位置
        drop_height_range = [1.0, 2.0]

    class control:
        control_type = 'P'  # 位置控制, PD控制器将目标角度转换为力矩
        # slider_joint 不在字典中 → p_gains=0, d_gains=0 → 被动自由落体
        stiffness = {
            'shoulder_pitch': 200.0,
            'shoulder_roll': 200.0,
            'shoulder_yaw': 200.0,
            'elbow': 200.0,
        }
        damping = {
            'shoulder_pitch': 4.0,
            'shoulder_roll': 4.0,
            'shoulder_yaw': 4.0,
            'elbow': 4.0,
        }
        action_scale = 0.5  # target = action * scale + default
        decimation = 4       # 策略频率 = 200Hz / 4 = 50Hz

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/fall_arm/urdf/fall_arm.urdf'
        name = 'fall_arm'
        foot_name = 'end_effector'              # 末端执行器（着地点）
        penalize_contacts_on = ['upper_arm', 'forearm']  # 非末端接触惩罚
        terminate_after_contacts_on = ['slider_link']    # 导轨体触地则终止
        fix_base_link = True            # 导轨基座固定于世界
        disable_gravity = False
        collapse_fixed_joints = False   # 保留完整关节结构
        self_collisions = 1             # 禁用自碰撞 (1=disable)
        flip_visual_attachments = False
        replace_cylinder_with_capsule = True
        default_dof_drive_mode = 3      # effort模式
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]

    class limitation:
        dof_vel_limit = 100.0           # [rad/s] 关节角速度上限
        slider_vel_limit = 20.0         # [m/s] 导轨速度上限

    class rewards:
        only_positive_rewards = False   # 允许负奖励以学习避免不良行为
        soft_dof_pos_limit = 0.9
        max_contact_force = 50.0

        # ---------- 自定义奖励参数 ----------
        cushioning_sigma = 10.0         # 缓冲奖励的指数衰减系数
        deceleration_sigma = 50.0       # 减速奖励的指数衰减系数

        class scales:
            # === 落臂缓冲专用奖励 ===
            termination = -100.0            # 异常终止（非末端着地）重罚
            soft_landing = 5.0              # 低接触力 → 软着陆奖励
            end_effector_contact = 2.0      # 末端触地奖励
            dof_pos_limits = -10.0          # 关节到达机械限位惩罚
            elbow_cushion = 3.0             # 肘关节未折叠奖励（着地时）
            impact_deceleration = 5.0       # 平缓减速奖励
            torques = -0.001                # 力矩过大惩罚
            action_rate = -0.05             # 动作抖动惩罚
            dof_vel = -0.001                # 关节速度过大惩罚
            dof_acc = -1.e-6                # 关节加速度惩罚
            smoothness = -0.02              # 二阶动作平滑惩罚
            collision = -5.0                # 非末端刚体接触惩罚
            arm_extension = 1.5             # 空中阶段手臂前伸准备奖励

    class normalization:
        class obs_scales:
            dof_pos = 1.0
            dof_vel = 0.1
            lin_vel = 2.0
            end_effector_height = 1.0
            end_effector_vel = 0.2
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 0.5  # 适度噪声

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            gravity = 0.05

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [3, -1, 2]               # 相机位置: 侧前方俯视
        lookat = [0., 0., 1.]           # 注视手臂大致高度

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]       # [m/s^2]
        up_axis = 1                      # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1              # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01        # [m]
            rest_offset = 0.0            # [m]
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2       # 0: never, 1: last sub-step, 2: all sub-steps


class FallArmCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 3.e-4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 3000
        save_interval = 100
        experiment_name = 'fall_arm'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
