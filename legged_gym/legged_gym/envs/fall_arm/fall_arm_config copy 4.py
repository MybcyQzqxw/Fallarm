from legged_gym.envs.base.base_config import BaseConfig


class FallArmCfg(BaseConfig):
    class init_state:
        pos = [0.0, 0.0, 0.0]          # 基座固定在原点
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

        default_joint_angles = {
            'left_shoulder_root_joint': 0.95,
            'left_shoulder_pitch_joint': 0.55,  # 31.6 deg
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.234,  # 70.7 deg
        }

        # 坠落高度随机范围 [min, max] (m), 用于 reset 时随机化 slider_joint 位置
        drop_height_range = [0.95, 1.0]

    class env:
        num_envs = 1024
        num_dofs = 5
        num_real_dofs = 4
        num_actions = 4
        # 观测维度: arm_dof_pos(4) + arm_dof_vel(4) + arm_torques(4)
        #          + action_rescale(1) = 13
        num_one_step_observations = 13
        num_actor_history = 6  # 历史观测步数
        num_observations = num_actor_history * num_one_step_observations
        episode_length_s = 5.0

        num_privileged_obs = None
        env_spacing = 3.0
        send_timeouts = True  # send time out information to the algorithm

    class control:
        control_type = 'P'  # 位置控制, PD控制器将目标角度转换为力矩
        stiffness = {
            'shoulder_pitch': 100.0,
            'shoulder_roll': 100.0,
            'shoulder_yaw': 100.0,
            'elbow': 100.0,
        }
        damping = {
            'shoulder_pitch': 1.0,
            'shoulder_roll': 1.0,
            'shoulder_yaw': 1.0,
            'elbow': 1.0,
        }
        action_scale = 1.0  # target = action * scale + default
        decimation = 4       # 策略频率 = 200Hz / 4 = 50Hz

        # 导轨滑块摩擦参数 (无策略控制, 仅物理模拟)
        slider_viscous_friction = 1.0   # [N·s/m] 粘性摩擦系数 (与速度成正比)
        slider_coulomb_friction = 0.1   # [N]     库仑摩擦力 (恒定干摩擦)

    class terrain:
        mesh_type = 'plane'
        static_friction = 0.8   # 静摩擦系数
        dynamic_friction = 0.7  # 动摩擦系数
        restitution = 0.3       # 恢复系数（0=完全非弹性，1=完全弹性碰撞）

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/fall_arm/fall_arm.urdf'
        name = 'fall_arm'

        # 惩罚和终止条件
        penalize_contacts_on = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw']
        terminate_after_contacts_on = ['shoulder_root']

        base_name = 'base'
        shoulder_root_name = 'shoulder_root'
        shoulder_pitch_name = 'shoulder_pitch'
        shoulder_roll_name = 'shoulder_roll'
        shoulder_yaw_name = 'shoulder_yaw'
        elbow_name = 'elbow'
        end_name = 'end'

        shoulder_root_joint = ['shoulder_root_joint']
        shoulder_pitch_joint = ['shoulder_pitch_joint']
        shoulder_roll_joint = ['shoulder_roll_joint']
        shoulder_yaw_joint = ['shoulder_yaw_joint']
        elbow_joint = ['elbow_joint']

        default_dof_drive_mode = 3  # 关节控制模式 (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        collapse_fixed_joints = True  # 合并被固定关节连接的连杆，给关节增加 <... dont_collapse="true"> 配置来不被合并
        replace_cylinder_with_capsule = True  # 替换碰撞圆柱体为胶囊体，提升仿真速度和稳定性
        flip_visual_attachments = False  # 一些 .obj 网格必须从 y-up 翻转到 z-up
        fix_base_link = True  # 固定机器人的基座
        density = 0.001         # 密度 [kg/m^3]
        angular_damping = 0.01  # 角阻尼
        linear_damping = 0.01   # 线阻尼
        max_angular_velocity = 1000.0  # 最大角速度 [rad/s]
        max_linear_velocity = 1000.0   # 最大线速度 [m/s]
        armature = 0.01       # 关节惯量补偿 [kg*m^2]
        thickness = 0.01      # 碰撞检测厚度 [m]
        disable_gravity = False
        self_collisions = 0   # 0：启用自碰撞，1：禁用自碰撞（可穿透）

    class domain_rand:
        use_random = True

        # _create_envs 中初始化下面 5 个
        # 负载质量
        randomize_payload_mass = use_random
        payload_mass_range = [-9.5, -5.5]
        # 质心偏移
        randomize_com_displacement = use_random
        com_displacement_range = [-0.05, 0.05]
        # 摩擦系数
        randomize_friction = use_random
        friction_range = [0.1, 1.0]
        # 恢复系数
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        # 连杆质量
        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]

        # _init_buffers 中初始化下面 4 个
        # kp
        randomize_kp = use_random
        kp_range = [0.85, 1.15]
        # kd
        randomize_kd = use_random
        kd_range = [0.85, 1.15]
        # 驱动偏置
        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]
        # 电机力矩
        randomize_motor_strength = use_random
        motor_strength_range = [0.9, 1.1]

        # 初始关节角随机化
        randomize_initial_joint_pos = use_random
        initial_joint_pos_scale = [0.9, 1.1]
        initial_joint_pos_offset = [-0.1, 0.1]

        delay = use_random
        max_delay_timesteps = 5

    class limitation:
        dof_vel_limit = 300.0           # [rad/s] 关节角速度上限
        slider_vel_limit = 100.0         # [m/s] 导轨速度上限
        slider_pos_min = 0.25           # [m] 导轨位置下限
        soft_dof_pos_limit = 0.9  # 软关节位置限制（安全范围比例）
        soft_dof_vel_limit = 0.9  # 软关节速度限制（安全范围比例）

    class curriculum:
        use_curriculum = True
        force_initial = 100.0               # [N] 初始辅助上升力 (接近完全抵消重力)
        force_decrement = 4.0              # [N] 通过课程后每次减小的力
        force_min = 0.0                     # [N] 最小辅助力 (完全无辅助)
        action_rescale_decrement = 0.01     # 通过课程后每次减小的动作缩放
        action_rescale_min = 0.6           # 最小动作缩放
        check_min_shoulder_root_height = True                   # 是否开启对回合内最低高度的判断
        # check_min_shoulder_root_height = False                   # 是否开启对回合内最低高度的判断
        min_shoulder_root_height_lower_threshold = 0.30         # [m] 回合内 shoulder_root 最低高度须高于此值才通过
        min_shoulder_root_height_upper_threshold = 0.40         # [m] 回合内 shoulder_root 最低高度须低于此值才通过
        check_final_shoulder_root_height = True                  # 是否开启对回合结束时高度的判断
        # check_final_shoulder_root_height = False                  # 是否开启对回合结束时高度的判断
        final_shoulder_root_height_lower_threshold = 0.52        # [m] 回合结束时 shoulder_root 高度须高于此值才通过
        final_shoulder_root_height_upper_threshold = 0.58        # [m] 回合结束时 shoulder_root 高度须低于此值才通过

    class rewards:
        reward_groups = ['task', 'regularization', 'behavior', 'effort', 'stabilization']
        num_reward_groups = len(reward_groups)
        reward_group_weights = [1.5, 0.1, 0.5, 0.02, 1]

        class scales:
            termination = -1
            task_arm_pose = 1
            task_all_dof_pos = 1

    class constraints:
        land_height = 0.55

        # task reward
        # arm_pose_threshold = 0.1
        # arm_pose_margin = 2.0
        # arm_pose_value_at_margin = 0.1
        arm_pose_sigma = 0.5

        # behavior reward
        no_releave_after_contact_threshold = 3  # [frames] 从接触开始的无接触候选必须持续至少这个帧数才被惩罚
        shoulder_pitch_dof_pos_threshold = 0.45
        elbow_dof_pos_threshold = 1.15
        ee_distance_threshold = 0.05
        ee_distance_margin = 0.50
        ee_distance_value_at_margin = 0.05
        high_shoulder_root_height_threshold = 0.35
        low_shoulder_root_height_threshold = 0.55
        # min_shoulder_root_height_range_lower_threshold = 0.30
        # min_shoulder_root_height_range_upper_threshold = 0.40
        # min_shoulder_root_height_range_margin = 0.05
        # min_shoulder_root_height_range_value_at_margin = 0.1

        # effort reward
        low_max_shoulder_pitch_torque_sigma = 150
        low_max_elbow_torque_sigma = 150
        low_max_shoulder_root_acc_threshold = 100
        low_max_shoulder_root_acc_margin = 200
        low_max_shoulder_root_acc_value_at_margin = 0.05

        # stabilization reward
        target_shoulder_root_height_threshold = 0.55
        target_shoulder_root_height_sigma = 0.1

        class scales:
            # regularization reward
            regularization_dof_acc = -2.5e-7
            regularization_dof_vel = -1e-3
            regularization_action_rate = -1e-2
            regularization_action_jerk = -1e-2
            regularization_torques = -2.5e-6
            regularization_joint_power = -2.5e-5
            regularization_dof_pos_limits = -1e2
            regularization_dof_vel_limits = -1e0

            # behavior reward
            behavior_penalised_contact = -50
            behavior_encourage_contact = 10
            behavior_no_releave_after_contact = -10
            behavior_shoulder_pitch_dof_pos = -50
            behavior_shoulder_roll_dof_pos = -50
            behavior_shoulder_yaw_dof_pos = -50
            behavior_elbow_dof_pos = -50
            behavior_ee_distance = 10
            behavior_ee_vel = -10
            behavior_high_shoulder_root_height = 50
            behavior_low_shoulder_root_height = -50
            # effort reward
            effort_low_max_shoulder_pitch_torque = 10
            effort_low_max_elbow_torque = 10
            effort_low_max_shoulder_root_acc = 10

            # stabilization reward
            stabilization_target_ee_vel = -50
            stabilization_target_shoulder_root_height = 10

    class normalization:
        clip_observations = 100.0
        clip_actions = 100.0

        class obs_scales:
            dof_pos = 1.0
            dof_vel = 0.05

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5

    class viewer:
        ref_env = 0
        pos = [5.0, 10.0, 2.0]
        lookat = [0.0, 0.0, 0.0]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]       # [m/s^2]
        up_axis = 1                      # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class FallArmCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256]
        activation = 'elu'

    class algorithm:
        learning_rate = 3.e-4
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        # smoothness
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0.1

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        init_at_random_ep_len = True
        num_steps_per_env = 50  # per iteration
        # logging
        save_interval = 2000  # check for potential saves every this many iterations
        experiment_name = 'fall_arm'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        max_iterations = 8000  # number of policy updates
