from legged_gym.envs.base.base_config import BaseConfig


class FallArmCfg(BaseConfig):
    class init_state:
        pos = [0.0, 0.0, 0.0]          # 基座固定在原点
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

        default_joint_angles = {
            'left_shoulder_root_joint': 0.8,
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.78,
        }

        # 坠落高度随机范围 [min, max] (m), 用于 reset 时随机化 slider_joint 位置
        drop_height_range = [0.8, 1.0]

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
        penalize_contacts_on = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow']  # 非末端接触惩罚
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
        payload_mass_range = [-1, 1]
        # 质心偏移
        randomize_com_displacement = use_random
        com_displacement_range = [-0.03, 0.03]
        # 摩擦系数
        randomize_friction = use_random
        friction_range = [0.1, 1]
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
        slider_vel_limit = 20.0         # [m/s] 导轨速度上限
        slider_pos_min = 0.2           # [m] 导轨位置下限
        soft_dof_pos_limit = 0.9  # 软关节位置限制（安全范围比例）
        soft_dof_vel_limit = 0.9  # 软关节速度限制（安全范围比例）

    class curriculum:
        use_curriculum = True
        force_initial = 100.0               # [N] 初始辅助上升力 (接近完全抵消重力)
        force_decrement = 2.0              # [N] 通过课程后每次减小的力
        force_min = 0.0                     # [N] 最小辅助力 (完全无辅助)
        action_rescale_decrement = 0.01     # 通过课程后每次减小的动作缩放
        action_rescale_min = 0.25           # 最小动作缩放
        min_height_threshold = 0.5         # [m] 回合内 shoulder_root 最低高度须高于此值才通过

    class rewards:
        reward_groups = ['task', 'regu', 'style', 'target']
        num_reward_groups = len(reward_groups)
        reward_group_weights = [1, 0.2, 1, 1]

        arm_pose_not_in_contact_sigma = 0.5
        low_max_slider_acc_threshold = 40
        low_max_slider_acc_margin = 20
        low_max_slider_acc_value_at_margin = 0.01
        high_min_shoulder_root_height_threshold = 0.45
        high_min_shoulder_root_height_margin = 0.05
        high_min_shoulder_root_height_value_at_margin = 0.01

        class scales:
            termination = -1
            task_arm_pose_not_in_contact = 1
            task_low_max_slider_acc = 10
            task_high_min_shoulder_root_height = 1

    class constraints:
        # style reward
        low_max_shoulder_pitch_torque_sigma = 150.0
        low_max_elbow_torque_sigma = 150.0
        arm_roll_yaw_deviation_sigma = 0.1

        # target reward
        arm_pose_at_contact_sigma = 2.0
        low_slider_acc_at_contact_threshold = 40
        low_slider_acc_at_contact_margin = 20
        low_slider_acc_at_contact_value_at_margin = 0.01
        high_shoulder_root_height_at_contact_threshold = 0.55
        high_shoulder_root_height_at_contact_margin = 0.15
        high_shoulder_root_height_at_contact_value_at_margin = 0.01

        class scales:
            # regularization reward
            regu_dof_acc = -2.5e-7
            regu_dof_vel = -1e-3
            regu_action_rate = -5e-2
            regu_smoothness = -1e-1
            regu_torques = -1e-5
            regu_joint_power = -1e-3
            regu_dof_pos_limits = -1
            regu_dof_vel_limits = -5

            # style reward
            style_low_max_shoulder_pitch_torque = 10
            style_low_max_elbow_torque = 10
            style_penalised_contact = -20
            style_arm_roll_yaw_deviation = 10

            # target reward
            target_arm_pose_at_contact = 5
            target_low_slider_acc_at_contact = 5
            target_high_shoulder_root_height_at_contact = 5

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
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = 'fall_arm'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        max_iterations = 12000  # number of policy updates
