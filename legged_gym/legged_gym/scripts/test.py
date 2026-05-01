"""
测试脚本: 在仿真环境中并行运行 NUM_TEST_ENVS 个随机化环境，每个环境经历一个完整回合，
计算 6 项评估指标并输出到 txt 文件。

用法 (与 play.py 一致):
    python legged_gym/scripts/test.py --task fall_arm

指标说明:
    (1) 任务成功率 E_succ:       h_min ∈ [H_min_lower, H_min_upper] 且 h_final ∈ [H_final_lower, H_final_upper]
    (2) 运动平滑性 E_smooth:     Σ_t Σ_j |Δ²θ_j/Δt²| / T  [rad/s²]  (mean absolute angular acceleration)
    (3) 能量消耗   E_energy:     Σ_t Σ_j |ω_j(t)| · |τ_j(t)| · Δt  [J]  (HoST convention)
    (4) 最大加速度 a_max:        成功回合中 max_t |a_root(t)| 的均值
    (5) 肩俯仰转矩裕度 m_pitch:  E[1 - max_t |τ_pitch(t)| / τ_pitch^upper]  (成功回合)
    (6) 肘关节转矩裕度 m_elbow:   E[1 - max_t |τ_elbow(t)| / τ_elbow^upper]  (成功回合)
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch

# ====================== 可调参数 ======================
NUM_TEST_ENVS = 1000
# ======================================================


def test(args):
    # ==================== 1. 配置 & 环境创建 ====================
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 覆盖环境数量
    env_cfg.env.num_envs = NUM_TEST_ENVS

    # 保持课程系统开启 (reset_idx 中无条件访问 self.force), 但中性化:
    #   force_initial=0  → 无辅助力
    #   decrement=0      → 不递减
    env_cfg.curriculum.force_initial = 0.0
    env_cfg.curriculum.force_decrement = 0.0
    env_cfg.curriculum.action_rescale_decrement = 0.0

    # 创建环境 (domain randomization 由 env_cfg.domain_rand 控制, 默认开启)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 将 action_rescale 设为课程的最终目标值 (策略训练完成时的工作点)
    env.action_rescale[:] = env_cfg.curriculum.action_rescale_min

    # ==================== 2. 加载训练好的策略 ====================
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # 干净重置: runner.__init__ 中调用了 env.reset() (含一步零动作),
    # 导致 episode_length_buf=1 且 obs_buf 引用变更. 此处显式重置,
    # 保证 episode_length_buf=0, 所有累积量归零, obs 引用正确.
    env.reset_idx(torch.arange(NUM_TEST_ENVS, device=env.device))
    env.compute_observations()
    obs = env.get_observations()

    # ==================== 3. 准备指标收集 buffer ====================
    N = NUM_TEST_ENVS

    # 6 个 N 维向量 (每个环境一个值)
    smoothness_vec = torch.zeros(N, device=env.device)        # (2) 运动平滑性
    energy_vec = torch.zeros(N, device=env.device)             # (3) 能量消耗
    max_acc_vec = torch.zeros(N, device=env.device)           # (4) 最大加速度
    max_pitch_torque_vec = torch.zeros(N, device=env.device)  # (5) 肩俯仰峰值转矩
    max_elbow_torque_vec = torch.zeros(N, device=env.device)  # (6) 肘关节峰值转矩
    min_height_vec = torch.full((N,), float('inf'), device=env.device)
    final_height_vec = torch.zeros(N, device=env.device)

    # 辅助状态
    episode_done = torch.zeros(N, dtype=torch.bool, device=env.device)
    step_count_vec = torch.zeros(N, dtype=torch.float, device=env.device)  # 每环境有效步数
    last_arm_pos = env.dof_pos[:, env.arm_dof_indices].clone()
    last_last_arm_pos = last_arm_pos.clone()

    max_total_steps = int(env.max_episode_length) * 3  # 安全上界
    step_count = 0

    # ==================== 4. 运行仿真 ====================
    # 每个环境只取第一个完整回合的数据
    #
    # env.step() 内部时序:
    #   物理子步循环 → post_physics_step:
    #     shoulder_root_height[:] = dof_pos[:, slider]   ← 设置当前高度 (reset 前)
    #     check_termination() → 标记 reset_buf
    #     reset_idx(env_ids):
    #       _reset_dofs()   → 修改 dof_pos / dof_vel (清零速度, 随机化位置)
    #       min_shoulder_root_height[env_ids] = inf      ← 清零
    #       但 shoulder_root_height 不被清零             ← 可读取正确的回合最终高度
    #     compute_observations()
    #
    # 因此 step() 返回后:
    #   env.shoulder_root_height[i]  对 done 环境 = 回合最终高度 ✓
    #   env.min_shoulder_root_height 对 done 环境 = inf (已清零)
    #   env.dof_pos / env.dof_vel    对 done 环境 = 新回合初始值
    #   env.torques                  对 done 环境 = 最后子步力矩 (未清零) ✓
    #   env.max_shoulder_root_acc_in_one_step = 当前步子步最大值 (未清零) ✓
    #   env.max_shoulder_pitch_torque 对 done 环境 = 0 (已清零)
    #   env.max_elbow_torque         对 done 环境 = 0 (已清零)

    while not episode_done.all() and step_count < max_total_steps:
        # 保存 env 回合级累积量 (下一步可能被 reset 清除)
        pre_step_min_height = env.min_shoulder_root_height.clone()
        pre_step_max_pitch_torque = env.max_shoulder_pitch_torque.clone()
        pre_step_max_elbow_torque = env.max_elbow_torque.clone()

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        step_count += 1

        active = ~episode_done                       # 本步仍在跑的环境
        newly_done = active & dones.bool()           # 本步刚结束的环境
        still_active = active & (~dones.bool())      # 本步仍未结束的环境

        # --- (2) 运动平滑性: Σ_j |Δ²θ_j / Δt²|  [rad/s²] ---
        #     Δ²θ/Δt² 为角加速度的标准离散近似, 取绝对值后单位为 rad/s²
        #     对 done 环境 dof_pos 已是新回合初值, delta 异常, 仅累积 still_active
        curr_arm_pos = env.dof_pos[:, env.arm_dof_indices].clone()
        second_diff = (curr_arm_pos - 2 * last_arm_pos + last_last_arm_pos) / (env.dt ** 2)
        smoothness_vec[still_active] += torch.sum(torch.abs(second_diff[still_active]), dim=1)
        step_count_vec[still_active] += 1
        last_last_arm_pos = last_arm_pos.clone()
        last_arm_pos = curr_arm_pos

        # --- (3) 能量消耗: Σ_j |ω_j|·|τ_j|·dt  [J] ---
        #     机械功率对时间积分, 参照 HoST eval_ground.py
        #     done 环境: reset 后 dof_vel=0, step_energy=0, 最后一步能量不可恢复 (~0.4%)
        arm_torques = env.torques[:, env.arm_dof_indices]
        arm_vel = env.dof_vel[:, env.arm_dof_indices]
        step_energy = torch.sum(torch.abs(arm_vel) * torch.abs(arm_torques), dim=1) * env.dt
        energy_vec[still_active] += step_energy[still_active]

        # --- (4) 最大加速度 (子步精确, 每步开头清零, reset_idx 不清零) ---
        max_acc_vec[active] = torch.maximum(
            max_acc_vec[active],
            env.max_shoulder_root_acc_in_one_step[active]
        )

        # --- (5)(6) 峰值转矩 ---
        #     env.max_shoulder_pitch_torque / max_elbow_torque 是回合级子步精确累积量
        #     still_active: 直接读取 (含当前步所有子步, 单调不减)
        #     newly_done:   reset_idx 已清零, 用 pre_step 快照 + 当前步最后子步值
        max_pitch_torque_vec[still_active] = env.max_shoulder_pitch_torque[still_active]
        max_elbow_torque_vec[still_active] = env.max_elbow_torque[still_active]
        if newly_done.any():
            max_pitch_torque_vec[newly_done] = torch.maximum(
                pre_step_max_pitch_torque[newly_done],
                torch.abs(env.torques[newly_done, env.shoulder_pitch_dof_idx])
            )
            max_elbow_torque_vec[newly_done] = torch.maximum(
                pre_step_max_elbow_torque[newly_done],
                torch.abs(env.torques[newly_done, env.elbow_dof_idx])
            )

        # --- 最低高度 ---
        # still_active: 使用 env 的子步精确累积值
        min_height_vec[still_active] = env.min_shoulder_root_height[still_active]
        # newly_done: env 已将 min_shoulder_root_height 清为 inf
        #   用 pre_step 快照 + 本步末高度取 min (本步子步内高度基本单调, 误差极小)
        if newly_done.any():
            min_height_vec[newly_done] = torch.minimum(
                pre_step_min_height[newly_done],
                env.shoulder_root_height[newly_done]
            )

        # --- 最终高度 ---
        # shoulder_root_height 在 post_physics_step 中于 reset 前设置, 不被 reset 清除
        if newly_done.any():
            final_height_vec[newly_done] = env.shoulder_root_height[newly_done]

        episode_done |= dones.bool()

    # 处理超时未结束的环境 (理论上不应发生)
    if not episode_done.all():
        still_running = ~episode_done
        final_height_vec[still_running] = env.shoulder_root_height[still_running]
        min_height_vec[still_running] = env.min_shoulder_root_height[still_running]
        max_pitch_torque_vec[still_running] = env.max_shoulder_pitch_torque[still_running]
        max_elbow_torque_vec[still_running] = env.max_elbow_torque[still_running]
        episode_done[:] = True
        print(f"WARNING: {still_running.sum().item()} envs did not finish within {max_total_steps} steps")

    # ==================== 5. 计算 6 项指标 ====================

    # (1) 任务成功率
    cfg_cur = env_cfg.curriculum
    success_vec = (
        (min_height_vec > cfg_cur.min_shoulder_root_height_lower_threshold) & (min_height_vec < cfg_cur.min_shoulder_root_height_upper_threshold) & (final_height_vec > cfg_cur.final_shoulder_root_height_lower_threshold) & (final_height_vec < cfg_cur.final_shoulder_root_height_upper_threshold)
    ).float()  # (N,) 0.0/1.0
    E_succ = success_vec.mean().item()
    num_success = int(success_vec.sum().item())

    # (2) 运动平滑性: 除以步数得到每步平均, 参照 HoST (所有环境均值)
    #     单位: rad/s²
    smoothness_vec /= step_count_vec.clamp(min=1)
    E_smooth = smoothness_vec.mean().item()

    # (3) 能量消耗: 累积总功, 单位 J (所有环境均值)
    E_energy = energy_vec.mean().item()

    # (4)~(6) 仅在成功回合上统计
    success_mask = success_vec.bool()
    if num_success > 0:
        # (4) 最大加速度
        a_max = max_acc_vec[success_mask].mean().item()

        # 转矩上限 (torque_limits 按 arm_dof_indices 索引)
        sp_arm_idx = env.arm_dof_indices.index(env.shoulder_pitch_dof_idx)
        el_arm_idx = env.arm_dof_indices.index(env.elbow_dof_idx)
        tau_pitch_upper = env.torque_limits[sp_arm_idx].item()
        tau_elbow_upper = env.torque_limits[el_arm_idx].item()

        # (5) 肩俯仰关节转矩裕度 (Normalized Torque Margin)
        m_pitch = (1.0 - max_pitch_torque_vec[success_mask] / tau_pitch_upper).mean().item()

        # (6) 肘关节转矩裕度
        m_elbow = (1.0 - max_elbow_torque_vec[success_mask] / tau_elbow_upper).mean().item()
    else:
        a_max = float('nan')
        m_pitch = float('nan')
        m_elbow = float('nan')
        tau_pitch_upper = float('nan')
        tau_elbow_upper = float('nan')

    # ==================== 6. 输出 ====================
    results_str = (
        f"==================== 测试结果 ({N} 环境, {step_count} 步) ====================\n"
        f"(1) 任务成功率         E_succ    = {E_succ:.4f}  ({num_success}/{N})\n"
        f"(2) 运动平滑性         E_smooth  = {E_smooth:.6f}  [rad/s²]\n"
        f"(3) 能量消耗           E_energy  = {E_energy:.4f}  [J]\n"
        f"(4) 最大加速度 (成功)  a_max     = {a_max:.4f}\n"
        f"(5) 肩俯仰转矩裕度    m_pitch   = {m_pitch:.4f}\n"
        f"(6) 肘关节转矩裕度     m_elbow   = {m_elbow:.4f}\n"
        f"====================================================================\n"
        f"\n成功判定阈值 (与课程推进准则一致):\n"
        f"  h_min   ∈ [{cfg_cur.min_shoulder_root_height_lower_threshold}, "
        f"{cfg_cur.min_shoulder_root_height_upper_threshold}]\n"
        f"  h_final ∈ [{cfg_cur.final_shoulder_root_height_lower_threshold}, "
        f"{cfg_cur.final_shoulder_root_height_upper_threshold}]\n"
    )
    if num_success > 0:
        results_str += (
            f"\n转矩上限:\n"
            f"  tau_pitch_upper = {tau_pitch_upper:.2f} N·m\n"
            f"  tau_elbow_upper = {tau_elbow_upper:.2f} N·m\n"
        )

    print(results_str)

    # 保存结果
    log_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, 'logs', args.task, 'test_results'
    )
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 写入 txt 摘要
    txt_path = os.path.join(log_dir, f'test_{timestamp}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(results_str)

    # 保存 6 个 N 维向量到 .pt 文件
    vectors = {
        'success': success_vec.cpu(),
        'smoothness': smoothness_vec.cpu(),
        'energy': energy_vec.cpu(),
        'max_acc': max_acc_vec.cpu(),
        'max_pitch_torque': max_pitch_torque_vec.cpu(),
        'max_elbow_torque': max_elbow_torque_vec.cpu(),
        'min_height': min_height_vec.cpu(),
        'final_height': final_height_vec.cpu(),
    }
    pt_path = os.path.join(log_dir, f'test_{timestamp}.pt')
    torch.save(vectors, pt_path)

    print(f"摘要已保存到: {txt_path}")
    print(f"向量已保存到: {pt_path}")


if __name__ == '__main__':
    args = get_args()
    test(args)
