"""
将训练好的 FallArm Actor 网络导出为 ONNX 格式，供 CtrlZ 框架部署使用。

用法:
    python export_onnx.py --task fall_arm --load_run <run_name> --checkpoint <step>

示例:
    python export_onnx.py --task fall_arm --load_run Apr14_06-35-15_test_fall_arm --checkpoint 5000
"""
import os
import sys
import copy
import torch
import argparse

# 添加项目路径
LEGGED_GYM_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, LEGGED_GYM_ROOT)

from legged_gym.utils import task_registry, get_args


def export_onnx():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 创建环境以获取观测维度
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 加载训练好的策略
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )

    # 提取 actor 网络
    actor = copy.deepcopy(ppo_runner.alg.actor_critic.actor).to('cpu')
    actor.eval()

    # 获取观测维度
    num_obs = env_cfg.env.num_observations  # 78
    print(f"网络输入维度: {num_obs}")
    print(f"网络输出维度: {env_cfg.env.num_actions}")
    print(f"Actor 结构:\n{actor}")

    # 创建示例输入
    dummy_input = torch.randn(1, num_obs)

    # 测试前向传播
    with torch.no_grad():
        test_output = actor(dummy_input)
    print(f"测试输出 shape: {test_output.shape}")
    print(f"测试输出值: {test_output}")

    # 导出 ONNX
    output_dir = os.path.join(
        LEGGED_GYM_ROOT, 'logs', args.task, args.load_run, 'exported'
    )
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, 'policy.onnx')

    torch.onnx.export(
        actor,
        dummy_input,
        onnx_path,
        input_names=['obs'],
        output_names=['actions'],
        opset_version=11,
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'actions': {0: 'batch_size'},
        },
    )
    print(f"\nONNX 模型已导出到: {onnx_path}")
    print(f"输入节点名: obs")
    print(f"输出节点名: actions")

    # 验证 ONNX 模型
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(onnx_path)
        print(f"\n=== ONNX 模型验证 ===")
        for inp in sess.get_inputs():
            print(f"输入: name={inp.name}, shape={inp.shape}, dtype={inp.type}")
        for out in sess.get_outputs():
            print(f"输出: name={out.name}, shape={out.shape}, dtype={out.type}")

        ort_output = sess.run(None, {'obs': dummy_input.numpy()})
        diff = abs(ort_output[0] - test_output.numpy()).max()
        print(f"PyTorch vs ONNX 最大差异: {diff:.8f}")
        if diff < 1e-5:
            print("✓ 验证通过")
        else:
            print("✗ 差异偏大，请检查")
    except ImportError:
        print("\n未安装 onnxruntime，跳过验证。可用 pip install onnxruntime 安装后手动验证。")


if __name__ == '__main__':
    export_onnx()
