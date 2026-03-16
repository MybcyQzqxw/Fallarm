#!/usr/bin/env python
"""
URDF转换工具 - 用于将通用URDF文件转换为适合Isaac Gym的格式

主要功能:
1. 将所有 continuous 关节转换为 revolute 关节并添加限制
2. 移除 package:// 前缀的 mesh 路径
3. 为固定关节(如 head_joint)添加 dont_collapse="true" 属性
4. 修改关节 origin 位置

使用方法:
    python fix_urdf.py <input_urdf> <output_urdf> [options]

    或直接修改脚本中的路径后运行:
    python fix_urdf.py
"""
import math
import re
import argparse
import sys
from pathlib import Path

# ==================== 文件路径配置 ====================
# 直接在这里修改输入输出路径，然后直接运行脚本即可
INPUT_URDF = "legged_gym/resources/robots/fall_arm/fall_arm.urdf"
OUTPUT_URDF = "legged_gym/resources/robots/fall_arm/fall_arm.urdf"

# ==================== 关节限制配置 ====================
# 定义各类关节的标准限制（角度制输入，程序自动转换为弧度）
# 格式: 'joint_keyword': (lower_limit_deg, upper_limit_deg, effort_Nm, velocity_rad/s)
# 注意：角度用度数(°)，扭矩用牛顿·米(N·m)，速度用弧度/秒(rad/s)


DEFAULT_JOINT_LIMITS = {
    # fall_arm 手臂关节
    'left_shoulder_pitch': (-90, 90, 120, 20),
    'left_shoulder_roll': (-10, 90, 120, 20),
    'left_shoulder_yaw': (-90, 45, 120, 20),
    'left_elbow': (10, 115, 120, 20),
}

# ==================== 移动关节 (prismatic) 限制配置 ====================
# 格式: 'joint_keyword': (lower_m, upper_m, effort_N, velocity_m/s)
# 注意: 单位直接为米/牛顿/米每秒, 不做角度转换
PRISMATIC_JOINT_LIMITS = {
    'left_shoulder_root': (0, 1, 0, 0),
}

# ==================== 关节 Origin 配置 ====================
# 定义各个关节的 origin xyz 和 rpy
# 格式: 'joint_name': {'xyz': 'x y z', 'rpy': 'r p y'}
# 只需填写要修改的关节, 未列出的关节保持不变
# xyz/rpy 也可以只填其中一个, 另一个保持原值
JOINT_ORIGINS = {
    'left_shoulder_root_joint': {'xyz': '0 0.114 0'},
}

# ==================== 核心转换函数 ====================


def check_if_already_processed(content, custom_limits=None, custom_origins=None, custom_prismatic_limits=None, verbose=False):
    # 检测 URDF 文件是否已经被处理过
    # Args:
    #     content: URDF 文件内容
    #     custom_limits: 自定义关节限制字典
    #     verbose: 是否打印详细信息
    # Returns:
    #     (is_processed, issues) - is_processed 为 True 表示已处理，issues 为问题列表
    issues = []
    # 1. 检查是否有 continuous 关节
    if re.search(r'type="continuous"', content):
        issues.append('存在 continuous 类型的关节')
    # 2. 检查是否有 package:// 前缀
    if re.search(r'filename="package://', content):
        issues.append('存在 package:// 前缀的 mesh 路径')
    # 3. 检查是否有 ../meshes/ 路径
    if re.search(r'meshdir="\.\./meshes/"', content):
        issues.append('存在 ../meshes/ 路径需要修改为 meshes/')
    # 4. 检查所有 fixed 关节是否都有 dont_collapse
    fixed_joints = re.findall(r'<joint name="([^"]+)"[^>]*type="fixed"[^>]*>', content)
    fixed_without_dont_collapse = []
    for joint_name in fixed_joints:
        # 查找这个关节的完整标签
        pattern = f'<joint name="{re.escape(joint_name)}"[^>]*type="fixed"[^>]*>'
        match = re.search(pattern, content)
        if match and 'dont_collapse' not in match.group(0):
            fixed_without_dont_collapse.append(joint_name)
    if fixed_without_dont_collapse:
        issues.append(f'{len(fixed_without_dont_collapse)} 个固定关节缺少 dont_collapse 属性')
    # 5.5. 检查关节 origin 是否与配置一致
    joint_origins = custom_origins if custom_origins else JOINT_ORIGINS
    if joint_origins:
        origin_mismatched = []
        for joint_name, expected_origin in joint_origins.items():
            pattern = rf'<joint name="{re.escape(joint_name)}"[^>]*>.*?<origin\s+([^/]*)/?>'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                origin_attrs = match.group(1)
                if 'xyz' in expected_origin:
                    xyz_match = re.search(r'xyz="([^"]+)"', origin_attrs)
                    if xyz_match and xyz_match.group(1).strip() != expected_origin['xyz'].strip():
                        origin_mismatched.append(f"{joint_name}: xyz 实际'{xyz_match.group(1)}' != 期望'{expected_origin['xyz']}'")
                if 'rpy' in expected_origin:
                    rpy_match = re.search(r'rpy="([^"]+)"', origin_attrs)
                    if rpy_match and rpy_match.group(1).strip() != expected_origin['rpy'].strip():
                        origin_mismatched.append(f"{joint_name}: rpy 实际'{rpy_match.group(1)}' != 期望'{expected_origin['rpy']}'")
        if origin_mismatched:
            issues.append(f'{len(origin_mismatched)} 个关节的 origin 与配置不一致')
            if verbose:
                for m in origin_mismatched:
                    issues.append(f'  └─ {m}')
    # 5. 检查 revolute 关节的限位是否与配置一致
    revolute_pattern = r'<joint name="([^"]+)"[^>]*type="revolute"[^>]*>.*?<limit\s+lower="([^"]+)"\s+upper="([^"]+)"\s+effort="([^"]+)"\s+velocity="([^"]+)"[^>]*/>.*?</joint>'
    revolute_joints = re.findall(revolute_pattern, content, re.DOTALL)
    mismatched_joints = []
    for joint_name, lower, upper, effort, velocity in revolute_joints:
        # 查找期望的限制值
        expected_limit = find_joint_limit(joint_name, custom_limits)
        if expected_limit:
            exp_lower, exp_upper, exp_effort, exp_velocity = expected_limit
            # 转换为字符串进行比较（允许小的浮点误差）
            actual = (float(lower), float(upper), float(effort), float(velocity))
            expected = (float(exp_lower), float(exp_upper), float(exp_effort), float(exp_velocity))
            # 比较值（允许0.01的误差）
            if not all(abs(a - e) < 0.01 for a, e in zip(actual, expected)):
                mismatched_joints.append(f'{joint_name}: 实际{actual} != 期望{expected}')

    if mismatched_joints:
        issues.append(f'{len(mismatched_joints)} 个关节的限位与配置不一致')
        if verbose:
            for mismatch in mismatched_joints:
                issues.append(f'  └─ {mismatch}')
    # 6. 检查 prismatic 关节的限位是否与配置一致
    prismatic_limits = custom_prismatic_limits if custom_prismatic_limits else PRISMATIC_JOINT_LIMITS
    prismatic_pattern = r'<joint name="([^"]+)"[^>]*type="prismatic"[^>]*>.*?<limit\s+lower="([^"]+)"\s+upper="([^"]+)"\s+effort="([^"]+)"\s+velocity="([^"]+)"[^>]*/>.*?</joint>'
    prismatic_joints = re.findall(prismatic_pattern, content, re.DOTALL)
    prismatic_mismatched = []
    for joint_name, lower, upper, effort, velocity in prismatic_joints:
        expected = find_prismatic_joint_limit(joint_name, prismatic_limits)
        if expected:
            exp_lower, exp_upper, exp_effort, exp_velocity = expected
            actual = (float(lower), float(upper), float(effort), float(velocity))
            exp = (float(exp_lower), float(exp_upper), float(exp_effort), float(exp_velocity))
            if not all(abs(a - e) < 0.01 for a, e in zip(actual, exp)):
                prismatic_mismatched.append(f'{joint_name}: 实际{actual} != 期望{exp}')
    if prismatic_mismatched:
        issues.append(f'{len(prismatic_mismatched)} 个移动关节的限位与配置不一致')
        if verbose:
            for m in prismatic_mismatched:
                issues.append(f'  └─ {m}')
    is_processed = len(issues) == 0
    if verbose:
        if is_processed:
            print("\n✅ 文件已经被处理过，满足所有转换要求")
        else:
            print("\n⚠️  检测到文件需要处理：")
            for issue in issues:
                print(f"  - {issue}")
    return is_processed, issues


def find_prismatic_joint_limit(joint_name, custom_limits=None):
    """根据关节名称查找 prismatic 关节的限制参数 (不做角度转换)"""
    limits = custom_limits if custom_limits else PRISMATIC_JOINT_LIMITS
    limit_tuple = None
    if joint_name in limits:
        limit_tuple = limits[joint_name]
    else:
        for key, limit in limits.items():
            if key in joint_name:
                limit_tuple = limit
                break
    return limit_tuple


def find_joint_limit(joint_name, custom_limits=None):
    # 根据关节名称查找对应的限制参数
    # Args:
    #     joint_name: 关节名称
    #     custom_limits: 自定义限制字典（可选）
    # Returns:
    #     (lower_rad, upper_rad, effort, velocity) 或 None
    #     注意：返回的角度已转换为弧度制
    limits = custom_limits if custom_limits else DEFAULT_JOINT_LIMITS
    limit_tuple = None
    # 优先精确匹配
    if joint_name in limits:
        limit_tuple = limits[joint_name]
    else:
        # 然后关键字匹配
        for key, limit in limits.items():
            if key in joint_name:
                limit_tuple = limit
                break
    if limit_tuple:
        # 将角度从度数转换为弧度，effort 和 velocity 保持不变
        lower_deg, upper_deg, effort, velocity = limit_tuple
        lower_rad = math.radians(lower_deg)
        upper_rad = math.radians(upper_deg)
        return (lower_rad, upper_rad, effort, velocity)
    return None


def convert_continuous_to_revolute(content, custom_limits=None, verbose=False):
    # 将所有 continuous 关节转换为 revolute 关节并添加限制
    # Args:
    #     content: URDF 文件内容
    #     custom_limits: 自定义关节限制字典
    #     verbose: 是否打印详细信息
    # Returns:
    #     转换后的内容和统计信息
    converted_count = 0
    skipped_joints = []

    def replace_joint(match):
        nonlocal converted_count, skipped_joints
        joint_name = match.group(1)
        joint_block = match.group(0)
        # 查找对应的限制
        limit = find_joint_limit(joint_name, custom_limits)
        if limit:
            lower, upper, effort, velocity = limit
            # 替换 type 为 revolute
            joint_block = joint_block.replace('type="continuous"', 'type="revolute"')
            # 在 </joint> 前添加 limit
            limit_tag = f'  <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="{velocity}"/>\n  '
            joint_block = joint_block.replace('</joint>', f'{limit_tag}</joint>')
            converted_count += 1
            if verbose:
                print(f"  ✓ 转换关节: {joint_name} -> [{lower}, {upper}]")
        else:
            skipped_joints.append(joint_name)
            if verbose:
                print(f"  ⚠ 跳过关节: {joint_name} (未找到匹配的限制)")
        return joint_block
    # 匹配所有 continuous 关节
    pattern = r'<joint name="([^"]+)"[^>]*type="continuous"[^>]*>.*?</joint>'
    new_content = re.sub(pattern, replace_joint, content, flags=re.DOTALL)
    return new_content, {'converted': converted_count, 'skipped': skipped_joints}


def fix_revolute_joint_limits(content, custom_limits=None, verbose=False):
    #
    # 修正已存在的 revolute 关节的限位值，确保与配置一致
    # Args:
    #     content: URDF 文件内容
    #     custom_limits: 自定义关节限制字典
    #     verbose: 是否打印详细信息
    # Returns:
    #     转换后的内容和统计信息
    #
    fixed_count = 0
    skipped_joints = []

    def fix_limit(match):
        nonlocal fixed_count, skipped_joints
        joint_name = match.group(1)
        joint_block = match.group(0)
        current_lower = match.group(2)
        current_upper = match.group(3)
        current_effort = match.group(4)
        current_velocity = match.group(5)
        # 查找期望的限制值
        expected_limit = find_joint_limit(joint_name, custom_limits)
        if expected_limit:
            exp_lower, exp_upper, exp_effort, exp_velocity = expected_limit
            # 检查是否需要修改
            actual = (float(current_lower), float(current_upper), float(current_effort), float(current_velocity))
            expected = (float(exp_lower), float(exp_upper), float(exp_effort), float(exp_velocity))
            if not all(abs(a - e) < 0.01 for a, e in zip(actual, expected)):
                # 需要修改，替换 limit 标签
                old_limit = f'<limit lower="{current_lower}" upper="{current_upper}" effort="{current_effort}" velocity="{current_velocity}"'
                new_limit = f'<limit lower="{exp_lower}" upper="{exp_upper}" effort="{exp_effort}" velocity="{exp_velocity}"'
                joint_block = joint_block.replace(old_limit, new_limit)
                fixed_count += 1
                if verbose:
                    print(f"  ✓ 修正关节限位: {joint_name} -> [{exp_lower}, {exp_upper}]")
        else:
            skipped_joints.append(joint_name)
        return joint_block
    # 匹配所有 revolute 关节及其 limit
    pattern = r'<joint name="([^"]+)"[^>]*type="revolute"[^>]*>.*?<limit\s+lower="([^"]+)"\s+upper="([^"]+)"\s+effort="([^"]+)"\s+velocity="([^"]+)"[^>]*/?>.*?</joint>'
    new_content = re.sub(pattern, fix_limit, content, flags=re.DOTALL)
    return new_content, {'fixed': fixed_count, 'skipped': skipped_joints}


def fix_prismatic_joint_limits(content, custom_prismatic_limits=None, verbose=False):
    """
    修正已存在的 prismatic 关节的限位值
    单位直接为米, 不做角度转换
    """
    limits = custom_prismatic_limits if custom_prismatic_limits else PRISMATIC_JOINT_LIMITS
    fixed_count = 0
    skipped_joints = []

    def fix_limit(match):
        nonlocal fixed_count, skipped_joints
        joint_name = match.group(1)
        joint_block = match.group(0)
        current_lower = match.group(2)
        current_upper = match.group(3)
        current_effort = match.group(4)
        current_velocity = match.group(5)
        expected = find_prismatic_joint_limit(joint_name, limits)
        if expected:
            exp_lower, exp_upper, exp_effort, exp_velocity = expected
            actual = (float(current_lower), float(current_upper), float(current_effort), float(current_velocity))
            exp = (float(exp_lower), float(exp_upper), float(exp_effort), float(exp_velocity))
            if not all(abs(a - e) < 0.01 for a, e in zip(actual, exp)):
                old_limit = f'<limit lower="{current_lower}" upper="{current_upper}" effort="{current_effort}" velocity="{current_velocity}"'
                new_limit = f'<limit lower="{exp_lower}" upper="{exp_upper}" effort="{exp_effort}" velocity="{exp_velocity}"'
                joint_block = joint_block.replace(old_limit, new_limit)
                fixed_count += 1
                if verbose:
                    print(f"  ✓ 修正移动关节限位: {joint_name} -> [{exp_lower}, {exp_upper}] m")
        else:
            skipped_joints.append(joint_name)
        return joint_block

    pattern = r'<joint name="([^"]+)"[^>]*type="prismatic"[^>]*>.*?<limit\s+lower="([^"]+)"\s+upper="([^"]+)"\s+effort="([^"]+)"\s+velocity="([^"]+)"[^>]*/?>.*?</joint>'
    new_content = re.sub(pattern, fix_limit, content, flags=re.DOTALL)
    return new_content, {'fixed': fixed_count, 'skipped': skipped_joints}


def remove_package_prefix(content, verbose=False):
    # 移除 mesh 文件路径中的 package:// 前缀
    # Args:
    #     content: URDF 文件内容
    #     verbose: 是否打印详细信息
    # Returns:
    #     转换后的内容和替换次数
    # 匹配 filename="package://PACKAGE_NAME/path"
    pattern = r'filename="package://[^/]+/'
    replacement = 'filename="'
    new_content, count = re.subn(pattern, replacement, content)
    if verbose and count > 0:
        print(f"  ✓ 移除了 {count} 个 package:// 前缀")
    return new_content, count


def fix_mujoco_meshdir(content, verbose=False):
    #
    # 修改 mujoco 标签中的 meshdir 路径，将 ../meshes/ 改为 meshes/
    # Args:
    #     content: URDF 文件内容
    #     verbose: 是否打印详细信息
    # Returns:
    #     转换后的内容和替换次数
    #
    # 匹配 meshdir="../meshes/" 并替换为 meshdir="meshes/"
    pattern = r'meshdir="\.\./meshes/"'
    replacement = 'meshdir="meshes"'
    new_content, count = re.subn(pattern, replacement, content)
    if verbose and count > 0:
        print(f"  ✓ 修改了 {count} 个 mujoco meshdir 路径")
    return new_content, count


def fix_joint_origins(content, custom_origins=None, verbose=False):
    """
    修改指定关节的 origin xyz/rpy 值
    Args:
        content: URDF 文件内容
        custom_origins: 自定义 origin 字典
        verbose: 是否打印详细信息
    Returns:
        转换后的内容和修改数量
    """
    origins = custom_origins if custom_origins else JOINT_ORIGINS
    modified_count = 0

    for joint_name, new_origin in origins.items():
        # 匹配关节块中的 origin 标签
        pattern = rf'(<joint name="{re.escape(joint_name)}"[^>]*>\s*<origin\s+)([^/]*)(/?>)'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            if verbose:
                print(f"  ⚠ 未找到关节: {joint_name}")
            continue

        old_attrs = match.group(2)
        new_attrs = old_attrs

        if 'xyz' in new_origin:
            new_attrs = re.sub(r'xyz="[^"]+"', f'xyz="{new_origin["xyz"]}"', new_attrs)
        if 'rpy' in new_origin:
            new_attrs = re.sub(r'rpy="[^"]+"', f'rpy="{new_origin["rpy"]}"', new_attrs)

        if new_attrs != old_attrs:
            content = content[:match.start(2)] + new_attrs + content[match.end(2):]
            modified_count += 1
            if verbose:
                print(f"  ✓ 修改关节 origin: {joint_name}")
                if 'xyz' in new_origin:
                    print(f"      xyz -> {new_origin['xyz']}")
                if 'rpy' in new_origin:
                    print(f"      rpy -> {new_origin['rpy']}")
        else:
            if verbose:
                print(f"  - 关节 origin 已经是目标值: {joint_name}")

    return content, modified_count


def add_dont_collapse_to_fixed_joints(content, verbose=False):
    #
    # 为所有固定关节自动添加 dont_collapse="true" 属性
    # Args:
    #     content: URDF 文件内容
    #     verbose: 是否打印详细信息
    # Returns:
    #     转换后的内容和修改次数
    #
    modified_count = 0
    # 匹配所有 type="fixed" 的关节
    pattern = r'<joint name="([^"]+)"([^>]*)type="fixed"([^>]*)>'

    def add_attribute(match):
        nonlocal modified_count
        joint_name = match.group(1)
        before = match.group(2)
        after = match.group(3)
        # 检查是否已经有 dont_collapse 属性
        if 'dont_collapse' not in before + after:
            modified_count += 1
            if verbose:
                print(f"  ✓ 添加 dont_collapse 到: {joint_name}")
            return f'<joint name="{joint_name}"{before}type="fixed"{after} dont_collapse="true">'
        return match.group(0)
    content = re.sub(pattern, add_attribute, content)
    return content, modified_count


# ==================== 主处理函数 ====================

def process_urdf(input_path, output_path=None, custom_limits=None, custom_origins=None, verbose=True):
    #
    # 处理 URDF 文件，应用所有转换
    #
    # Args:
    #     input_path: 输入 URDF 文件路径
    #     output_path: 输出 URDF 文件路径（如果为 None，则覆盖输入文件）
    #     custom_limits: 自定义关节限制字典
    #     verbose: 是否打印详细信息
    #
    # Returns:
    #     转换统计字典
    #
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\n{'='*60}")
        print("开始处理 URDF 文件")
        print(f"{'='*60}")
        print(f"输入文件: {input_path}")
        print(f"输出文件: {output_path}")
        print()
    # 读取文件
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 检测文件是否已绋处理
    is_processed, issues = check_if_already_processed(content, custom_limits, custom_origins, None, verbose)
    if is_processed:
        if verbose:
            print(f"\n{'='*60}")
            print("文件无需处理")
            print(f"{'='*60}")
            print("✅ 文件已经满足所有转换要求，跳过处理")
            print(f"{'='*60}\n")
        return {'already_processed': True, 'skipped': True}
    stats = {}
    # 1. 转换 continuous 关节
    if verbose:
        print("步骤 1: 转换 continuous 关节为 revolute")
    content, joint_stats = convert_continuous_to_revolute(content, custom_limits, verbose)
    stats['joints'] = joint_stats
    # 2. 修正已存在的 revolute 关节限位
    if verbose:
        print("\n步骤 2: 修正 revolute 关节的限位值")
    content, limit_stats = fix_revolute_joint_limits(content, custom_limits, verbose)
    stats['joint_limits'] = limit_stats
    # 2.5. 修正 prismatic 关节限位
    if verbose:
        print("\n步骤 2.5: 修正 prismatic 关节的限位值")
    content, prismatic_stats = fix_prismatic_joint_limits(content, verbose=verbose)
    stats['prismatic_limits'] = prismatic_stats
    # 3. 移除 package:// 前缀
    if verbose:
        print("\n步骤 3: 移除 mesh 路径中的 package:// 前缀")
    content, mesh_count = remove_package_prefix(content, verbose)
    stats['mesh_paths'] = mesh_count
    # 4. 修改 mujoco meshdir 路径
    if verbose:
        print("\n步骤 4: 修改 mujoco meshdir 路径")
    content, mujoco_count = fix_mujoco_meshdir(content, verbose)
    stats['mujoco_meshdir'] = mujoco_count
    # 5. 为固定关节添加 dont_collapse
    if verbose:
        print("\n步骤 5: 为所有固定关节添加 dont_collapse 属性")
    content, fixed_count = add_dont_collapse_to_fixed_joints(content, verbose)
    stats['fixed_joints'] = fixed_count
    # 6. 修改关节 origin
    if verbose:
        print("\n步骤 6: 修改关节 origin 位置")
    content, origin_count = fix_joint_origins(content, custom_origins, verbose)
    stats['joint_origins'] = origin_count
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    if verbose:
        print(f"\n{'='*60}")
        print("处理完成！")
        print(f"{'='*60}")
        print(f"✓ 转换了 {stats['joints']['converted']} 个 continuous 关节")
        if stats['joints']['skipped']:
            print(f"⚠ 跳过了 {len(stats['joints']['skipped'])} 个关节: {', '.join(stats['joints']['skipped'])}")
        print(f"✓ 修正了 {stats['joint_limits']['fixed']} 个关节的限位值")
        if stats['joint_limits']['skipped']:
            print(f"⚠ {len(stats['joint_limits']['skipped'])} 个关节未找到匹配的限位配置")
        print(f"✓ 修正了 {stats['prismatic_limits']['fixed']} 个移动关节的限位值")
        print(f"✓ 修改了 {stats['mesh_paths']} 个 mesh 路径")
        print(f"✓ 修改了 {stats['mujoco_meshdir']} 个 mujoco meshdir 路径")
        print(f"✓ 修改了 {stats['fixed_joints']} 个固定关节")
        print(f"✓ 修改了 {stats['joint_origins']} 个关节 origin")
        print(f"{'='*60}\n")
    return stats


# ==================== 命令行接口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='URDF 转换工具 - 将 URDF 转换为 Isaac Gym 兼容格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置（在脚本顶部设置的路径）
  python fix_urdf.py

  # 指定输入输出文件
  python fix_urdf.py input.urdf output.urdf

  # 静默模式
  python fix_urdf.py --quiet
        """
    )

    parser.add_argument('input', nargs='?',
                        default=INPUT_URDF,
                        help=f'输入 URDF 文件路径 (默认: {INPUT_URDF})')
    parser.add_argument('output', nargs='?',
                        default=OUTPUT_URDF,
                        help=f'输出 URDF 文件路径 (默认: {OUTPUT_URDF})')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='静默模式，不打印详细信息')

    args = parser.parse_args()

    try:
        process_urdf(
            input_path=args.input,
            output_path=args.output,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
