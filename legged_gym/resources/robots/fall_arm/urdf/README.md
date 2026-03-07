# FallArm URDF 文件请放在此目录下
# 文件名: fall_arm.urdf
#
# URDF 关节结构要求:
#
# world (固定)
#   └─ base_link
#        └─ slider_joint (prismatic, axis="0 0 1")  ← 垂直导轨, 被动
#             └─ slider_link
#                  └─ shoulder_pitch_joint (revolute)
#                       └─ upper_arm
#                            └─ shoulder_roll_joint (revolute)
#                                 └─ upper_arm_roll
#                                      └─ shoulder_yaw_joint (revolute)
#                                           └─ upper_arm_yaw
#                                                └─ elbow_joint (revolute)
#                                                     └─ forearm
#                                                          └─ end_effector (末端着地点)
#
# 注意:
#   1. 关节名称必须与 fall_arm_config.py 中的 default_joint_angles 和 stiffness/damping 一致
#   2. 末端刚体名称须包含 "end_effector" (对应 asset.foot_name)
#   3. slider_joint 的 effort 可设为 0 (被动) 或设大值 (代码中增益已为0)
#   4. slider_link 名称要与 terminate_after_contacts_on 一致
#   5. upper_arm / forearm 名称要与 penalize_contacts_on 一致
#   6. fix_base_link=True, 即 base_link 固定在世界中
