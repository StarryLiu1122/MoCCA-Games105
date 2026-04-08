# ##############
# # 姓名：刘星云
# # 学号：2300012297
# ##############
# # 以下部分均为可更改部分，你可以把需要的数据结构定义进来，可以继承自Graph class
from graph import *
from answer_task1 import *
from typing import List
from bvh_motion import BVHMotion
from scipy.spatial.transform import Rotation as R

class CharacterController():
    def __init__(self, controller) -> None:
        # 手柄/键盘控制器
        self.controller = controller
        # 读取graph结构
        self.graph = Graph('./nodes.npy')
        self.graph.load_from_file()
        # node name组成的List
        self.node_names = [nd.name for nd in self.graph.nodes]
        # edge name组成的List
        self.edge_names = []
        for nd in self.graph.nodes:
            for eg in nd.edges:
                self.edge_names.append(eg.label)

        # 下面是你可能会需要的成员变量，只是一个例子形式
        # 当然，你可以任意编辑，来符合你的要求
        # 当前角色的参考root位置
        self.cur_root_pos = None
        # 当前角色的参考root旋转
        self.cur_root_rot = None
        # 当前角色处于Graph的哪一个节点
        self.cur_node : Node = None
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)  # 没用edge
        self.cur_edge : Edge = None 
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = -1
        # 当前角色的动作
        self.cur_motion : BVHMotion = None
        # 初始化上述参数
        self.initialize()
        
    def initialize(self):
        # 当前角色处于Graph的哪一个节点
        self.cur_node = self.graph.nodes[0]
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_node.motion.motion_length
        # 当前角色的参考root位置
        self.cur_root_pos = self.cur_node.motion.joint_position[0,0,:].copy()
        self.cur_root_pos[1] = 0 # 忽略竖直方向，即y方向的位移
        # 当前角色的动作
        self.cur_motion = self.cur_node.motion
        
        # 当前角色的参考root旋转
        self.cur_root_rot, _ = BVHMotion.decompose_rotation_with_yaxis(self.cur_node.motion.joint_rotation[0, 0])
    
    def choose_next_node(self, desired_rot_list): 
        current_rot = R.from_quat(self.cur_root_rot)
        desired_rot = R.from_quat(desired_rot_list[-1])
        diff_rot = desired_rot * current_rot.inv()
        delta_angle = diff_rot.as_rotvec()[1]

        if delta_angle > np.pi * 3 / 4:
            return self.graph.nodes[4]
        
        elif delta_angle > np.pi / 4:
            return self.graph.nodes[1]
        
        elif delta_angle > -np.pi / 4:
            return self.graph.nodes[0]
        
        elif delta_angle > -np.pi * 3 / 4:
            return self.graph.nodes[2]
        
        else:
            return self.graph.nodes[3]


    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list
                     ):
        '''
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态
            当然我们只是提供了速度和角速度的输入，如果通过pos和rot已经很好选择下一个动作了，可以不必须使用速度和角速度
            desired_pos_list: 期望位置, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望位置(XoZ平面)， 期望位置可以用来拟合根节点位置
            desired_rot_list: 期望旋转, 6x4的矩阵, 四元数, 每一行对应0，20，40...帧的期望旋转(Y旋转), 期望旋转可以用来拟合根节点旋转
            desired_vel_list: 期望速度, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望速度(XoZ平面), 期望速度可以用来拟合根节点速度
            desired_avel_list: 期望角速度, 6x3的矩阵, [0, y, 0], 每一行对应0，20，40...帧的期望角速度(Y旋转), 期望角速度可以用来拟合根节点角速度
        
        Output: 输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            1. 注意应该利用的期望位置和期望速度应该都是在XoZ平面内，期望旋转和期望角速度都是绕Y轴的旋转。其他的项没有意义

        '''
        
        if self.cur_frame < self.cur_end_frame:
            joint_name = self.cur_motion.joint_name
            joint_translation, joint_orientation = self.cur_motion.batch_forward_kinematics()
            joint_translation = joint_translation[self.cur_frame]
            joint_orientation = joint_orientation[self.cur_frame]
            self.cur_root_pos = joint_translation[0]
            self.cur_root_rot = joint_orientation[0]
            self.cur_frame += 1
            return joint_name, joint_translation, joint_orientation
        else:
            next_node = self.choose_next_node(desired_rot_list)
            self.cur_node = next_node
            joint_name = self.cur_motion.joint_name
            self.cur_frame = 0
            
            facing_axis = R.from_quat(self.cur_root_rot).apply(np.array([0,0,1])).flatten()[[0,2]]
            next_motion = self.cur_node.motion.translation_and_rotation(0, self.cur_root_pos[[0,2]], facing_axis)
            smooth_motion = concatenate_two_motions(self.cur_motion, next_motion, self.cur_end_frame-1, 30)
            smooth_motion.joint_position = smooth_motion.joint_position[self.cur_end_frame:,:,:]
            smooth_motion.joint_rotation = smooth_motion.joint_rotation[self.cur_end_frame:,:,:]
            self.cur_motion = smooth_motion
            self.cur_end_frame = self.cur_motion.motion_length

            joint_translation, joint_orientation = self.cur_motion.batch_forward_kinematics()
            joint_translation = joint_translation[0]
            joint_orientation = joint_orientation[0]
            self.cur_root_pos = joint_translation[0]
            self.cur_root_rot = joint_orientation[0]
        return joint_name, joint_translation, joint_orientation
    
    
# from graph import *
# from answer_task1 import *
# from typing import List
# from bvh_motion import BVHMotion
# from scipy.spatial.transform import Rotation as R


# def decompose_rotation_with_yaxis(self, rotation):
#     Ry = R.from_quat(rotation).as_euler("XYZ", degrees=True)
#     Ry = R.from_euler("XYZ", [0, Ry[1], 0], degrees=True)
#     Rxz = Ry.inv() * R.from_quat(rotation)
#     return Ry, Rxz

# def translation_and_rotation(self, frame_num, target_translation_xz, target_facing_direction_xz):
#     res = self.raw_copy() 
#     offset = target_translation_xz - res.joint_position[frame_num, 0, [0,2]]
#     res.joint_position[:, 0, [0, 2]] += offset
#     sin_theta_xz = np.cross(target_facing_direction_xz, np.array([0, 1])) / np.linalg.norm(target_facing_direction_xz)
#     cos_theta_xz = np.dot(target_facing_direction_xz, np.array([0, 1])) / np.linalg.norm(target_facing_direction_xz)
#     theta = np.arccos(cos_theta_xz)
#     if sin_theta_xz < 0:
#         theta = 2 * np.pi - theta
#     new_root_Ry = R.from_euler("Y", theta, degrees=False)
#     R_y, _ = decompose_rotation_with_yaxis(self, res.joint_rotation[frame_num, 0, :])
#     res.joint_rotation[:, 0, :] = (new_root_Ry * R_y.inv() * R.from_quat(res.joint_rotation[:, 0, :])).as_quat()
#     for i in range(len(res.joint_position)):
#          res.joint_position[i, 0,:] = (new_root_Ry * R_y.inv()).as_matrix()  @ (res.joint_position[i, 0, :] - res.joint_position[frame_num, 0, :]) + res.joint_position[frame_num,0,:]
#     return res
    
# def get_yaw_from_quat(quat):
#     """获取绕Y轴的旋转角度，范围为 [0, 360)"""
#     forward = R.from_quat(quat).apply([0, 0, 1])
#     x, z = forward[0], forward[2]
#     yaw = np.degrees(np.arctan2(x, z))  # 注意是x在前
#     if yaw < 0:
#         yaw += 360
#     return yaw

# def angle_difference(a, b):
#     """计算两个角度之间的最短差值 [-180, 180]"""
#     d = a - b
#     # if d >= 0:
#     #     if d >= 180:
#     #         d -= 180
#     # else:
#     #     if d <= -180:
#     #         d += 180
#     return d

# def choose_motion_by_angle(delta_angle):
#     """根据角度选择合适的motion节点"""
#     if abs(delta_angle) < 10:  # 直行
#         return "walk"
#     elif delta_angle > 0:
#         if delta_angle < 100:
#             return "turn_left"
#         else:
#             return "spin_counter_clockwise"
#     else:
#         if delta_angle > -100:
#             return "turn_right"
#         else:
#             return "spin_clockwise"

# def find_node_by_name(graph, name):
#     for node in graph.nodes:
#         if node.name.startswith(name):
#             return node
#     return None


# class CharacterController():
#     def __init__(self, controller) -> None:
#         # 手柄/键盘控制器
#         self.controller = controller
#         # 动作
#         self.motions = []
#         self.motion_id = 0
#         self.motion_state = "walk"
#         # 读取graph结构
#         self.graph = Graph('./nodes.npy')
#         self.graph.load_from_file()
#         # node name组成的List
#         self.node_names = [nd.name for nd in self.graph.nodes]
#         # edge name组成的List
#         self.edge_names = []
#         for nd in self.graph.nodes:
#             for eg in nd.edges:
#                 self.edge_names.append(eg.label)

#         # 下面是你可能会需要的成员变量，只是一个例子形式
#         # 当然，你可以任意编辑，来符合你的要求
#         # 当前角色的参考root位置
#         self.cur_root_pos = None
#         # 当前角色的参考root旋转
#         self.cur_root_rot = None
#         # 当前角色处于Graph的哪一个节点
#         self.cur_node : Node = None
#         # 角色将转到Graph的哪一个节点
#         self.next_node : Node = None
#         # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
#         self.cur_edge : Edge = None
#         # 当前角色处于正在跑的BVH的第几帧
#         self.cur_frame = 0
#         # 当前角色对应的BVH的结束的帧数
#         self.cur_end_frame = -1

#         self.spring_w = 2.0  # 弹簧频率
#         self.spring_d = 0.9  # 阻尼比
#         self.fps = 60        # 帧率
        
#         # 初始化上述参数
#         self.initialize()
        
#     def initialize(self):
#         # 动作
#         # self.motions.append(BVHMotion('motion_material/walk.bvh')) 
#         # self.motions.append(BVHMotion('motion_material/spin_clockwise.bvh'))
#         # self.motions.append(BVHMotion('motion_material/spin_counter_clockwise.bvh')) 
#         # self.motions.append(BVHMotion('motion_material/turn_left.bvh'))
#         # self.motions.append(BVHMotion('motion_material/turn_right.bvh')) 
#         motion = BVHMotion('motion_material/walk.bvh')
#         motion = build_loop_motion(motion, 0.5, 0.2)
#         pos = motion.joint_position[-1,0,[0,2]]
#         rot = motion.joint_rotation[-1,0]      
#         facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
#         new_motion = motion.translation_and_rotation(0, pos, facing_axis)
#         self.walk_motion = new_motion
#         self.motions.append(self.walk_motion)
#         self.motion_id = len(self.motions) - 1
        
#         # self.walk_motion = BVHMotion('motion_material/walk.bvh')
#         # self.motions.append(self.walk_motion)
#         # self.motion_id = len(self.motions) - 1

#         self.turn_left_motion = BVHMotion('motion_material/turn_left.bvh')
#         self.motions.append(self.turn_left_motion)
#         self.motion_id_turn_left = len(self.motions) - 1

#         self.turn_right_motion = BVHMotion('motion_material/turn_right.bvh')
#         self.motions.append(self.turn_right_motion)
#         self.motion_id_turn_right = len(self.motions) - 1

#         self.spin_clockwise_motion = BVHMotion('motion_material/spin_clockwise.bvh')
#         self.motions.append(self.spin_clockwise_motion)
#         self.motion_id_spin_clockwise = len(self.motions) - 1

#         self.spin_counter_clockwise_motion = BVHMotion('motion_material/spin_counter_clockwise.bvh')
#         self.motions.append(self.spin_counter_clockwise_motion)
#         self.motion_id_spin_counter_clockwise = len(self.motions) - 1
        
#         # self.walk_motion = build_loop_motion(self.motions[0], ratio=0.5, half_life=0.2)
#         # self.turn_left_motion = build_loop_motion(self.motions[1], ratio=0.5, half_life=0.2)
#         # self.left2walk_motion = concatenate_two_motions(self.motions[1], self.motions[0], 60, 30)
#         # self.walk2left_motion = concatenate_two_motions(self.motions[0], self.motions[1], 60, 30)
#         # self.motion_id = 0
#         self.motion_state = "walk"
        
#         # 当前角色处于Graph的哪一个节点
#         self.cur_node = self.graph.nodes[0]
#         # 角色将转到Graph的哪一个节点
#         self.next_node : Node = None
#         # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
#         self.cur_edge = None
#         # 当前角色处于正在跑的BVH的第几帧
#         self.cur_frame = 0
#         # 当前角色对应的BVH的结束的帧数
#         self.cur_end_frame = self.cur_node.motion.motion_length
        
#         # 当前角色的参考root位置
#         self.cur_root_pos = self.cur_node.motion.joint_position[0,0,:].copy()
#         self.cur_root_pos[1] = 0 # 忽略竖直方向，即y方向的位移
        
#         # 当前角色的参考root旋转
#         self.cur_root_rot, _ = BVHMotion.decompose_rotation_with_yaxis(self.cur_node.motion.joint_rotation[0, 0])
    
#     def update_state(self, 
#                      desired_pos_list, 
#                      desired_rot_list,
#                      desired_vel_list,
#                      desired_avel_list
#                      ):
#         '''
#         Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态
#             当然我们只是提供了速度和角速度的输入，如果通过pos和rot已经很好选择下一个动作了，可以不必须使用速度和角速度
#             desired_pos_list: 期望位置, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望位置(XoZ平面)， 期望位置可以用来拟合根节点位置
#             desired_rot_list: 期望旋转, 6x4的矩阵, 四元数, 每一行对应0，20，40...帧的期望旋转(Y旋转), 期望旋转可以用来拟合根节点旋转
#             desired_vel_list: 期望速度, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望速度(XoZ平面), 期望速度可以用来拟合根节点速度
#             desired_avel_list: 期望角速度, 6x3的矩阵, [0, y, 0], 每一行对应0，20，40...帧的期望角速度(Y旋转), 期望角速度可以用来拟合根节点角速度
        
#         Output: 输出下一帧的关节名字,关节位置,关节旋转
#             joint_name: List[str], 代表了所有关节的名字
#             joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
#             joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
#         Tips:
#             1. 注意应该利用的期望位置和期望速度应该都是在XoZ平面内，期望旋转和期望角速度都是绕Y轴的旋转。其他的项没有意义

#         '''
#         # print(desired_vel_list)
#         joint_name = self.cur_node.motion.joint_name

#         current_rot = desired_rot_list[0]
#         raw_rot = desired_rot_list[-1]
#         current_yaw = get_yaw_from_quat(current_rot)
#         desired_yaw = get_yaw_from_quat(raw_rot)
#         delta_angle = angle_difference(desired_yaw, current_yaw)

#         last_motion_state = self.motion_state

#         # 判断当前状态
#         if abs(delta_angle) < 15:
#             self.motion_state = "walk"
#         elif 15 <= delta_angle < 60:
#             self.motion_state = "turn_right" if delta_angle > 0 else "turn_left"
#         else:
#             self.motion_state = "spin_clockwise" if delta_angle > 0 else "spin_counter_clockwise"

#         # 动作选择
#         if self.motion_state == "walk":
#             current_motion = self.walk_motion.raw_copy()
#             motion_id = self.motion_id

#         elif self.motion_state == "turn_left":
#             current_motion = self.turn_left_motion.raw_copy()
#             motion_id = self.motion_id_turn_left

#         elif self.motion_state == "turn_right":
#             current_motion = self.turn_right_motion.raw_copy()
#             motion_id = self.motion_id_turn_right

#         elif self.motion_state == "spin_clockwise":
#             current_motion = self.spin_clockwise_motion.raw_copy()
#             motion_id = self.motion_id_spin_clockwise

#         elif self.motion_state == "spin_counter_clockwise":
#             current_motion = self.spin_counter_clockwise_motion.raw_copy()
#             motion_id = self.motion_id_spin_counter_clockwise

#         # 切换动作时需要对齐
#         if self.motion_state != last_motion_state:
#             # 计算当前和目标旋转的过渡（插值）
#             facing_axis = R.from_quat(self.walk_motion.joint_rotation[self.cur_frame, 0, :]).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
#             current_motion = translation_and_rotation(current_motion, 0, self.walk_motion.joint_position[self.cur_frame, 0, [0, 2]], facing_axis)
#             self.cur_frame = 0

#         # 调整位置和旋转的平滑过渡
#         key_frame = [(self.cur_frame + 20 * i) % self.motions[motion_id].motion_length for i in range(6)]
#         current_motion_key_frame_vel = current_motion.joint_position[key_frame, 0, :] - current_motion.joint_position[[(frame - 1) for frame in key_frame], 0, :]
#         current_motion_avel = quat_to_avel(current_motion.joint_rotation[:, 0, :], 1 / 60)

#         # 计算位置和旋转的差异
#         diff_root_pos = desired_pos_list - current_motion.joint_position[key_frame, 0, :]
#         diff_root_pos[:, 1] = 0
#         diff_root_rot = (R.from_quat(desired_rot_list[0:6]) * R.from_quat(current_motion.joint_rotation[key_frame, 0, :]).inv()).as_rotvec()
#         diff_root_vel = (desired_vel_list - current_motion_key_frame_vel) / 60
#         diff_root_avel = desired_avel_list[0:6] - current_motion_avel[[(frame - 1) for frame in key_frame]]

#         # 动作惰性化过渡，减少突然变化
#         for i in range(self.cur_frame, self.cur_frame + self.motions[motion_id].motion_length // 2):
#             half_time = 0.2  # 惰性过渡时间
#             index = (i - self.cur_frame) // 20
#             dt = (i - self.cur_frame) % 20

#             # 过渡位置和旋转
#             off_pos, _ = decay_spring_implicit_damping_pos(diff_root_pos[index], diff_root_vel[index], half_time, dt / 60)
#             off_rot, _ = decay_spring_implicit_damping_rot(diff_root_rot[index], diff_root_avel[index], half_time, dt / 60)

#             current_motion.joint_position[i % self.motions[motion_id].motion_length, 0, :] += off_pos
#             current_motion.joint_rotation[i % self.motions[motion_id].motion_length, 0, :] = (
#                 R.from_rotvec(off_rot) * R.from_quat(current_motion.joint_rotation[i % self.motions[motion_id].motion_length, 0, :])
#             ).as_quat()

#         joint_translation, joint_orientation = current_motion.batch_forward_kinematics()
#         joint_translation = joint_translation[self.cur_frame]
#         joint_orientation = joint_orientation[self.cur_frame]
#         self.cur_root_pos = joint_translation[0]
#         self.cur_root_rot = joint_orientation[0]

#         # 更新当前动作
#         if self.motion_state == "walk":
#             self.walk_motion = current_motion
#         elif self.motion_state == "turn_left":
#             self.turn_left_motion = current_motion
#         elif self.motion_state == "turn_right":
#             self.turn_right_motion = current_motion
#         elif self.motion_state == "spin_clockwise":
#             self.spin_clockwise_motion = current_motion
#         elif self.motion_state == "spin_counter_clockwise":
#             self.spin_counter_clockwise_motion = current_motion

#         self.cur_frame = (self.cur_frame + 1) % self.motions[motion_id].motion_length

#         return joint_name, joint_translation, joint_orientation



