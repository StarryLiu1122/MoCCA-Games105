##############
# 姓名：
# 学号：
##############
# 以下部分均为可更改部分，你可以把需要的数据结构定义进来，可以继承自Graph class
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
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge : Edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = -1
        

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
        
        # 当前角色的参考root旋转
        self.cur_root_rot, _ = BVHMotion.decompose_rotation_with_yaxis(self.cur_node.motion.joint_rotation[0, 0])
    
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
        # # 一个简单的例子，循环播放第0个动画，不会响应输入信号
        joint_name = self.cur_node.motion.joint_name
        joint_translation, joint_orientation = self.cur_node.motion.batch_forward_kinematics()
        joint_translation = joint_translation[self.cur_frame]
        joint_orientation = joint_orientation[self.cur_frame]
        
        # 更新你的表示角色的信息
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        self.cur_frame = (self.cur_frame + 1) % self.cur_node.motion.motion_length
        # 一直处于第0个动画所在的node
        self.cur_node = self.graph.nodes[0]
        # 不会切换，所以一直不会播放transition动画
        self.cur_edge = None


        return joint_name, joint_translation, joint_orientation
    
    
    
    
    
##############
# 姓名：刘星云
# 学号：2300012297
##############
# 以下部分均为可更改部分，你可以把需要的数据结构定义进来，可以继承自Graph class
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
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge : Edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = -1

        self.motions = []
        self.motions.append(BVHMotion('motion_material/walk_forward.bvh'))  # 100
        self.motions.append(BVHMotion('motion_material/idle.bvh'))
        
        self.blending_motion = build_loop_motion(self.motions[0])
        self.idle_motion = build_loop_motion(self.motions[1])
        self.idle2move_motion = concatenate_two_motions(self.motions[1], self.motions[0], 60, 30)
        self.move2idle_motion = concatenate_two_motions(self.motions[0], self.motions[1], 60, 30)
        self.motion_state = "idle"
        
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
        
        # 当前角色的参考root旋转
        self.cur_root_rot, _ = BVHMotion.decompose_rotation_with_yaxis(self.cur_node.motion.joint_rotation[0, 0])
    
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
        # # 一个简单的例子，循环播放第0个动画，不会响应输入信号
        joint_name = self.cur_node.motion.joint_name
        joint_translation, joint_orientation = self.cur_node.motion.batch_forward_kinematics()
        joint_translation = joint_translation[self.cur_frame]
        joint_orientation = joint_orientation[self.cur_frame]
        
        # 更新你的表示角色的信息
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        self.cur_frame = (self.cur_frame + 1) % self.cur_node.motion.motion_length
        # 一直处于第0个动画所在的node
        self.cur_node = self.graph.nodes[0]
        # 不会切换，所以一直不会播放transition动画
        self.cur_edge = None
        
        
        # joint_name = self.blending_motion.joint_name

        # last_motion_state = self.motion_state
        # self.motion_state = "idle" if abs(desired_vel_list[0,0])+abs(desired_vel_list[0,1]) < 1e-2 else "move"

        # if self.motion_state == "move":
        #     motion_id = self.motion_id
        #     current_motion = self.blending_motion.raw_copy()
        #     if self.motion_state != last_motion_state:
        #         facing_axis = R.from_quat(self.idle_motion.joint_rotation[self.cur_frame, 0, :]).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
        #         current_motion = current_motion.translation_and_rotation(0, self.idle_motion.joint_position[self.cur_frame, 0, [0, 2]],
        #                                                                  facing_axis)



        #         self.cur_frame = 0
        #     key_frame = [(self.cur_frame + 20 * i) % self.motions[motion_id].motion_length for i in range(6)]
        #     current_motion_key_frame_vel = current_motion.joint_position[key_frame, 0, :] - current_motion.joint_position[[(frame - 1) for frame in key_frame], 0, :]
        #     current_motion_avel = quat_to_avel(current_motion.joint_rotation[:, 0, :], 1 / 60)

        #     # It is only for root bone

        #     diff_root_pos = desired_pos_list - current_motion.joint_position[ key_frame, 0, :]
        #     diff_root_pos[:, 1] = 0
        #     diff_root_rot = (R.from_quat(desired_rot_list[0:6]) * R.from_quat(current_motion.joint_rotation[ key_frame, 0, :]).inv()).as_rotvec()
        #     diff_root_vel = (desired_vel_list - current_motion_key_frame_vel)/60
        #     diff_root_avel = desired_avel_list[0:6] - current_motion_avel[[(frame-1) for frame in key_frame]]

        #     for i in range(self.cur_frame, self.cur_frame+self.motions[motion_id].motion_length//2):
        #         half_time = 0.2
        #         index = (i - self.cur_frame) // 20
        #         dt = (i-self.cur_frame) % 20

        #         off_pos, _ = decay_spring_implicit_damping_pos(diff_root_pos[index], diff_root_vel[index], half_time, dt/60)
        #         off_rot, _ = decay_spring_implicit_damping_rot(diff_root_rot[index], diff_root_avel[index], half_time, dt/60)

        #         current_motion.joint_position[ i % self.motions[motion_id].motion_length, 0, :] += off_pos
        #         current_motion.joint_rotation[ i % self.motions[motion_id].motion_length, 0, :] = (R.from_rotvec(off_rot) * R.from_quat(current_motion.joint_rotation[ i % self.motions[motion_id].motion_length, 0, :])).as_quat()

        #     joint_translation, joint_orientation = current_motion.batch_forward_kinematics()
        #     joint_translation = joint_translation[self.cur_frame]
        #     joint_orientation = joint_orientation[self.cur_frame]
        #     self.cur_root_pos = joint_translation[0]
        #     self.cur_root_rot = joint_orientation[0]

        #     self.blending_motion = current_motion
        #     self.cur_frame = (self.cur_frame + 1) % self.motions[motion_id].motion_length

        # elif self.motion_state == "idle":
        #     motion_id = self.motion_id
        #     current_motion = self.idle_motion
        #     if self.motion_state != last_motion_state:
        #         facing_axis = R.from_quat(self.blending_motion.joint_rotation[self.cur_frame, 0, :]).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
        #         current_motion = current_motion.translation_and_rotation(0, self.blending_motion.joint_position[self.cur_frame, 0, [0, 2]],
        #                                                                  facing_axis)
        #         self.cur_frame = 0

        #     joint_translation, joint_orientation = current_motion.batch_forward_kinematics()
        #     joint_translation = joint_translation[self.cur_frame]
        #     joint_orientation = joint_orientation[self.cur_frame]
        #     self.cur_root_pos = joint_translation[0]
        #     self.cur_root_rot = joint_orientation[0]
        #     self.cur_frame = 0
        #     self.idle_motion = current_motion
        #     self.cur_frame = (self.cur_frame + 1) % self.motions[motion_id].motion_length


        return joint_name, joint_translation, joint_orientation












##############
# 姓名：刘星云
# 学号：2300012297
##############
# 以下部分均为可更改部分，你可以把需要的数据结构定义进来，可以继承自Graph class
from graph import *
from answer_task1 import *
from typing import List
from bvh_motion import BVHMotion
from scipy.spatial.transform import Rotation as R

def normalize_and_check_quaternion(q):
    norm = np.linalg.norm(q)
    if norm < 1e-5:
        print("Warning: Quaternion norm too small, using identity quaternion.")
        return np.array([0, 0, 0, 1]) 
    return q / norm

def get_yaw_from_quat(quat):
    """从四元数中获取绕Y轴的旋转角度，范围为 [0, 360)"""
    forward = R.from_quat(quat).apply([0, 0, 1])
    x, z = forward[0], forward[2]
    yaw = np.degrees(np.arctan2(x, z))  # 注意是x在前
    if yaw < 0:
        yaw += 360
    return yaw

def angle_difference(a, b):
    """计算两个角度之间的最短差值 [-180, 180]"""
    d = a - b
    # if d >= 0:
    #     if d >= 180:
    #         d -= 180
    # else:
    #     if d <= -180:
    #         d += 180
    return d

def choose_motion_by_angle(delta_angle):
    """根据角度选择合适的motion节点"""
    if abs(delta_angle) < 10:  # 直行
        return "walk"
    elif delta_angle > 0:
        if delta_angle < 95:
            return "turn_left"
        else:
            return "spin_counter_clockwise"
        # return "spin_counter_clockwise"
    else:
        if delta_angle > -95:
            return "turn_right"
        else:
            return "spin_clockwise"

# def choose_motion_by_angle(delta_angle):
#     """根据角度选择合适的motion节点"""
#     if abs(delta_angle) < 5:  # 直行
#         return "walk"
#     elif delta_angle > 0:
#         if delta_angle < 90:
#             return "spin_counter_clockwise"
#         else:
#             return "turn_left"
#     else:
#         if delta_angle > -90:
#             return "spin_clockwise"
#         else:
#             return "turn_right"

def find_node_by_name(graph, name):
    for node in graph.nodes:
        if node.name.startswith(name):
            return node
    return None


class CharacterController():
    def __init__(self, controller) -> None:
        # 手柄/键盘控制器
        self.controller = controller
        # 动作
        self.motions = []
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
        # 角色将转到Graph的哪一个节点
        self.next_node : Node = None
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge : Edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = -1

        # 初始化上述参数
        self.initialize()
        
    def initialize(self):
        # 动作
        self.motions.append(BVHMotion('motion_material/walk.bvh')) 
        self.motions.append(BVHMotion('motion_material/spin_clockwise.bvh'))
        self.motions.append(BVHMotion('motion_material/spin_counter_clockwise.bvh')) 
        self.motions.append(BVHMotion('motion_material/turn_left.bvh'))
        self.motions.append(BVHMotion('motion_material/turn_right.bvh')) 
        # self.blending_motion = build_loop_motion(self.motions[0])
        # self.idle_motion = build_loop_motion(self.motions[1])
        # self.idle2move_motion = concatenate_two_motions(self.motions[1], self.motions[0], 60, 30)
        # self.move2idle_motion = concatenate_two_motions(self.motions[0], self.motions[1], 60, 30)
        
        # 当前角色处于Graph的哪一个节点
        self.cur_node = self.graph.nodes[0]
        # 角色将转到Graph的哪一个节点
        self.next_node : Node = None
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_node.motion.motion_length
        
        # 当前角色的参考root位置
        self.cur_root_pos = self.cur_node.motion.joint_position[0,0,:].copy()
        self.cur_root_pos[1] = 0 # 忽略竖直方向，即y方向的位移
        
        # 当前角色的参考root旋转
        self.cur_root_rot, _ = BVHMotion.decompose_rotation_with_yaxis(self.cur_node.motion.joint_rotation[0, 0])
    
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

        joint_name = self.cur_node.motion.joint_name

        # 当前角色的朝向
        # current_rot = self.cur_root_rot.squeeze()
        
        current_rot = desired_rot_list[0]
        raw_rot = desired_rot_list[-1]
        # current_euler = R.from_quat(current_rot).as_euler('XYZ', degrees=True).squeeze()
        # desired_rot = R.from_quat(raw_rot).as_euler('XYZ', degrees=True)
        # delta_angle = angle_difference(desired_rot[1], current_euler[1])
        # print(delta_angle,current_euler[1],desired_rot[1])
        
        current_yaw = get_yaw_from_quat(current_rot)
        desired_yaw = get_yaw_from_quat(raw_rot)
        delta_angle = angle_difference(desired_yaw, current_yaw)
        print(delta_angle, current_yaw, desired_yaw)

        # 选择合适的动作
        motion_name = choose_motion_by_angle(delta_angle)
        next_node = find_node_by_name(self.graph, motion_name)

        
        if next_node is None:
            next_node = self.graph.nodes[0]
        else:
            self.next_node = next_node
            if next_node.name != self.cur_node.name:
                print(f"{next_node.name},{self.cur_node.name}")

        # 末尾，切换动作
        if self.cur_frame >= self.cur_end_frame - 1:
            rot = self.cur_node.motion.joint_rotation[-1,0]
            self.cur_node = next_node
            # print(self.cur_node.name)
            self.cur_frame = 0
            self.cur_end_frame = self.cur_node.motion.motion_length
            # 对齐新动作
            facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
            self.cur_node.motion = BVHMotion.translation_and_rotation(self.cur_node.motion, 0, self.cur_root_pos[[0, 2]], facing_axis)
        
        # 状态转换
        if next_node.name != self.cur_node.name:
            rot = self.cur_node.motion.joint_rotation[self.cur_frame,0]
            self.cur_node = next_node
            self.cur_frame = 0
            self.cur_end_frame = self.cur_node.motion.motion_length
            # 对齐新动作
            facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
            self.cur_node.motion = BVHMotion.translation_and_rotation(self.cur_node.motion, 0, self.cur_root_pos[[0, 2]], facing_axis)
            
            
        # 执行前向运动学
        joint_translation, joint_orientation = self.cur_node.motion.batch_forward_kinematics()
        joint_translation = joint_translation[self.cur_frame]
        joint_orientation = joint_orientation[self.cur_frame]

        # 更新root位置和方向
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]

        self.cur_frame += 1

        return joint_name, joint_translation, joint_orientation


