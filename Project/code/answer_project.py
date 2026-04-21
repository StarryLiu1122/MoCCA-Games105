##############
# 姓名：刘星云
# 学号：2300012297
##############

# 以下部分均为可更改部分，你可以把需要的数据结构定义进来
from typing import List
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
from physics_warpper import PhysicsInfo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def part1_cal_torque(pose, physics_info: PhysicsInfo, **kargs):
    '''
    输入： pose: (20, 4)的numpy数组，表示每个关节的目标旋转(相对于父关节的)
           physics_info: PhysicsInfo类，包含了当前的物理信息，参见physics_warpper.py
           **kargs: 指定参数，可能包含kp,kd
    输出： global_torque: (20, 3)的numpy数组，表示每个关节的全局坐标下的目标力矩，因为不需要控制方向，根节点力矩会被后续代码无视
    '''
    # ------一些提示代码，你可以随意修改------------#
    kp = kargs.get('kp', 800) # 需要自行调整kp和kd！ 而且也可以是一个数组，指定每个关节的kp和kd
    kd = kargs.get('kd', 40) 
    parent_index = physics_info.parent_index
    joint_name = physics_info.joint_name
    # 注意关节没有自己的朝向和角速度，这里用子body的朝向和角速度表示此时关节的信息
    joint_orientation = physics_info.get_body_orientation()
    # print(physics_info.get_root_pos_and_vel())
    parent_index = physics_info.parent_index
    joint_avel = physics_info.get_body_angular_velocity()
    target_rotation = pose

    global_torque = np.zeros((20,3))
    # 你的代码

    n = joint_orientation.shape[0]
    R_joint = R.from_quat(joint_orientation[1:])
    R_parent = R.from_quat(joint_orientation[parent_index[1:]])
    joint_cur_rotation_inv = R_joint.inv() * R_parent
    joint_aim_rotation = R.from_quat(pose[1:])
    rotation_diff = (joint_aim_rotation * joint_cur_rotation_inv).as_euler('xyz', degrees=True)
    joint_cur_avel = joint_avel[1:]
    local_torque = kp * rotation_diff
    global_torque[1:] = R_parent.apply(local_torque) - kd * joint_cur_avel
    for i in range(n):
        if np.linalg.norm(global_torque[i]) > 1000:
            global_torque[i] /= 50
    
    # 抹去根节点力矩
    global_torque[0] = np.zeros_like(global_torque[0])
    return global_torque


def part2_cal_float_base_torque(target_position, pose, physics_info, **kargs):
    '''
    输入： target_position: (3,)的numpy数组，表示根节点的目标位置，其余同上
    输出： global_root_force: (3,)的numpy数组，表示根节点的全局坐标下的辅助力，在后续仿真中只会保留y方向的力
           global_root_torque: (3,)的numpy数组，表示根节点的全局坐标下的辅助力矩，用来控制角色的朝向，实际上就是global_torque的第0项
           global_torque: 同上
    注意：
        1. 你需要自己计算kp和kd，并且可以通过kargs调整part1中的kp和kd
        2. global_torque[0]在track静止姿态时会被无视，但是track走路时会被加到根节点上，不然无法保持根节点朝向
        3. 可以适当将根节点目标位置上提以产生更大的辅助力，使角色走得更自然
    '''
    # ------一些提示代码，你可以随意修改------------#
    global_torque = part1_cal_torque(pose, physics_info)
    kp = kargs.get('root_kp', 4000)
    kd = kargs.get('root_kd', 20)
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    global_root_force = np.zeros((3,))
    global_root_torque = global_torque[0]
    
    kp1 = kargs.get('kp', 1000) 
    kd1 = kargs.get('kd', 20) 
    joint_orientation = physics_info.get_body_orientation()
    joint_avel = physics_info.get_body_angular_velocity()

    global_error = np.zeros((20, 3))
    q_current = R.from_quat(joint_orientation[0])
    q_target = R.from_quat(pose[0])
    q_error = q_target * q_current.inv()
    global_error[0] = q_error.as_rotvec()  
    local_torque = kp1 * global_error[0] - kd1 * joint_avel[0]
    max_torque = 100
    local_torque = np.clip(local_torque, -max_torque, max_torque)
    global_torque[0] = local_torque
    global_root_torque = global_torque[0]
    
    
    target_pos = target_position.copy()
    # target_pos[1] += 0.1  # 上提，有助于角色保持站立时不沉下去
    # target_pos[1] += 0.05

    # PD控制
    position_error = target_pos - root_position
    velocity_error = -root_velocity
    global_root_force = kp * position_error + kd * velocity_error

    return global_root_force, global_root_torque, global_torque

frame_cnt = 0
def part3_cal_static_standing_torque(target_position, pose, physics_info):
    joint_name = physics_info.joint_name
    joint_positions = physics_info.get_joint_translation()
    joint_orientations = physics_info.get_body_orientation()
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    
    tar_pos = target_position.copy()
    tar_pos[1] += 0.05
    
    global frame_cnt
    frame_cnt += 1
    
    # print(tar_pos)
    
    torque = np.zeros((20,3))
    # 你的代码
    lower_body_joints = [
        'RootJoint', 
        'rHip', 'lHip', 
        'rKnee', 'lKnee', 
        'rAnkle', 'lAnkle', 
        'rToeJoint', 'lToeJoint'
    ]
    
    global_torque = part1_cal_torque(pose, physics_info, kp=400, kd=20) 
    # global_torque = part1_cal_torque(pose, physics_info, kp=200, kd=10) 

    Kp = 4000
    Kd = 20
    virtual_force = Kp * (tar_pos - root_position) - Kd * root_velocity
    
    torque = global_torque
    for i, name in enumerate(joint_name):
        if name in lower_body_joints:
            torque[i] -= np.cross( root_position - joint_positions[i], virtual_force)
    
    max_torque = 1000
    torque = np.clip(torque, -max_torque, max_torque)

    # 抹去根节点力矩
    torque[0] = np.zeros_like(torque[0])
    return torque


class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pose = self.get_pose(self.cnt)
        # print("================")
        # print(pose)
        # print(self.cnt)
        torque = part1_cal_torque(pose, self.physics_info)
        torque[0] = np.zeros_like(torque[0])
        self.viewer.set_torque(torque)
        self.cnt += 1
        pass

    def apply_root_force_and_torque(self):
        position, pose = self.get_pose(self.cnt)
        global_root_force, global_root_torque, torque = part2_cal_float_base_torque(position[0], pose, self.physics_info)
        self.viewer.set_torque(torque)
        # 只保留y方向的辅助力
        global_root_force[0] = 0.0
        global_root_force[2] = 0.0
        self.viewer.set_root_force(global_root_force)
        self.viewer.set_root_torque(global_root_torque)
        self.cnt += 1
        pass
    
    def apply_static_torque(self):
        position, pose = self.get_pose(self.cnt)
        torque = part3_cal_static_standing_torque(position[0], pose, self.physics_info)
        torque[0] = np.zeros_like(torque[0])
        self.viewer.set_torque(torque)
        self.viewer.add_horizontal_force()
        self.cnt += 1
        pass

class CharacterController():
    def __init__(self, viewer, controller, pd_controller) -> None:
        motion_list = [r"motion_material/idle.bvh", r"motion_material/walk.bvh"]
        motion = BVHMotion(motion_list[1])
        # set pose
        motion.adjust_joint_name(viewer.joint_name)
        joint_translation, joint_orientation = motion.batch_forward_kinematics(frame_id_list = [0], root_pos=viewer.root_pos)
        
        # viewer.add_noise_force = True
        viewer.set_pose(motion.joint_name, joint_translation[0], joint_orientation[0])
        viewer.pre_simulation_func = pd_controller.apply_static_torque

        # 手柄/键盘控制器
        self.controller = controller
        # pd controller
        idx_map = lambda x: (x//60)%motion.num_frames
        pd_controller.get_pose = lambda x: (motion.joint_position[idx_map(x)], motion.joint_rotation[idx_map(x)])
        # pd_controller.get_pose = lambda x: motion
        self.pd_controller = pd_controller
        # viewer 类，封装physics
        self.viewer = viewer
        # self.viewer.pre_simulation_func = pd_controller.apply_static_torque
        
        # motion
        self.motions = []
        # 添加motion
        self.motions.append(BVHMotion(bvh_file_name='./motion_material/idle.bvh'))
        
        # 下面是你可能会需要的成员变量，只是一个例子形式
        # 当然，你可以任意编辑，来符合你的要求
        # 当前角色的参考root位置
        self.cur_root_pos = None
        # 当前角色的参考root旋转
        self.cur_root_rot = None
        
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        

    
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
        
        # 一个简单的例子，循环播放第0个动画，不会响应输入信号
        motion = self.motions[0]
        motion.adjust_joint_name(self.viewer.joint_name)
        joint_name = motion.joint_name
        
        
        joint_translation, joint_orientation = motion.batch_forward_kinematics()
        joint_translation = joint_translation[self.cur_frame]
        joint_orientation = joint_orientation[self.cur_frame]
        
        self.cur_frame += 1
        if self.cur_frame >= motion.num_frames:
            self.cur_frame = 0
            
            
            
        # # --- 构造输入状态向量 ---
        # state_vec = np.concatenate([
        #     desired_pos_list[:, [0, 2]].reshape(-1),
        #     desired_rot_list[:, [1, 2, 3]].reshape(-1),
        # ])

        # # --- 使用强化学习策略网络输出动作 ---
        # action = self.policy.select_action(state_vec, deterministic=True)

        # # --- 将动作映射为局部姿态（rotation） ---
        # motion = self.motions[0].copy()
        # motion.set_pose_from_action(action)  # 你需要写一个函数来映射动作到 BVH pose结构

        # # --- FK 得到 global 坐标 ---
        # motion.adjust_joint_name(self.viewer.joint_name)
        # joint_name = motion.joint_name
        # joint_translation, joint_orientation = motion.batch_forward_kinematics()
        # joint_translation = joint_translation[0]
        # joint_orientation = joint_orientation[0]
        
        
        return joint_name, joint_translation, joint_orientation
    

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，你可以按需调整
        '''
        controller_pos = character_state[1][0]
        self.controller.set_pos(controller_pos)
    
