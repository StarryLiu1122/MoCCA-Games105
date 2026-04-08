# 以下部分均为可更改部分，你可以把需要的数据结构定义进来
from typing import List
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
from physics_warpper import PhysicsInfo
import numpy as np

class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pass

    def apply_root_force_and_torque(self):
        pass
    
    def apply_static_torque(self):
        pass

class CharacterController():
    def __init__(self, viewer, controller, pd_controller) -> None:
        # viewer 类，封装physics
        self.viewer = viewer
        # 手柄/键盘控制器
        self.controller = controller
        # pd controller
        self.pd_controller = pd_controller
        
        # 当前角色的参考root位置和旋转
        self.cur_root_pos = None
        self.cur_root_rot = None
        
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        
        self.cur_motion_idx = 0
        
        # motion
        self.motions = []
        # self.motions.append(BVHMotion(bvh_file_name='./motion_material/run.bvh'))
        self.motions = [
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk.bvh'),  # 长动作
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_run.bvh'),   # 长动作
        ]
        self.joint_translation, self.joint_orientation = self.motions[0].batch_forward_kinematics()

        # 预处理所有运动的特征
        self.motion_features = None
        self.preprocess_motion_features()
    
    
    def preprocess_motion_features(self):
        """预处理所有运动的特征数据"""
        self.motion_features = []
        
        for motion in self.motions:
            # 计算所有帧的全局位置和旋转
            joint_trans, joint_ori = motion.batch_forward_kinematics()
            num_frames = motion.num_frames
            
            # 为每帧提取特征
            features = []
            for i in range(num_frames):
                # 根节点特征
                root_pos = joint_trans[i][0]  # 根节点位置
                root_rot = R.from_quat(joint_ori[i][0])  # 根节点旋转
                
                # 计算速度和角速度(有限差分)
                if i < num_frames - 1:
                    next_pos = joint_trans[i+1][0]
                    next_rot = R.from_quat(joint_ori[i+1][0])
                    vel = (next_pos - root_pos) / 0.01
                    avel = (next_rot * root_rot.inv()).as_rotvec() / 0.01
                else:
                    vel = np.zeros(3)
                    avel = np.zeros(3)
                
                # 脚部特征
                left_foot_pos = joint_trans[i][9]  # 左脚位置
                right_foot_pos = joint_trans[i][10] # 右脚位置
                
                # 存储特征向量
                feature = {
                    'root_pos': root_pos,
                    'root_rot': root_rot,
                    'velocity': vel,
                    'angular_vel': avel,
                    'left_foot_pos': left_foot_pos,
                    'right_foot_pos': right_foot_pos
                }
                features.append(feature)
            
            self.motion_features.append(features)
            
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
        # motion = self.motions[0]
        # motion.adjust_joint_name(self.viewer.joint_name)
        # joint_name = motion.joint_name
        
        # joint_translation, joint_orientation = motion.batch_forward_kinematics()
        # joint_translation = joint_translation[self.cur_frame]
        # joint_orientation = joint_orientation[self.cur_frame]
        
        # self.cur_frame += 1
        # if self.cur_frame >= motion.num_frames:
        #     self.cur_frame = 0
        
        
        motion = self.motions[0] 
        motion.adjust_joint_name(self.viewer.joint_name)
        joint_name = motion.joint_name
        # print(joint_name)
        
        current_pos = desired_pos_list[0] 
        current_vel = desired_vel_list[0]  
        current_rot = R.from_quat(desired_rot_list[0])
        current_avel = desired_avel_list[0][1]   # 绕Y轴的角速度

        
        # 1. 运动选择 - 根据输入速度选择走路或跑步
        speed = np.linalg.norm(desired_vel_list[5]) 
        if speed > 2.0:  # 跑步阈值
            motion_idx = 1  # 跑步
        elif speed > 0.5:  # 走路阈值
            motion_idx = 0  # 走路
        else:
            motion_idx = self.cur_motion_idx  # 保持当前运动
            
        # 检查是否需要转向
        turn_threshold = 0.5  # 角速度阈值
        if desired_avel_list[5][1] > turn_threshold:
            motion_idx = 3  # 右转
        elif desired_avel_list[5][1] < -turn_threshold:
            motion_idx = 2  # 左转
            
        # 2. 运动匹配 - 在当前运动中找到最佳帧
        best_frame = self.find_best_match(
            motion_idx,
            desired_pos_list[5],
            desired_vel_list[5],
            desired_rot_list[5],
            desired_avel_list[5]
        )
        
        # 更新当前状态
        self.cur_motion_idx = motion_idx
        self.cur_frame = best_frame
        
        # 获取匹配的姿势
        motion = self.motions[motion_idx]
        motion.adjust_joint_name(self.viewer.joint_name)
        joint_translation, joint_orientation = motion.batch_forward_kinematics()
        
        # 3. 根节点调整 - 将匹配姿势的根节点位置和旋转与目标对齐
        matched_root_pos = joint_translation[best_frame][0]
        matched_root_rot = R.from_quat(joint_orientation[best_frame][0])
        
        # 计算位置和旋转偏移
        pos_offset = desired_pos_list[5] - matched_root_pos
        rot_diff = R.from_quat(desired_rot_list[5]) * matched_root_rot.inv()
        
        # 应用偏移到整个骨骼
        for i in range(len(joint_translation[best_frame])):
            # 位置调整
            joint_translation[best_frame][i] += pos_offset
            
            # 旋转调整(只调整Y轴旋转)
            joint_rot = R.from_quat(joint_orientation[best_frame][i])
            adjusted_rot = rot_diff * joint_rot
            joint_orientation[best_frame][i] = adjusted_rot.as_quat()
            
        
        joint_translation = self.joint_translation[self.cur_frame]
        joint_orientation = self.joint_orientation[self.cur_frame]
        
        return joint_name, joint_translation, joint_orientation
    
    
    def find_best_match(self, motion_idx, target_pos, target_vel, target_rot, target_avel):
        """在指定运动中寻找最佳匹配帧"""
        features = self.motion_features[motion_idx]
        min_cost = float('inf')
        best_frame = self.cur_frame  # 默认从当前帧开始
        
        # 搜索范围(当前帧前后120帧)
        start = max(0, self.cur_frame - 60)
        end = min(len(features), self.cur_frame + 60)
        
        for i in range(start, end):
            frame_feat = features[i]
            
            # 计算各项特征代价(加权)
            pos_cost = 1.0 * np.linalg.norm(target_pos - frame_feat['root_pos'])
            vel_cost = 2.0 * np.linalg.norm(target_vel - frame_feat['velocity'])
            # tarrot = R.from_quat(target_rot)
            # currot = frame_feat['root_rot']
            # rot_diff = tarrot.inv() * currot
            # rot_cost = 0.5 * abs(rot_diff.as_euler('XYZ')[1])
            rot_cost = 0.5 * abs((R.from_quat(target_rot).inv() * frame_feat['root_rot']).as_euler('XYZ')[1])
            avel_cost = 0.5 * abs(target_avel[1] - frame_feat['angular_vel'][1])
            
            # 脚部代价(防止滑步)
            left_foot_cost = 0.3 * np.linalg.norm(frame_feat['left_foot_pos'][1])  # Y坐标(越低越好)
            right_foot_cost = 0.3 * np.linalg.norm(frame_feat['right_foot_pos'][1])
            
            # 连续性代价(鼓励选择接近当前帧的帧)
            continuity_cost = 0.1 * abs(i - self.cur_frame) / 60.0
            
            total_cost = (
                pos_cost + vel_cost + rot_cost + avel_cost +
                left_foot_cost + right_foot_cost +
                continuity_cost
            )
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_frame = i
        
        return best_frame

    def _compute_velocity(self, motion, frame, dt=0.016667):
        """通过有限差分计算根节点速度（XoZ平面）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_pos = motion.joint_position[frame + 1][0]
        current_pos = motion.joint_position[frame][0]
        return (next_pos - current_pos) / dt

    def _compute_angular_velocity(self, motion, frame, dt=0.016667):
        """通过有限差分计算根节点角速度（绕Y轴）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_rot = R.from_quat(motion.joint_rotation[frame + 1][0])
        current_rot = R.from_quat(motion.joint_rotation[frame][0])
        delta_rot = next_rot * current_rot.inv()
        return delta_rot.as_rotvec() / dt
    

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，你可以按需调整
        '''
        controller_pos = character_state[1][0]
        self.controller.set_pos(controller_pos)
    




























# 以下部分均为可更改部分，你可以把需要的数据结构定义进来
from typing import List
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
from physics_warpper import PhysicsInfo
import numpy as np
from smooth_utils import *

def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, mix_frame1:int, mix_time:int):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第mix_frame1帧
          mix_time表示用于混合的帧数
    输出: 平滑地连接后的动作，类型为BVHMotion
    '''
    res = bvh_motion1.raw_copy()
    
    # inertialization
    # 对齐
    pos = bvh_motion1.joint_position[mix_frame1, 0, [0, 2]]
    rot = bvh_motion1.joint_rotation[mix_frame1, 0]
    facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
    new_bvh_motion2 = bvh_motion2.translation_and_rotation(0, pos, facing_axis)

    half_life = 0.2
    fps = 60
    dt = 1 / fps

    motion1_avel = quat_to_avel(bvh_motion1.joint_rotation[mix_frame1 - int(mix_time / 2):mix_frame1], dt)
    motion2_avel = quat_to_avel(new_bvh_motion2.joint_rotation[0:int(mix_time / 2)], dt)
    avel_diff = motion1_avel[-1] - motion2_avel[0]
    rot_diff = (R.from_quat(bvh_motion1.joint_rotation[mix_frame1])*R.from_quat(new_bvh_motion2.joint_rotation[0].copy()).inv()).as_rotvec()
    
    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_rot(
            rot_diff, avel_diff, half_life, i / fps
        )
        offset_rot = R.from_rotvec(offset[0])
        new_bvh_motion2.joint_rotation[i] = (offset_rot * R.from_quat(new_bvh_motion2.joint_rotation[i])).as_quat() 
        
    motion1_vel = bvh_motion1.joint_position[mix_frame1] - bvh_motion1.joint_position[mix_frame1-1]
    motion2_vel = new_bvh_motion2.joint_position[1]-new_bvh_motion2.joint_position[0]
    vel_diff = (motion1_vel - motion2_vel) / fps
    pos_diff = bvh_motion1.joint_position[mix_frame1] - new_bvh_motion2.joint_position[0]

    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_pos(
            pos_diff, vel_diff, half_life, i / fps
        )
        offset_pos = offset[0]
        new_bvh_motion2.joint_position[i] += offset_pos

    res.joint_position = np.concatenate([res.joint_position[:mix_frame1],  new_bvh_motion2.joint_position], axis=0)
    res.joint_rotation = np.concatenate([res.joint_rotation[:mix_frame1],  new_bvh_motion2.joint_rotation], axis=0)
    return res


class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pass

    def apply_root_force_and_torque(self):
        pass
    
    def apply_static_torque(self):
        pass

class CharacterController():
    def __init__(self, viewer, controller, pd_controller) -> None:
        # viewer 类，封装physics
        self.viewer = viewer
        # 手柄/键盘控制器
        self.controller = controller
        # pd controller
        self.pd_controller = pd_controller
        
        # 当前角色的参考root位置和旋转
        self.cur_root_pos = None
        self.cur_root_rot = None
        
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        self.cur_motion_idx = 0
        
        
        # 动作转换
        self.is_transition = False
        self.transition_frame = 0
        self.max_transition_frame = 30
        
        # motion
        # 短motion
        self.motion = []
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/walk.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/run.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/idle.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/jump.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/turn_left.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/turn_right.bvh'))
        # 长motion
        self.motions = []
        self.motions = [
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk.bvh'),  # 长动作
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_run.bvh'),   # 长动作
        ]
        self.joint_translation, self.joint_orientation = self.motions[0].batch_forward_kinematics()
        self.cur_motion = self.motion[0]
        self.cur_motion.adjust_joint_name(self.viewer.joint_name)
        self.joint_name = self.cur_motion.joint_name
        
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_motion.motion_length
        
        # 预处理所有运动的特征
        self.motion_features = None
        self.preprocess_motion_features()
    
    
    def preprocess_motion_features(self):
        """预处理所有运动的特征数据"""
        self.motion_features = []
        
        # 处理短motion
        for motion in self.motion:
            # 计算所有帧的全局位置和旋转
            joint_trans, joint_ori = motion.batch_forward_kinematics()
            num_frames = motion.num_frames
            
            # 为每帧提取特征
            features = []
            for i in range(num_frames):
                root_pos = joint_trans[i][0]  # 根节点位置
                
                # 计算速度
                if i < num_frames - 1:
                    root_vel = (joint_trans[i+1][0] - root_pos) / 0.01
                    left_foot_vel = (joint_trans[i+1][9] - joint_trans[i][9]) / 0.01
                    right_foot_vel = (joint_trans[i+1][10] - joint_trans[i][10]) / 0.01
                else:
                    continue
                
                # 脚部位置
                left_foot_pos = joint_trans[i][9] - root_pos   # 左脚局部位置
                right_foot_pos = joint_trans[i][10] - root_pos # 右脚局部位置
                
                # 存储特征向量
                feature = {
                    'root_vel': root_vel,
                    'left_foot_pos': left_foot_pos,
                    'right_foot_pos': right_foot_pos,
                    'left_foot_vel': left_foot_vel,
                    'right_foot_vel': right_foot_vel
                }
                features.append(feature)
            
            self.motion_features.append(features)
            
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
        # motion = self.motion[1]
        # motion.adjust_joint_name(self.viewer.joint_name)
        # joint_name = motion.joint_name
        
        # joint_translation, joint_orientation = motion.batch_forward_kinematics()
        # joint_translation = joint_translation[self.cur_frame]
        # joint_orientation = joint_orientation[self.cur_frame]
        # self.cur_root_pos = joint_translation[0]
        
        # self.cur_frame += 1
        # if self.cur_frame >= motion.num_frames:
        #     self.cur_frame = 0
        
        # print(self.cur_root_pos)
        # print(desired_vel_list[5])
        # print(desired_avel_list[5][1])
        
        joint_name = self.joint_name
        
        # 正在转移动作
        if self.is_transition and self.transition_frame < self.max_transition_frame: 
            # 直接更新数据
            self.transition_frame += 1
            self.cur_frame = self.transition_frame
            self.cur_root_pos = self.joint_translation[self.transition_frame][0]
            self.cur_root_rot = self.joint_orientation[self.transition_frame][0]
            return joint_name, self.joint_translation[self.transition_frame], self.joint_orientation[self.transition_frame]
        elif self.is_transition and self.transition_frame >= self.max_transition_frame: 
            self.is_transition = False
            self.transition_frame = 0
        
        # 1. 运动选择:根据输入速度选择走路或跑步
        motion_idx = 0
        speed = np.linalg.norm(desired_vel_list[5]) 
        if speed > 1.5:  # 跑步阈值
            motion_idx = 1  # 跑步
        else:
            motion_idx = 0  # 走路
        # 检查是否需要转向
        if self.cur_frame > 0:
            motion_idx = self.choose_dir_idx(desired_rot_list)
        
        # 如果动作没有发生变化
        if motion_idx == self.cur_motion_idx:
            if self.cur_frame < self.cur_end_frame:
                self.joint_translation, self.joint_orientation = self.cur_motion.batch_forward_kinematics()  # 可以改成预处理
                joint_translation = self.joint_translation[self.cur_frame]
                joint_orientation = self.joint_orientation[self.cur_frame]
                self.cur_root_pos = joint_translation[0]
                self.cur_root_rot = joint_orientation[0]
                self.cur_frame += 1
                return joint_name, joint_translation, joint_orientation
            else:
                self.cur_frame = 0
                
                facing_axis = R.from_quat(self.cur_root_rot).apply(np.array([0,0,1])).flatten()[[0,2]]
                next_motion = self.cur_motion.translation_and_rotation(0, self.cur_root_pos[[0,2]], facing_axis)
                smooth_motion = concatenate_two_motions(self.cur_motion, next_motion, self.cur_end_frame-1, 30)
                smooth_motion.joint_position = smooth_motion.joint_position[self.cur_end_frame:,:,:]
                smooth_motion.joint_rotation = smooth_motion.joint_rotation[self.cur_end_frame:,:,:]
                self.cur_motion = smooth_motion
                self.cur_end_frame = self.cur_motion.motion_length

                self.joint_translation, self.joint_orientation = self.cur_motion.batch_forward_kinematics()
                joint_translation = self.joint_translation[0]
                joint_orientation = self.joint_orientation[0]
                self.cur_root_pos = joint_translation[0]
                self.cur_root_rot = joint_orientation[0]
        
        # 动作变化
        else: 
            # 运动匹配 - 在当前运动中找到最佳帧
            best_frame = self.find_best_match(
                motion_idx,
                desired_pos_list[5],
                desired_vel_list[5],
                desired_rot_list[5],
                desired_avel_list[5]
            )
                
            # 动作拼接
            facing_axis = R.from_quat(self.cur_root_rot).apply(np.array([0,0,1])).flatten()[[0,2]]
            next_motion = self.motion[motion_idx][best_frame:].translation_and_rotation(0, self.cur_root_pos[[0,2]], facing_axis)
                
            smooth_motion = concatenate_two_motions(self.cur_motion, next_motion, self.cur_frame, 30)
            smooth_motion.joint_position = smooth_motion.joint_position[self.cur_frame+1:,:,:]
            smooth_motion.joint_rotation = smooth_motion.joint_rotation[self.cur_frame+1:,:,:]
            self.cur_motion = smooth_motion
            self.cur_end_frame = self.cur_motion.motion_length

            self.joint_translation, self.joint_orientation = self.cur_motion.batch_forward_kinematics()
            joint_translation = self.joint_translation[0]
            joint_orientation = self.joint_orientation[0]
            self.cur_root_pos = self.joint_translation[0]
            self.cur_root_rot = self.joint_orientation[0]
            self.is_transition = True
            self.transition_frame = 0
                
            # 更新当前状态
            self.cur_motion_idx = motion_idx
            self.cur_frame = best_frame
             
        return joint_name, joint_translation, joint_orientation
    
    def choose_dir_idx(self, desired_rot_list): 
        current_rot = R.from_quat(self.cur_root_rot)
        desired_rot = R.from_quat(desired_rot_list[-1])
        diff_rot = desired_rot * current_rot.inv()
        delta_angle = diff_rot.as_rotvec()[1]

        if delta_angle > np.pi * 1 / 6:
            idx = 5 # 右转
        elif delta_angle < -np.pi * 1 / 6:
            idx = 4 # 左转  
        else:
            idx = self.cur_motion_idx  
        return idx                             
    
    def find_best_match(self, motion_idx, target_pos, target_vel, target_rot, target_avel):
        """在指定运动中寻找最佳匹配帧"""
        features = self.motion_features[motion_idx]
        min_cost = float('inf')
        best_frame = 0
        
        for i in range(0, len(features)):
            frame_feat = features[i]
            
            
            future_cost = (
                np.linalg.norm(frame_feat['left_foot_pos'] - target_pos[9]) 
                + np.linalg.norm(frame_feat['right_foot_pos'] - target_pos[10]) 
                + np.linalg.norm(frame_feat['left_foot_vel'] - target_vel[9]) 
                + np.linalg.norm(frame_feat['right_foot_vel'] - target_vel[10]) 
                + np.linalg.norm(frame_feat['root_vel'] - target_vel[0]) 
            )
            
            total_cost = future_cost
            # total_cost = future_cost + current_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_frame = i
        
        return best_frame

    def _compute_velocity(self, motion, frame, dt=0.016667):
        """通过有限差分计算根节点速度（XoZ平面）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_pos = motion.joint_position[frame + 1][0]
        current_pos = motion.joint_position[frame][0]
        return (next_pos - current_pos) / dt

    def _compute_angular_velocity(self, motion, frame, dt=0.016667):
        """通过有限差分计算根节点角速度（绕Y轴）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_rot = R.from_quat(motion.joint_rotation[frame + 1][0])
        current_rot = R.from_quat(motion.joint_rotation[frame][0])
        delta_rot = next_rot * current_rot.inv()
        return delta_rot.as_rotvec() / dt
    

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，你可以按需调整
        '''
        controller_pos = character_state[1][0]
        self.controller.set_pos(controller_pos)
    






















# 以下部分均为可更改部分，你可以把需要的数据结构定义进来
from typing import List
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
from physics_warpper import PhysicsInfo
import numpy as np
from smooth_utils import *

def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, mix_frame1:int, mix_time:int):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第mix_frame1帧
          mix_time表示用于混合的帧数
    输出: 平滑地连接后的动作，类型为BVHMotion
    '''
    res = bvh_motion1.raw_copy()
    
    # inertialization
    # 对齐
    pos = bvh_motion1.joint_position[mix_frame1, 0, [0, 2]]
    rot = bvh_motion1.joint_rotation[mix_frame1, 0]
    facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
    new_bvh_motion2 = bvh_motion2.translation_and_rotation(0, pos, facing_axis)

    half_life = 0.2
    fps = 60
    dt = 1 / fps

    motion1_avel = quat_to_avel(bvh_motion1.joint_rotation[mix_frame1 - int(mix_time / 2):mix_frame1], dt)
    motion2_avel = quat_to_avel(new_bvh_motion2.joint_rotation[0:int(mix_time / 2)], dt)
    avel_diff = motion1_avel[-1] - motion2_avel[0]
    rot_diff = (R.from_quat(bvh_motion1.joint_rotation[mix_frame1])*R.from_quat(new_bvh_motion2.joint_rotation[0].copy()).inv()).as_rotvec()
    
    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_rot(
            rot_diff, avel_diff, half_life, i / fps
        )
        offset_rot = R.from_rotvec(offset[0])
        new_bvh_motion2.joint_rotation[i] = (offset_rot * R.from_quat(new_bvh_motion2.joint_rotation[i])).as_quat() 
        
    motion1_vel = bvh_motion1.joint_position[mix_frame1] - bvh_motion1.joint_position[mix_frame1-1]
    motion2_vel = new_bvh_motion2.joint_position[1]-new_bvh_motion2.joint_position[0]
    vel_diff = (motion1_vel - motion2_vel) / fps
    pos_diff = bvh_motion1.joint_position[mix_frame1] - new_bvh_motion2.joint_position[0]

    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_pos(
            pos_diff, vel_diff, half_life, i / fps
        )
        offset_pos = offset[0]
        new_bvh_motion2.joint_position[i] += offset_pos

    res.joint_position = np.concatenate([res.joint_position[:mix_frame1],  new_bvh_motion2.joint_position], axis=0)
    res.joint_rotation = np.concatenate([res.joint_rotation[:mix_frame1],  new_bvh_motion2.joint_rotation], axis=0)
    return res


class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pass

    def apply_root_force_and_torque(self):
        pass
    
    def apply_static_torque(self):
        pass

class CharacterController():
    def __init__(self, viewer, controller, pd_controller) -> None:
        # viewer 类，封装physics
        self.viewer = viewer
        # 手柄/键盘控制器
        self.controller = controller
        # pd controller
        self.pd_controller = pd_controller
        
        # 当前角色的参考root位置和旋转
        self.cur_root_pos = None
        self.cur_root_rot = None
        
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        self.cur_motion_idx = 0
        
        
        # 动作转换
        self.is_transition = False
        self.transition_frame = 0
        self.max_transition_frame = 30
        
        # motion
        # 短motion
        self.motion = []
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/walk.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/run.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/idle.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/jump.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/turn_left.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/turn_right.bvh'))
        # 长motion
        self.motions = []
        self.motions = [
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk.bvh'),  # 长动作
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_run.bvh'),   # 长动作
        ]
        self.joint_translation, self.joint_orientation = self.motions[0].batch_forward_kinematics()
        self.cur_motion = self.motion[0]
        self.cur_motion.adjust_joint_name(self.viewer.joint_name)
        self.joint_name = self.cur_motion.joint_name
        
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_motion.motion_length
        
        # 预处理所有运动的特征
        self.motion_features = None
        self.preprocess_motion_features()
    
    
    def preprocess_motion_features(self):
        """预处理所有运动的特征数据"""
        self.motion_features = []
        
        # 处理短motion
        for motion in self.motion:
            # 计算所有帧的全局位置和旋转
            joint_trans, joint_ori = motion.batch_forward_kinematics()
            num_frames = motion.num_frames
            
            # 为每帧提取根节点特征
            features = []
            for i in range(num_frames):
                root_pos = joint_trans[i][0] 
                root_rot = joint_ori[i][0]
                
                # 计算速度
                if i < num_frames - 1:
                    root_vel = (joint_trans[i+1][0] - root_pos) / 0.01
                else:
                    continue
                
                # 计算角速度
                avel = quat_to_avel(joint_ori[i], 0.01)
                root_avel = avel[0]
                
                # 存储特征向量
                feature = {
                    'root_pos': root_pos,
                    'root_vel': root_vel,
                    'root_rot': root_rot,
                    'root_avel': root_avel,
                }
                features.append(feature)
            
            self.motion_features.append(features)
            
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
        # motion = self.motion[1]
        # motion.adjust_joint_name(self.viewer.joint_name)
        # joint_name = motion.joint_name
        
        # joint_translation, joint_orientation = motion.batch_forward_kinematics()
        # joint_translation = joint_translation[self.cur_frame]
        # joint_orientation = joint_orientation[self.cur_frame]
        # self.cur_root_pos = joint_translation[0]
        
        # self.cur_frame += 1
        # if self.cur_frame >= motion.num_frames:
        #     self.cur_frame = 0
        
        # print(self.cur_root_pos)
        # print(desired_vel_list[5])
        # print(desired_avel_list[5][1])
        
        joint_name = self.joint_name
        
        # 正在转移动作
        if self.is_transition and self.transition_frame < self.max_transition_frame: 
            # 直接更新数据
            self.transition_frame += 1
            self.cur_frame = self.transition_frame
            self.cur_root_pos = self.joint_translation[self.transition_frame][0]
            self.cur_root_rot = self.joint_orientation[self.transition_frame][0]
            return joint_name, self.joint_translation[self.transition_frame], self.joint_orientation[self.transition_frame]
        elif self.is_transition and self.transition_frame >= self.max_transition_frame: 
            self.is_transition = False
            self.transition_frame = 0
        
        # 1. 运动选择:根据输入速度选择走路或跑步
        motion_idx = 0
        speed = np.linalg.norm(desired_vel_list[5]) 
        if speed > 1.5:  # 跑步阈值
            motion_idx = 1  # 跑步
        else:
            motion_idx = 0  # 走路
        # 检查是否需要转向
        if self.cur_frame > 0:
            motion_idx = self.choose_dir_idx(desired_rot_list)
        
        # 如果动作没有发生变化
        if motion_idx == self.cur_motion_idx:
            if self.cur_frame < self.cur_end_frame:
                self.joint_translation, self.joint_orientation = self.cur_motion.batch_forward_kinematics()  # 可以改成预处理
                joint_translation = self.joint_translation[self.cur_frame]
                joint_orientation = self.joint_orientation[self.cur_frame]
                self.cur_root_pos = joint_translation[0]
                self.cur_root_rot = joint_orientation[0]
                self.cur_frame += 1
                return joint_name, joint_translation, joint_orientation
            else:
                self.cur_frame = 0
                
                facing_axis = R.from_quat(self.cur_root_rot).apply(np.array([0,0,1])).flatten()[[0,2]]
                next_motion = self.cur_motion.translation_and_rotation(0, self.cur_root_pos[[0,2]], facing_axis)
                smooth_motion = concatenate_two_motions(self.cur_motion, next_motion, self.cur_end_frame-1, 30)
                smooth_motion.joint_position = smooth_motion.joint_position[self.cur_end_frame:,:,:]
                smooth_motion.joint_rotation = smooth_motion.joint_rotation[self.cur_end_frame:,:,:]
                self.cur_motion = smooth_motion
                self.cur_end_frame = self.cur_motion.motion_length

                self.joint_translation, self.joint_orientation = self.cur_motion.batch_forward_kinematics()
                joint_translation = self.joint_translation[0]
                joint_orientation = self.joint_orientation[0]
                self.cur_root_pos = joint_translation[0]
                self.cur_root_rot = joint_orientation[0]
        
        # 动作变化
        else: 
            # 运动匹配 - 在当前运动中找到最佳帧
            best_frame = self.find_best_match(
                motion_idx,
                desired_pos_list[5],
                desired_vel_list[5],
                desired_rot_list[5],
                desired_avel_list[5]
            )
                
            # 动作拼接
            facing_axis = R.from_quat(self.cur_root_rot).apply(np.array([0,0,1])).flatten()[[0,2]]
            next_motion = self.motion[motion_idx].raw_copy()
            next_motion.joint_position = self.motion[motion_idx].joint_position[best_frame:]
            next_motion.joint_rotation = self.motion[motion_idx].joint_rotation[best_frame:]
            next_motion = next_motion.translation_and_rotation(0, self.cur_root_pos[[0,2]], facing_axis)
                
            smooth_motion = concatenate_two_motions(self.cur_motion, next_motion, self.cur_frame, 30)
            smooth_motion.joint_position = smooth_motion.joint_position[self.cur_frame+1:,:,:]
            smooth_motion.joint_rotation = smooth_motion.joint_rotation[self.cur_frame+1:,:,:]
            self.cur_motion = smooth_motion
            self.cur_end_frame = self.cur_motion.motion_length

            self.joint_translation, self.joint_orientation = self.cur_motion.batch_forward_kinematics()
            joint_translation = self.joint_translation[0]
            joint_orientation = self.joint_orientation[0]
            self.cur_root_pos = self.joint_translation[0]
            self.cur_root_rot = self.joint_orientation[0]
            self.is_transition = True
            self.transition_frame = 0
                
            # 更新当前状态
            self.cur_motion_idx = motion_idx
            self.cur_frame = best_frame
             
        return joint_name, joint_translation, joint_orientation
    
    def choose_dir_idx(self, desired_rot_list): 
        current_rot = R.from_quat(self.cur_root_rot)
        desired_rot = R.from_quat(desired_rot_list[-1])
        diff_rot = desired_rot * current_rot.inv()
        delta_angle = diff_rot.as_rotvec()[1]

        if delta_angle > np.pi * 1 / 6:
            idx = 5 # 右转
        elif delta_angle < -np.pi * 1 / 6:
            idx = 4 # 左转  
        else:
            idx = self.cur_motion_idx  
        return idx                             
    
    def find_best_match(self, motion_idx, target_pos, target_vel, target_rot, target_avel):
        """在指定运动中寻找最佳匹配帧"""
        features = self.motion_features[motion_idx]
        min_cost = float('inf')
        best_frame = 0
        
        for i in range(0, len(features)):
            frame_feat = features[i]
            
            future_cost = (
                np.linalg.norm(frame_feat['root_pos'] - target_pos) 
                + np.linalg.norm(frame_feat['root_vel'] - target_vel) 
                + np.linalg.norm(frame_feat['root_rot'] - target_rot) 
                + np.linalg.norm(frame_feat['root_avel'] - target_avel) 
            )
            
            total_cost = future_cost
            # total_cost = future_cost + current_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_frame = i
        
        return best_frame

    def _compute_velocity(self, motion, frame, dt=0.016667):
        """通过有限差分计算根节点速度（XoZ平面）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_pos = motion.joint_position[frame + 1][0]
        current_pos = motion.joint_position[frame][0]
        return (next_pos - current_pos) / dt

    def _compute_angular_velocity(self, motion, frame, dt=0.016667):
        """通过有限差分计算根节点角速度（绕Y轴）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_rot = R.from_quat(motion.joint_rotation[frame + 1][0])
        current_rot = R.from_quat(motion.joint_rotation[frame][0])
        delta_rot = next_rot * current_rot.inv()
        return delta_rot.as_rotvec() / dt
    

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，你可以按需调整
        '''
        controller_pos = character_state[1][0]
        self.controller.set_pos(controller_pos)
    








# 以下部分均为可更改部分，你可以把需要的数据结构定义进来
from typing import List
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
from physics_warpper import PhysicsInfo
import numpy as np
from smooth_utils import *

def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, mix_frame1:int, mix_time:int):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第mix_frame1帧
          mix_time表示用于混合的帧数
    输出: 平滑地连接后的动作，类型为BVHMotion
    '''
    res = bvh_motion1.raw_copy()
    
    # inertialization
    # 对齐
    pos = bvh_motion1.joint_position[mix_frame1, 0, [0, 2]]
    rot = bvh_motion1.joint_rotation[mix_frame1, 0]
    facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
    new_bvh_motion2 = bvh_motion2.translation_and_rotation(0, pos, facing_axis)

    half_life = 0.2
    fps = 60
    dt = 1 / fps

    motion1_avel = quat_to_avel(bvh_motion1.joint_rotation[mix_frame1 - int(mix_time / 2):mix_frame1], dt)
    motion2_avel = quat_to_avel(new_bvh_motion2.joint_rotation[0:int(mix_time / 2)], dt)
    avel_diff = motion1_avel[-1] - motion2_avel[0]
    rot_diff = (R.from_quat(bvh_motion1.joint_rotation[mix_frame1])*R.from_quat(new_bvh_motion2.joint_rotation[0].copy()).inv()).as_rotvec()
    
    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_rot(
            rot_diff, avel_diff, half_life, i / fps
        )
        offset_rot = R.from_rotvec(offset[0])
        new_bvh_motion2.joint_rotation[i] = (offset_rot * R.from_quat(new_bvh_motion2.joint_rotation[i])).as_quat() 
        
    motion1_vel = bvh_motion1.joint_position[mix_frame1] - bvh_motion1.joint_position[mix_frame1-1]
    motion2_vel = new_bvh_motion2.joint_position[1]-new_bvh_motion2.joint_position[0]
    vel_diff = (motion1_vel - motion2_vel) / fps
    pos_diff = bvh_motion1.joint_position[mix_frame1] - new_bvh_motion2.joint_position[0]

    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_pos(
            pos_diff, vel_diff, half_life, i / fps
        )
        offset_pos = offset[0]
        new_bvh_motion2.joint_position[i] += offset_pos

    res.joint_position = np.concatenate([res.joint_position[:mix_frame1],  new_bvh_motion2.joint_position], axis=0)
    res.joint_rotation = np.concatenate([res.joint_rotation[:mix_frame1],  new_bvh_motion2.joint_rotation], axis=0)
    return res


class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pass

    def apply_root_force_and_torque(self):
        pass
    
    def apply_static_torque(self):
        pass

class CharacterController():
    def __init__(self, viewer, controller, pd_controller) -> None:
        # viewer 类，封装physics
        self.viewer = viewer
        # 手柄/键盘控制器
        self.controller = controller
        # pd controller
        self.pd_controller = pd_controller
          
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        self.cur_motion_idx = 0
        
        
        # 动作转换
        self.is_transition = False
        self.transition_frame = 0
        self.max_transition_frame = 30
        
        # motion
        # 短motion
        self.motion = []
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/walk.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/run.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/idle.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/jump.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/turn_left.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/turn_right.bvh'))
        # 长motion
        self.motions = []
        self.motions = [
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk.bvh'),  # 长动作
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_run.bvh'),   # 长动作
        ]
        self.joint_translation, self.joint_orientation = self.motions[0].batch_forward_kinematics()
        # 当前角色的参考root位置和旋转
        self.cur_root_pos = self.joint_translation[0][0]
        self.cur_root_rot = self.joint_orientation[0][0]
        
        self.cur_motion = self.motion[0]
        self.cur_motion.adjust_joint_name(self.viewer.joint_name)
        self.joint_name = self.cur_motion.joint_name
        
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_motion.motion_length
        
        # 预处理所有运动的特征
        self.motion_features = None
        self.preprocess_motion_features()
    
    
    def preprocess_motion_features(self):
        """预处理所有运动的特征数据"""
        self.motion_features = []
        for motion in self.motion:
            joint_trans, joint_ori = motion.batch_forward_kinematics()
            num_frames = motion.num_frames
            features = []
            
            for i in range(num_frames):
                # 当前帧特征
                root_pos = joint_trans[i][0]
                root_rot = joint_ori[i][0]
                root_vel = self._compute_velocity(motion, i) if i < num_frames - 1 else np.zeros(3)
                root_avel = self._compute_angular_velocity(motion, i) if i < num_frames - 1 else np.zeros(3)
                
                # 未来轨迹特征（20/40/60帧）
                future_pos = []
                future_rot = []
                for offset in [20, 40, 60]:
                    future_frame = min(i + offset, num_frames - 1)
                    future_pos.append(joint_trans[future_frame][0][[0, 2]])  # XZ平面
                    future_dir = R.from_quat(joint_ori[future_frame][0]).apply(np.array([0, 0, 1]))[[0, 2]]  # 前方方向
                    future_rot.append(future_dir)
                
                features.append({
                    'root_pos': root_pos[[0, 2]],  # XZ平面
                    'root_vel': root_vel[[0, 2]],
                    'root_rot': R.from_quat(root_rot).as_rotvec()[1],  # 绕Y轴旋转角度
                    'root_avel': root_avel[1],  # 绕Y轴角速度
                    'future_pos': np.concatenate(future_pos),
                    'future_rot': np.concatenate(future_rot),
                })
            self.motion_features.append(features)
            
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
        # motion = self.motion[1]
        # motion.adjust_joint_name(self.viewer.joint_name)
        # joint_name = motion.joint_name
        
        # joint_translation, joint_orientation = motion.batch_forward_kinematics()
        # joint_translation = joint_translation[self.cur_frame]
        # joint_orientation = joint_orientation[self.cur_frame]
        # self.cur_root_pos = joint_translation[0]
        
        # self.cur_frame += 1
        # if self.cur_frame >= motion.num_frames:
        #     self.cur_frame = 0
        
        # print(self.cur_root_pos)
        # print(desired_vel_list[5])
        # print(desired_avel_list[5][1])
        
        joint_name = self.joint_name
        
        # 1. 生成目标特征
        target_pos = desired_pos_list[-1] 
        target_vel = desired_vel_list[-1]
        target_rot = desired_rot_list[-1]
        target_avel = desired_avel_list[-1]
        
        # 2. 选择动作类型（行走/跑步）
        motion_idx = 0 if np.linalg.norm(target_vel) < 1.5 else 1
        
        # 3. 搜索最佳帧
        best_frame = self.find_best_match(motion_idx, target_pos, target_vel, target_rot, target_avel)
        
        # 4. 混合动画
        if motion_idx != self.cur_motion_idx or self.cur_frame >= self.cur_end_frame:
            self.transition_to_new_motion(motion_idx, best_frame)
        
        # 5. 返回当前帧数据
        self.cur_frame += 1
        joint_translation = self.joint_translation[self.cur_frame]
        joint_orientation = self.joint_orientation[self.cur_frame]
        return joint_name, joint_translation, joint_orientation
            
    def transition_to_new_motion(self, motion_idx, best_frame):
        """平滑过渡到新动作"""
        next_motion = self.motion[motion_idx].raw_copy()
        next_motion.joint_position = next_motion.joint_position[best_frame:]
        next_motion.joint_rotation = next_motion.joint_rotation[best_frame:]
        
        # 对齐位置和朝向
        facing_axis = R.from_quat(self.cur_root_rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
        next_motion = next_motion.translation_and_rotation(0, self.cur_root_pos[[0, 2]], facing_axis)
        
        # 惯性化混合
        smooth_motion = concatenate_two_motions(self.cur_motion, next_motion, self.cur_frame, 30)
        self.cur_motion = smooth_motion
        self.cur_motion_idx = motion_idx
        self.cur_frame = 0
        self.cur_end_frame = self.cur_motion.motion_length
        self.joint_translation, self.joint_orientation = self.cur_motion.batch_forward_kinematics()
    
    def choose_dir_idx(self, desired_rot_list): 
        current_rot = R.from_quat(self.cur_root_rot)
        desired_rot = R.from_quat(desired_rot_list[-1])
        diff_rot = desired_rot * current_rot.inv()
        delta_angle = diff_rot.as_rotvec()[1]

        if delta_angle > np.pi * 1 / 6:
            idx = 5 # 右转
        elif delta_angle < -np.pi * 1 / 6:
            idx = 4 # 左转  
        else:
            idx = self.cur_motion_idx  
        return idx                             
    
    def find_best_match(self, motion_idx, target_pos, target_vel, target_rot, target_avel):
        features = self.motion_features[motion_idx]
        min_cost = float('inf')
        best_frame = 0
        
        # 目标特征
        target_future_pos = np.array([target_pos[0], target_pos[2]])  # 仅用XZ平面
        target_future_rot = R.from_quat(target_rot).apply(np.array([0, 0, 1]))[[0, 2]]  # 前方方向
        
        for i in range(len(features)):
            frame_feat = features[i]
            
            # 计算特征距离（加权）
            pos_cost = np.linalg.norm(frame_feat['root_pos'] - target_pos[[0, 2]])
            vel_cost = np.linalg.norm(frame_feat['root_vel'] - target_vel[[0, 2]])
            rot_cost = abs(frame_feat['root_rot'] - R.from_quat(target_rot).as_rotvec()[1])
            avel_cost = abs(frame_feat['root_avel'] - target_avel[1])
            # future_pos_cost = np.linalg.norm(frame_feat['future_pos'] - target_future_pos)
            # future_rot_cost = np.linalg.norm(frame_feat['future_rot'] - target_future_rot)
            
            # 总成本（权重可调）
            total_cost = (
                1.0 * pos_cost + 
                0.5 * vel_cost + 
                1.2 * rot_cost + 
                0.5 * avel_cost
            )
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_frame = i
        
        return best_frame

    def _compute_velocity(self, motion, frame, dt=0.016667):
        """通过有限差分计算根节点速度（XoZ平面）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_pos = motion.joint_position[frame + 1][0]
        current_pos = motion.joint_position[frame][0]
        return (next_pos - current_pos) / dt

    def _compute_angular_velocity(self, motion, frame, dt=0.016667):
        """通过有限差分计算根节点角速度（绕Y轴）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_rot = R.from_quat(motion.joint_rotation[frame + 1][0])
        current_rot = R.from_quat(motion.joint_rotation[frame][0])
        delta_rot = next_rot * current_rot.inv()
        return delta_rot.as_rotvec() / dt
    

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，你可以按需调整
        '''
        controller_pos = character_state[1][0]
        self.controller.set_pos(controller_pos)
    











# 以下部分均为可更改部分，你可以把需要的数据结构定义进来
from typing import List
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
from physics_warpper import PhysicsInfo
import numpy as np
from smooth_utils import *

def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, mix_frame1:int, mix_frame2:int, mix_time:int):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第mix_frame1帧
          mix_time表示用于混合的帧数
    输出: 平滑地连接后的动作，类型为BVHMotion
    '''
    res = bvh_motion2.raw_copy()
    # 剪切
    res.joint_position = res.joint_position[mix_frame2-30:mix_frame2] # smooth_motion共mix_time帧
    res.joint_rotation = res.joint_rotation[mix_frame2-30:mix_frame2]
    
    # inertialization
    # 对齐motion2
    pos = bvh_motion1.joint_position[mix_frame1, 0, [0, 2]]
    rot = bvh_motion1.joint_rotation[mix_frame1, 0]
    facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
    new_bvh_motion2 = res.translation_and_rotation(0, pos, facing_axis)

    half_life = 0.2
    fps = 60
    dt = 1 / fps

    motion1_avel = quat_to_avel(bvh_motion1.joint_rotation[mix_frame1 - int(mix_time / 2):mix_frame1], dt)
    motion2_avel = quat_to_avel(new_bvh_motion2.joint_rotation[0:int(mix_time / 2)], dt)
    avel_diff = motion1_avel[-1] - motion2_avel[0]
    rot_diff = (R.from_quat(bvh_motion1.joint_rotation[mix_frame1])*R.from_quat(new_bvh_motion2.joint_rotation[0].copy()).inv()).as_rotvec()
    
    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_rot(
            rot_diff, avel_diff, half_life, i / fps
        )
        offset_rot = R.from_rotvec(offset[0])
        new_bvh_motion2.joint_rotation[i] = (offset_rot * R.from_quat(new_bvh_motion2.joint_rotation[i])).as_quat() 
        
    motion1_vel = bvh_motion1.joint_position[mix_frame1] - bvh_motion1.joint_position[mix_frame1-1]
    motion2_vel = new_bvh_motion2.joint_position[1]-new_bvh_motion2.joint_position[0]
    vel_diff = (motion1_vel - motion2_vel) / fps
    pos_diff = bvh_motion1.joint_position[mix_frame1] - new_bvh_motion2.joint_position[0]

    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_pos(
            pos_diff, vel_diff, half_life, i / fps
        )
        offset_pos = offset[0]
        new_bvh_motion2.joint_position[i] += offset_pos

    return new_bvh_motion2


class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pass

    def apply_root_force_and_torque(self):
        pass
    
    def apply_static_torque(self):
        pass

class CharacterController():
    def __init__(self, viewer, controller, pd_controller) -> None:
        # viewer 类，封装physics
        self.viewer = viewer
        # 手柄/键盘控制器
        self.controller = controller
        # pd controller
        self.pd_controller = pd_controller
          
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 101
        self.cur_motion_idx = 0
        
        # 转换motion
        self.smooth_motion = None
        
        # 动作转换
        self.is_transition = False
        self.transition_frame = 0
        self.max_transition_frame = 30
        
        # motion
        # 短motion
        self.motion = []
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/walk.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/run.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/idle.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/jump.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/turn_left.bvh'))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/turn_right.bvh'))
        # 长motion
        self.motions = []
        self.motions = [
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk.bvh'),  
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_run.bvh'),  
            # BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk_mirror.bvh'),  
            # BVHMotion(bvh_file_name='./motion_material/long_motion/long_run_mirror.bvh'),  
        ]
        self.joint_translation = []
        self.joint_orientation = []
        for i, motion in enumerate(self.motions):
            self.motions[i].adjust_joint_name(self.viewer.joint_name)
            trans, orient = motion.batch_forward_kinematics()
            self.joint_translation.append(trans)
            self.joint_orientation.append(orient)
        
        # 当前角色的参考root位置和旋转
        self.cur_root_pos = self.joint_translation[0][0]
        self.cur_root_rot = self.joint_orientation[0][0]
        
        self.cur_motion = self.motions[0]
        self.cur_motion.adjust_joint_name(self.viewer.joint_name)
        self.joint_name = self.cur_motion.joint_name
        
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_motion.motion_length
        
        # 预处理所有运动的特征
        self.motion_features = None
        self.preprocess_motion_features()
    
    
    def preprocess_motion_features(self):
        """预处理所有运动的特征数据"""
        self.motion_features = []
        for motion in self.motions:
            joint_trans, joint_ori = motion.batch_forward_kinematics()
            num_frames = motion.num_frames
            features = []
            
            for i in range(num_frames):
                # 当前帧特征
                root_pos = joint_trans[i][0]
                root_rot = joint_ori[i][0]
                root_vel = self._compute_velocity(motion, i) if i < num_frames - 1 else np.zeros(3)
                root_avel = self._compute_angular_velocity(motion, i) if i < num_frames - 1 else np.zeros(3)
                
                # 未来轨迹特征（20/40/60帧）
                future_pos = []
                future_rot = []
                for offset in [20, 40, 60]:
                    future_frame = min(i + offset, num_frames - 1)
                    future_pos.append(joint_trans[future_frame][0][[0, 2]])  # XZ平面
                    future_dir = R.from_quat(joint_ori[future_frame][0]).apply(np.array([0, 0, 1]))[[0, 2]]  # 前方方向
                    future_rot.append(future_dir)
                
                features.append({
                    'root_pos': root_pos[[0, 2]],  # XZ平面
                    'root_vel': root_vel[[0, 2]],
                    'root_rot': R.from_quat(root_rot).as_rotvec()[1],  # 绕Y轴旋转角度
                    'root_avel': root_avel[1],  # 绕Y轴角速度
                    'future_pos': np.concatenate(future_pos),
                    'future_rot': np.concatenate(future_rot),
                })
            self.motion_features.append(features)
            
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
        joint_name = self.joint_name
    
        # 正在转换到最佳动作帧
        if self.is_transition:
            # 转换还未结束
            if self.transition_frame < 29:
                self.transition_frame += 1
                smoooth_joint_translation, smoooth_joint_orientation = self.smooth_motion.batch_forward_kinematics()
                return joint_name, smoooth_joint_translation[self.transition_frame], smoooth_joint_orientation[self.transition_frame]
            else: # 转换结束
                # 重置
                self.is_transition = False
                self.transition_frame = 0
                # 开始从 best_frame 播放
                return joint_name, self.joint_translation[self.cur_motion_idx][self.cur_frame], self.joint_orientation[self.cur_motion_idx][self.cur_frame]
        
        # 没有在转换动作
        # 目标特征
        target_pos = desired_pos_list[-1] 
        target_vel = desired_vel_list[-1]
        target_rot = desired_rot_list[-1]
        target_avel = desired_avel_list[-1]
        
        # 选择动作类型（行走/跑步）
        motion_idx = 0 if np.linalg.norm(target_vel[[0, 2]]) < 1.8 else 1
        
        # 搜索最佳帧
        if self.cur_frame % 30 == 0 or self.cur_frame >= self.cur_end_frame:
            best_frame = self.find_best_match(motion_idx, target_pos, target_vel, target_rot, target_avel)
            # 记录转换
            self.is_transition = True
            
            # 平滑动作
            self.smooth_motion = concatenate_two_motions(self.cur_motion, self.motions[motion_idx], self.cur_frame, best_frame, 30)
            self.smooth_motion.adjust_joint_name(self.viewer.joint_name)
            # 更新
            self.cur_frame = best_frame
            self.cur_motion_idx = motion_idx
            self.cur_motion = self.motions[motion_idx]
            
            smoooth_joint_translation, smoooth_joint_orientation = self.smooth_motion.batch_forward_kinematics()
            return self.joint_name, smoooth_joint_translation[0], smoooth_joint_orientation[0]
        # 顺序播放
        else:
            # 正常进行下一帧
            joint_translation = self.joint_translation[self.cur_motion_idx][self.cur_frame]
            joint_orientation = self.joint_orientation[self.cur_motion_idx][self.cur_frame]
            self.cur_frame += 1
            return self.joint_name, joint_translation, joint_orientation
                
    
    def find_best_match(self, motion_idx, target_pos, target_vel, target_rot, target_avel):
        features = self.motion_features[motion_idx]
        min_cost = float('inf')
        best_frame = 0
        
        # 目标特征
        target_future_pos = np.array([target_pos[0], target_pos[2]])  # 仅用XZ平面
        target_future_rot = R.from_quat(target_rot).apply(np.array([0, 0, 1]))[[0, 2]]  # 前方方向
        
        for i in range(300, len(features)-50):
            frame_feat = features[i]
            
            # 计算特征距离（加权）
            pos_cost = np.linalg.norm(frame_feat['root_pos'] - target_pos[[0, 2]])
            vel_cost = np.linalg.norm(frame_feat['root_vel'] - target_vel[[0, 2]])
            rot_cost = abs(frame_feat['root_rot'] - R.from_quat(target_rot).as_rotvec()[1])
            avel_cost = abs(frame_feat['root_avel'] - target_avel[1])
            # future_pos_cost = np.linalg.norm(frame_feat['future_pos'] - target_future_pos)
            # future_rot_cost = np.linalg.norm(frame_feat['future_rot'] - target_future_rot)
            
            # 总成本
            total_cost = (
                1.0 * pos_cost + 
                1.0 * vel_cost + 
                1.2 * rot_cost + 
                0.5 * avel_cost
            )
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_frame = i
        
        return best_frame

    def _compute_velocity(self, motion, frame, dt=0.016667):
        """通过有限差分计算根节点速度（XoZ平面）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_pos = motion.joint_position[frame + 1][0]
        current_pos = motion.joint_position[frame][0]
        return (next_pos - current_pos) / dt

    def _compute_angular_velocity(self, motion, frame, dt=0.016667):
        """通过有限差分计算根节点角速度（绕Y轴）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_rot = R.from_quat(motion.joint_rotation[frame + 1][0])
        current_rot = R.from_quat(motion.joint_rotation[frame][0])
        delta_rot = next_rot * current_rot.inv()
        return delta_rot.as_rotvec() / dt

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，你可以按需调整
        '''
        controller_pos = character_state[1][0]
        self.controller.set_pos(controller_pos)
    













# 以下部分均为可更改部分，你可以把需要的数据结构定义进来
from typing import List
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
from physics_warpper import PhysicsInfo
import numpy as np
from smooth_utils import *

def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, mix_frame1:int, mix_frame2:int, mix_time:int):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第mix_frame1帧
          mix_time表示用于混合的帧数
    输出: 平滑地连接后的动作，类型为BVHMotion
    '''
    res = bvh_motion2.raw_copy()
    # 剪切
    res.joint_position = res.joint_position[mix_frame2-40:mix_frame2]  # smooth_motion共40帧
    res.joint_rotation = res.joint_rotation[mix_frame2-40:mix_frame2]
    
    # inertialization
    # 对齐motion2
    pos = bvh_motion1.joint_position[mix_frame1, 0, [0, 2]]
    rot = bvh_motion1.joint_rotation[mix_frame1, 0]
    facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
    new_bvh_motion2 = res.translation_and_rotation(0, pos, facing_axis)

    half_life = 0.3
    fps = 100
    dt = 1 / fps

    motion1_avel = quat_to_avel(bvh_motion1.joint_rotation[mix_frame1 - int(mix_time / 2):mix_frame1], dt)
    motion2_avel = quat_to_avel(new_bvh_motion2.joint_rotation[0:int(mix_time / 2)], dt)
    avel_diff = motion1_avel[-1] - motion2_avel[0]
    rot_diff = (R.from_quat(bvh_motion1.joint_rotation[mix_frame1])*R.from_quat(new_bvh_motion2.joint_rotation[0].copy()).inv()).as_rotvec()
    
    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_rot(
            rot_diff, avel_diff, half_life, i / fps
        )
        offset_rot = R.from_rotvec(offset[0])
        new_bvh_motion2.joint_rotation[i] = (offset_rot * R.from_quat(new_bvh_motion2.joint_rotation[i])).as_quat() 
        
    motion1_vel = bvh_motion1.joint_position[mix_frame1] - bvh_motion1.joint_position[mix_frame1-1]
    motion2_vel = new_bvh_motion2.joint_position[1]-new_bvh_motion2.joint_position[0]
    vel_diff = (motion1_vel - motion2_vel) / fps
    pos_diff = bvh_motion1.joint_position[mix_frame1] - new_bvh_motion2.joint_position[0]

    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_pos(
            pos_diff, vel_diff, half_life, i / fps
        )
        offset_pos = offset[0]
        new_bvh_motion2.joint_position[i] += offset_pos

    return new_bvh_motion2

def initial_smooth_motion(cur_motion, next_motion, cur_frame, best_frame):
    res = next_motion.raw_copy()
    # 剪切
    res.joint_position = res.joint_position[best_frame-100:best_frame] 
    res.joint_rotation = res.joint_rotation[best_frame-100:best_frame]
    # 开头
    pos = cur_motion.joint_position[cur_frame]
    rot = cur_motion.joint_rotation[cur_frame]
    # 修改
    res.joint_position[0] = pos
    res.joint_rotation[0] = rot
    return res
    
def build_smooth_motion(bvh_motion, half_life = 0.2, fps = 100):
    
    # ---------------处理rotations----------------#
    rotations = bvh_motion.joint_rotation
    avel = quat_to_avel(rotations, 1/100)
    
    # 计算最后一帧和第一帧的旋转差
    rot_diff = (R.from_quat(rotations[-1]) * R.from_quat(rotations[0].copy()).inv()).as_rotvec()
    avel_diff = (avel[-1] - avel[0])
    
    # 将旋转差均匀分布到每一帧
    for i in range(bvh_motion.motion_length):
        offset1 = decay_spring_implicit_damping_rot(
            0.5*rot_diff, 0.5*avel_diff, half_life, i/fps
        )
        offset2 = decay_spring_implicit_damping_rot(
            -0.5*rot_diff, -0.5*avel_diff, half_life, (bvh_motion.motion_length-i-1)/fps
        )
        offset_rot = R.from_rotvec(offset1[0] + offset2[0])
        bvh_motion.joint_rotation[i] = (offset_rot * R.from_quat(rotations[i])).as_quat() 
    
    # -------------------处理positions------------------#
    
    pos_diff = bvh_motion.joint_position[-1] - bvh_motion.joint_position[0]
    # pos_diff[:,[0,2]] = 0
    vel1 = bvh_motion.joint_position[-1] - bvh_motion.joint_position[-2]
    vel2 = bvh_motion.joint_position[1] - bvh_motion.joint_position[0]
    vel_diff = (vel1 - vel2)/100
    
    for i in range(bvh_motion.motion_length):
        offset1 = decay_spring_implicit_damping_pos(
            0.5*pos_diff, 0.5*vel_diff, half_life, i/fps
        )
        offset2 = decay_spring_implicit_damping_pos(
            -0.5*pos_diff, -0.5*vel_diff, half_life, (bvh_motion.motion_length-i-1)/fps
        )
        offset_pos = offset1[0] + offset2[0]
        bvh_motion.joint_position[i] += offset_pos
    
    return bvh_motion

class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pass

    def apply_root_force_and_torque(self):
        pass
    
    def apply_static_torque(self):
        pass

class CharacterController():
    def __init__(self, viewer, controller, pd_controller) -> None:
        # viewer 类，封装physics
        self.viewer = viewer
        # 手柄/键盘控制器
        self.controller = controller
        # pd controller
        self.pd_controller = pd_controller
          
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 101
        self.cur_motion_idx = 0
        
        # 转换motion
        self.smooth_motion = None
        
        # 动作转换
        self.is_transition = False
        self.transition_frame = 0
        
        # motion
        # 长motion
        self.motions = []
        self.motions = [
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk.bvh'),  
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_run.bvh'),  
            # BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk_mirror.bvh'),  
            # BVHMotion(bvh_file_name='./motion_material/long_motion/long_run_mirror.bvh'),  
        ]
        self.joint_translation = []
        self.joint_orientation = []
        for i, motion in enumerate(self.motions):
            self.motions[i].adjust_joint_name(self.viewer.joint_name)
            trans, orient = motion.batch_forward_kinematics()
            self.joint_translation.append(trans)
            self.joint_orientation.append(orient)
        
        # 当前角色的参考root位置和旋转
        self.cur_root_pos = self.joint_translation[0][0][0]
        self.cur_root_rot = self.joint_orientation[0][0][0]
        
        self.cur_motion = self.motions[0]
        self.cur_motion.adjust_joint_name(self.viewer.joint_name)
        self.joint_name = self.cur_motion.joint_name
        
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_motion.motion_length
        
        # 预处理所有运动的特征
        self.motion_features = None
        self.preprocess_motion_features()
    
    
    def preprocess_motion_features(self):
        """预处理所有运动的特征数据"""
        self.motion_features = []
        for motion in self.motions:
            joint_trans, joint_ori = motion.batch_forward_kinematics()
            num_frames = motion.num_frames
            features = []
            
            for i in range(num_frames):
                # 当前帧特征
                root_pos = joint_trans[i][0]
                root_rot = joint_ori[i][0]
                root_vel = self._compute_velocity(motion, i) if i < num_frames - 1 else np.zeros(3)
                root_avel = self._compute_angular_velocity(motion, i) if i < num_frames - 1 else np.zeros(3)
                
                # 未来轨迹特征（40帧）
                future_frame = min(i + 40, num_frames - 2)
                future_pos = joint_trans[future_frame][0]
                future_rot = joint_ori[future_frame][0]
                future_vel = self._compute_velocity(motion, future_frame) if future_frame < num_frames - 1 else np.zeros(3)
                future_avel = self._compute_angular_velocity(motion, future_frame) if future_frame < num_frames - 1 else np.zeros(3)
                
                features.append({
                    'root_pos': root_pos[[0, 2]],  # XZ平面
                    'root_vel': root_vel[[0, 2]],
                    'root_rot': R.from_quat(root_rot).as_rotvec()[1],  # 绕Y轴旋转角度
                    'root_avel': root_avel[1],  # 绕Y轴角速度
                    'future_pos': future_pos[[0, 2]],
                    'future_vel': future_vel[[0, 2]],
                    'future_rot': R.from_quat(future_rot).as_rotvec()[1],
                    'future_avel': future_avel[1],  
                })
            self.motion_features.append(features)
            
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
        joint_name = self.joint_name
    
        # 正在转换到最佳动作帧
        if self.is_transition:
            # 转换还未结束
            if self.transition_frame < 39:
                self.transition_frame += 1
                smoooth_joint_translation, smoooth_joint_orientation = self.smooth_motion.batch_forward_kinematics()
                return joint_name, smoooth_joint_translation[self.transition_frame], smoooth_joint_orientation[self.transition_frame]
            else: # 转换结束
                # 重置
                self.is_transition = False
                self.transition_frame = 0
                # 更新
                self.cur_root_pos = self.joint_translation[self.cur_motion_idx][self.cur_frame][0]
                self.cur_root_rot = self.joint_orientation[self.cur_motion_idx][self.cur_frame][0]
                # 开始从 best_frame 播放
                return joint_name, self.joint_translation[self.cur_motion_idx][self.cur_frame], self.joint_orientation[self.cur_motion_idx][self.cur_frame]
        
        # 没有在转换动作
        # 目标特征(未来第40帧)
        target_pos = desired_pos_list[2] 
        target_vel = desired_vel_list[2]
        target_rot = desired_rot_list[2]
        target_avel = desired_avel_list[2]
        
        # 选择动作类型（行走/跑步）
        motion_idx = 0 if np.linalg.norm(target_vel[[0, 2]]) < 1.7 else 1
        
        # 搜索最佳帧
        if self.cur_frame % 40 == 0 or self.cur_frame >= self.cur_end_frame:
            best_frame = self.find_best_match(motion_idx, target_pos, target_vel, target_rot, target_avel)
            # 记录转换
            self.is_transition = True
            self.transition_frame = 0
            # 平滑动作
            self.smooth_motion = concatenate_two_motions(self.cur_motion, self.motions[motion_idx], self.cur_frame, best_frame, 30)
            # bvh_motion = initial_smooth_motion(self.cur_motion, self.motions[motion_idx], self.cur_frame, best_frame)
            # self.smooth_motion = build_smooth_motion(bvh_motion, half_life = 0.2, fps = 100)
            self.smooth_motion.adjust_joint_name(self.viewer.joint_name)
            # 更新
            self.cur_frame = best_frame
            self.cur_motion_idx = motion_idx
            self.cur_motion = self.motions[motion_idx]
            
            smoooth_joint_translation, smoooth_joint_orientation = self.smooth_motion.batch_forward_kinematics()
            self.cur_root_pos = smoooth_joint_translation[0][0]
            self.cur_root_rot = smoooth_joint_orientation[0][0]
            return self.joint_name, smoooth_joint_translation[0], smoooth_joint_orientation[0]
        # 顺序播放
        else:
            # 正常进行下一帧
            joint_translation = self.joint_translation[self.cur_motion_idx][self.cur_frame]
            joint_orientation = self.joint_orientation[self.cur_motion_idx][self.cur_frame]
            # 更新
            self.cur_root_pos = joint_translation[0]
            self.cur_root_rot = joint_orientation[0]
            self.cur_frame += 1
            return self.joint_name, joint_translation, joint_orientation
                
    
    def find_best_match(self, motion_idx, target_pos, target_vel, target_rot, target_avel):
        features = self.motion_features[motion_idx]
        min_cost = float('inf')
        best_frame = 0
        
        for i in range(200, len(features)-150):
            frame_feat = features[i]
            
            # 计算特征距离（加权）
            cur_pos_cost = np.linalg.norm(frame_feat['root_pos'] - self.cur_root_pos[[0, 2]])
            # cur_vel_cost = np.linalg.norm(frame_feat['root_vel'] - target_vel[[0, 2]])
            cur_rot_cost = abs(frame_feat['root_rot'] - R.from_quat(self.cur_root_rot).as_rotvec()[1])
            # cur_avel_cost = abs(frame_feat['root_avel'] - target_avel[1])
            future_pos_cost = np.linalg.norm(frame_feat['future_pos'] - target_pos[[0, 2]])
            future_vel_cost = np.linalg.norm(frame_feat['future_vel'] - target_vel[[0, 2]])
            future_rot_cost = abs(frame_feat['future_rot'] - R.from_quat(target_rot).as_rotvec()[1])
            future_avel_cost = abs(frame_feat['future_avel'] - target_avel[1])
            
            # 总成本
            total_cost = (
                1.0 * cur_pos_cost + 
                2.0 * cur_rot_cost + 
                1.0 * future_pos_cost + 
                5.0 * future_vel_cost + 
                2.0 * future_rot_cost + 
                0.5 * future_avel_cost
            )
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_frame = i
        
        return best_frame + 40

    def _compute_velocity(self, motion, frame, dt=0.01):
        """通过有限差分计算根节点速度（XoZ平面）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_pos = motion.joint_position[frame + 1][0]
        current_pos = motion.joint_position[frame][0]
        return (next_pos - current_pos) / dt

    def _compute_angular_velocity(self, motion, frame, dt=0.01):
        """通过有限差分计算根节点角速度（绕Y轴）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_rot = R.from_quat(motion.joint_rotation[frame + 1][0])
        current_rot = R.from_quat(motion.joint_rotation[frame][0])
        delta_rot = next_rot * current_rot.inv()
        return delta_rot.as_rotvec() / dt

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，你可以按需调整
        '''
        controller_pos = character_state[1][0]
        self.controller.set_pos(controller_pos)
    


























# 19:33
# 以下部分均为可更改部分，你可以把需要的数据结构定义进来
from typing import List
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
from physics_warpper import PhysicsInfo
import numpy as np
from smooth_utils import *

def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, mix_frame1:int, mix_frame2:int, cur_root_pos, cur_root_rot, mix_time:int):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第mix_frame1帧
          mix_time表示用于混合的帧数
    输出: 平滑地连接后的动作，类型为BVHMotion
    '''
    res = bvh_motion2.raw_copy()
    # 剪切
    res.joint_position = res.joint_position[mix_frame2-40:mix_frame2]  # smooth_motion共40帧
    res.joint_rotation = res.joint_rotation[mix_frame2-40:mix_frame2]
    
    # inertialization
    # 对齐motion2
    pos = cur_root_pos[[0, 2]]
    rot = cur_root_rot
    facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
    new_bvh_motion2 = res.translation_and_rotation(0, pos, facing_axis)

    half_life = 0.3
    fps = 100
    dt = 1 / fps

    motion1_avel = quat_to_avel(bvh_motion1.joint_rotation[mix_frame1 - int(mix_time / 2):mix_frame1], dt)
    motion2_avel = quat_to_avel(new_bvh_motion2.joint_rotation[0:int(mix_time / 2)], dt)
    avel_diff = motion1_avel[-1] - motion2_avel[0]
    rot_diff = (R.from_quat(bvh_motion1.joint_rotation[mix_frame1])*R.from_quat(new_bvh_motion2.joint_rotation[0].copy()).inv()).as_rotvec()
    
    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_rot(
            rot_diff, avel_diff, half_life, i / fps
        )
        offset_rot = R.from_rotvec(offset[0])
        new_bvh_motion2.joint_rotation[i] = (offset_rot * R.from_quat(new_bvh_motion2.joint_rotation[i])).as_quat() 
        
    motion1_vel = bvh_motion1.joint_position[mix_frame1] - bvh_motion1.joint_position[mix_frame1-1]
    motion2_vel = new_bvh_motion2.joint_position[1]-new_bvh_motion2.joint_position[0]
    vel_diff = (motion1_vel - motion2_vel) / fps
    pos_diff = bvh_motion1.joint_position[mix_frame1] - new_bvh_motion2.joint_position[0]

    for i in range(new_bvh_motion2.motion_length):
        offset = decay_spring_implicit_damping_pos(
            pos_diff, vel_diff, half_life, i / fps
        )
        offset_pos = offset[0]
        new_bvh_motion2.joint_position[i] += offset_pos

    return new_bvh_motion2

def initial_smooth_motion(cur_motion, next_motion, cur_frame, best_frame):
    res = next_motion.raw_copy()
    # 剪切
    res.joint_position = res.joint_position[best_frame-100:best_frame] 
    res.joint_rotation = res.joint_rotation[best_frame-100:best_frame]
    # 开头
    pos = cur_motion.joint_position[cur_frame]
    rot = cur_motion.joint_rotation[cur_frame]
    # 修改
    res.joint_position[0] = pos
    res.joint_rotation[0] = rot
    return res
    
def build_smooth_motion(bvh_motion, half_life = 0.2, fps = 100):
    
    # ---------------处理rotations----------------#
    rotations = bvh_motion.joint_rotation
    avel = quat_to_avel(rotations, 1/100)
    
    # 计算最后一帧和第一帧的旋转差
    rot_diff = (R.from_quat(rotations[-1]) * R.from_quat(rotations[0].copy()).inv()).as_rotvec()
    avel_diff = (avel[-1] - avel[0])
    
    # 将旋转差均匀分布到每一帧
    for i in range(bvh_motion.motion_length):
        offset1 = decay_spring_implicit_damping_rot(
            0.5*rot_diff, 0.5*avel_diff, half_life, i/fps
        )
        offset2 = decay_spring_implicit_damping_rot(
            -0.5*rot_diff, -0.5*avel_diff, half_life, (bvh_motion.motion_length-i-1)/fps
        )
        offset_rot = R.from_rotvec(offset1[0] + offset2[0])
        bvh_motion.joint_rotation[i] = (offset_rot * R.from_quat(rotations[i])).as_quat() 
    
    # -------------------处理positions------------------#
    
    pos_diff = bvh_motion.joint_position[-1] - bvh_motion.joint_position[0]
    # pos_diff[:,[0,2]] = 0
    vel1 = bvh_motion.joint_position[-1] - bvh_motion.joint_position[-2]
    vel2 = bvh_motion.joint_position[1] - bvh_motion.joint_position[0]
    vel_diff = (vel1 - vel2)/100
    
    for i in range(bvh_motion.motion_length):
        offset1 = decay_spring_implicit_damping_pos(
            0.5*pos_diff, 0.5*vel_diff, half_life, i/fps
        )
        offset2 = decay_spring_implicit_damping_pos(
            -0.5*pos_diff, -0.5*vel_diff, half_life, (bvh_motion.motion_length-i-1)/fps
        )
        offset_pos = offset1[0] + offset2[0]
        bvh_motion.joint_position[i] += offset_pos
    
    return bvh_motion

class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pass

    def apply_root_force_and_torque(self):
        pass
    
    def apply_static_torque(self):
        pass

class CharacterController():
    def __init__(self, viewer, controller, pd_controller) -> None:
        # viewer 类，封装physics
        self.viewer = viewer
        # 手柄/键盘控制器
        self.controller = controller
        # pd controller
        self.pd_controller = pd_controller
          
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 101
        self.cur_motion_idx = 0
        
        # 转换motion
        self.smooth_motion = None
        
        # 动作转换
        self.is_transition = False
        self.transition_frame = 0
        
        # motion
        # 长motion
        self.motions = []
        self.motions = [
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk.bvh'),  
            BVHMotion(bvh_file_name='./motion_material/long_motion/long_run.bvh'),  
            # BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk_mirror.bvh'),  
            # BVHMotion(bvh_file_name='./motion_material/long_motion/long_run_mirror.bvh'),  
        ]
        self.joint_translation = []
        self.joint_orientation = []
        for i, motion in enumerate(self.motions):
            self.motions[i].adjust_joint_name(self.viewer.joint_name)
            trans, orient = motion.batch_forward_kinematics()
            self.joint_translation.append(trans)
            self.joint_orientation.append(orient)
        
        
        self.cur_motion = self.motions[0]
        self.cur_motion.adjust_joint_name(self.viewer.joint_name)
        self.joint_name = self.cur_motion.joint_name
        
        # 当前角色的参考root位置和旋转
        self.cur_root_pos = self.cur_motion.joint_position[self.cur_frame][0]
        self.cur_root_rot = self.cur_motion.joint_rotation[self.cur_frame][0]
        
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_motion.motion_length
        
        # 预处理所有运动的特征
        self.motion_features = None
        self.preprocess_motion_features()
    
    
    def preprocess_motion_features(self):
        """预处理所有运动的特征数据"""
        self.motion_features = []
        for motion in self.motions:
            num_frames = motion.num_frames
            features = []
            
            for i in range(num_frames):
                # 当前帧特征
                root_pos = motion.joint_position[i][0]
                root_rot = motion.joint_rotation[i][0]
                root_vel = self._compute_velocity(motion, i) if i < num_frames - 1 else np.zeros(3)
                root_avel = self._compute_angular_velocity(motion, i) if i < num_frames - 1 else np.zeros(3)
                
                # 未来轨迹特征（40帧）
                future_frame = min(i + 40, num_frames - 2)
                future_pos = motion.joint_position[future_frame][0]
                future_rot = motion.joint_rotation[future_frame][0]
                future_vel = self._compute_velocity(motion, future_frame) if future_frame < num_frames - 1 else np.zeros(3)
                future_avel = self._compute_angular_velocity(motion, future_frame) if future_frame < num_frames - 1 else np.zeros(3)
                
                features.append({
                    'root_pos': root_pos[[0, 2]],  # XZ平面
                    'root_vel': root_vel[[0, 2]],
                    'root_rot': R.from_quat(root_rot).as_rotvec()[1],  # 绕Y轴旋转角度
                    'root_avel': root_avel[1],  # 绕Y轴角速度
                    'future_pos': future_pos[[0, 2]],
                    'future_vel': future_vel[[0, 2]],
                    'future_rot': R.from_quat(future_rot).as_rotvec()[1],
                    'future_avel': future_avel[1],  
                })
            self.motion_features.append(features)
            
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
        joint_name = self.joint_name
    
        # 正在转换到最佳动作帧
        if self.is_transition:
            # 转换还未结束
            if self.transition_frame < 39:
                self.transition_frame += 1
                smoooth_joint_translation, smoooth_joint_orientation = self.smooth_motion.batch_forward_kinematics()
                return joint_name, smoooth_joint_translation[self.transition_frame], smoooth_joint_orientation[self.transition_frame]
            else: # 转换结束
                # 重置
                self.is_transition = False
                self.transition_frame = 0
                # 更新
                self.cur_root_pos = self.cur_motion.joint_position[self.cur_frame][0]
                self.cur_root_rot = self.cur_motion.joint_rotation[self.cur_frame][0]
                # 开始从 best_frame 播放
                return joint_name, self.joint_translation[self.cur_motion_idx][self.cur_frame], self.joint_orientation[self.cur_motion_idx][self.cur_frame]
        
        # 没有在转换动作
        # 目标特征(未来第40帧)
        target_pos = desired_pos_list[2] 
        target_vel = desired_vel_list[2]
        target_rot = desired_rot_list[2]
        target_avel = desired_avel_list[2]
        
        # 选择动作类型（行走/跑步）
        motion_idx = 0 if np.linalg.norm(target_vel[[0, 2]]) < 1.7 else 1
        
        # 搜索最佳帧
        if self.cur_frame % 40 == 0 or self.cur_frame >= self.cur_end_frame:
            best_frame = self.find_best_match(motion_idx, target_pos, target_vel, target_rot, target_avel)
            # 记录转换
            self.is_transition = True
            self.transition_frame = 0
            # 平滑动作
            self.smooth_motion = concatenate_two_motions(self.cur_motion, self.motions[motion_idx], self.cur_frame, best_frame, self.cur_root_pos, self.cur_root_rot, 30)
            # bvh_motion = initial_smooth_motion(self.cur_motion, self.motions[motion_idx], self.cur_frame, best_frame)
            # self.smooth_motion = build_smooth_motion(bvh_motion, half_life = 0.2, fps = 100)
            self.smooth_motion.adjust_joint_name(self.viewer.joint_name)
            # 更新
            self.cur_frame = best_frame
            self.cur_motion_idx = motion_idx
            self.cur_motion = self.motions[motion_idx]
            
            smoooth_joint_translation, smoooth_joint_orientation = self.smooth_motion.batch_forward_kinematics()
            self.cur_root_pos = self.smooth_motion.joint_position[0][0]
            self.cur_root_rot = self.smooth_motion.joint_rotation[0][0]
            return self.joint_name, smoooth_joint_translation[0], smoooth_joint_orientation[0]
        # 顺序播放
        else:
            # 正常进行下一帧
            self.cur_frame += 1
            joint_translation = self.joint_translation[self.cur_motion_idx][self.cur_frame]
            joint_orientation = self.joint_orientation[self.cur_motion_idx][self.cur_frame]
            # 更新
            self.cur_root_pos = self.cur_motion.joint_position[self.cur_frame][0]
            self.cur_root_rot = self.cur_motion.joint_rotation[self.cur_frame][0]
            
            return self.joint_name, joint_translation, joint_orientation
                
    
    def find_best_match(self, motion_idx, target_pos, target_vel, target_rot, target_avel):
        features = self.motion_features[motion_idx]
        min_cost = float('inf')
        best_frame = 0
        
        for i in range(200, len(features)-150):
            frame_feat = features[i]
            
            # 计算特征距离（加权）
            cur_pos_cost = np.linalg.norm(frame_feat['root_pos'] - self.cur_root_pos[[0, 2]])
            # cur_vel_cost = np.linalg.norm(frame_feat['root_vel'] - target_vel[[0, 2]])
            cur_rot_cost = abs(frame_feat['root_rot'] - R.from_quat(self.cur_root_rot).as_rotvec()[1])
            # cur_avel_cost = abs(frame_feat['root_avel'] - target_avel[1])
            future_pos_cost = np.linalg.norm(frame_feat['future_pos'] - target_pos[[0, 2]])
            future_vel_cost = np.linalg.norm(frame_feat['future_vel'] - target_vel[[0, 2]])
            future_rot_cost = abs(frame_feat['future_rot'] - R.from_quat(target_rot).as_rotvec()[1])
            future_avel_cost = abs(frame_feat['future_avel'] - target_avel[1])
            
            # 总成本
            total_cost = (
                1.0 * cur_pos_cost + 
                2.0 * cur_rot_cost + 
                1.0 * future_pos_cost + 
                5.0 * future_vel_cost + 
                2.0 * future_rot_cost + 
                0.5 * future_avel_cost
            )
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_frame = i
        
        return best_frame + 40

    def _compute_velocity(self, motion, frame, dt=0.01):
        """通过有限差分计算根节点速度（XoZ平面）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_pos = motion.joint_position[frame + 1][0]
        current_pos = motion.joint_position[frame][0]
        return (next_pos - current_pos) / dt

    def _compute_angular_velocity(self, motion, frame, dt=0.01):
        """通过有限差分计算根节点角速度（绕Y轴）"""
        if frame + 1 >= motion.num_frames:
            return np.zeros(3)
        next_rot = R.from_quat(motion.joint_rotation[frame + 1][0])
        current_rot = R.from_quat(motion.joint_rotation[frame][0])
        delta_rot = next_rot * current_rot.inv()
        return delta_rot.as_rotvec() / dt

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，你可以按需调整
        '''
        controller_pos = character_state[1][0]
        self.controller.set_pos(controller_pos)
    
