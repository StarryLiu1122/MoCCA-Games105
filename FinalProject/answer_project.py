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
    
