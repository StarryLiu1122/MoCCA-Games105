##############
# 姓名：刘星云
# 学号：2300012297
##############
"""
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
"""
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from bvh_motion import BVHMotion
from smooth_utils import *

# part1
def blend_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, v:float=None, input_alpha:np.ndarray=None) -> BVHMotion:
    '''
    输入: 两个将要blend的动作，类型为BVHMotion
          将要生成的BVH的速度v
          如果给出插值的系数alpha就不需要再计算了
          target_fps,将要生成BVH的fps
    输出: blend两个BVH动作后的动作，类型为BVHMotion
    假设两个动作的帧数分别为n1, n2
    首先需要制作blend 的权重适量 alpha
    插值系数alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    Tips:
        1. 计算给出两端动作的速度，两个BVH已经将Root Joint挪到(0.0, 0.0)的XOZ位置上了，为了便于你计算，我们假定提供的bvh都是沿着z轴正方向向前运动的
        2. 利用v计算插值系数alpha
        3. 线性插值以及Slerp
    '''
    
    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros_like(res.joint_position)
    res.joint_rotation = np.zeros_like(res.joint_rotation)
    res.joint_rotation[...,3] = 1.0

    ### Your code here
    n1 = bvh_motion1.joint_position.shape[0]    # walk
    n2 = bvh_motion2.joint_position.shape[0]    # run
    v1 = bvh_motion1.joint_position[-1,0,2] / n1 * 60
    v2 = bvh_motion2.joint_position[-1,0,2] / n2 * 60
    blend_weight = (v2 - v) / (v2 - v1) 
    distance = blend_weight * bvh_motion1.joint_position[-1,0,2] + (1 - blend_weight) * bvh_motion2.joint_position[-1,0,2]
    cycle_time = np.around(distance / v*60).astype(np.int32)
    alpha = np.ones((cycle_time,)) * blend_weight
    n3 = len(alpha)
    
    res.joint_position = np.zeros((n3, res.joint_position.shape[1], res.joint_position.shape[2]))
    res.joint_rotation = np.zeros((n3, res.joint_rotation.shape[1], res.joint_rotation.shape[2]))
    for frame in range(n3):
        walk_frame = int(n1/n3 * frame)
        run_frame = int(n2/n3 * frame)
        
        res.joint_position[frame, :, :] = alpha[frame] * bvh_motion1.joint_position[walk_frame, :, :] + (1-alpha[frame]) * bvh_motion2.joint_position[run_frame, :, :]
        
        for joint in range(len(bvh_motion1.joint_name)):
            rotation_walk = bvh_motion1.joint_rotation[walk_frame,joint,:]
            rotation_run = bvh_motion2.joint_rotation[run_frame,joint, :]
            cos_theta = np.dot(rotation_walk, rotation_run)
            
            if cos_theta < 0.:
                cos_theta = -cos_theta
                rotation_walk = -rotation_walk
            
            theta = np.arccos(cos_theta)
            sin_theta = np.sin(theta)
            
            if sin_theta > 0.0001:
                alpha_1 = np.sin(alpha[frame] * theta) / sin_theta
                alpha_2 = np.sin((1-alpha[frame]) * theta) / sin_theta
            else:
                alpha_1 = alpha[frame]
                alpha_2 = 1-alpha[frame]
            res.joint_rotation[frame,joint,:] = alpha_1 * rotation_walk + alpha_2 * rotation_run
            res.joint_rotation[frame,joint,:] /= np.linalg.norm(res.joint_rotation[frame,joint,:])
    
    ### Your code here
    
    return res

# part2
def build_loop_motion(bvh_motion:BVHMotion, ratio:float, half_life:float) -> BVHMotion:
    '''
    输入: 将要loop化的动作，类型为BVHMotion
          damping在前在后的比例ratio, ratio介于[0,1]
          弹簧振子damping效果的半衰期 half_life
          如果你使用的方法不含上面两个参数，就忽视就可以了，因接口统一保留
    输出: loop化后的动作，类型为BVHMotion
    
    Tips:
        1. 计算第一帧和最后一帧的旋转差、Root Joint位置差 (不用考虑X和Z的位置差)
        2. 如果使用"inertialization"，可以利用`smooth_utils.py`的
        `quat_to_avel`函数计算对应角速度的差距，对应速度的差距请自己填写
        3. 逐帧计算Rotations和Postions的变化
        4. 注意 BVH的fps需要考虑，因为需要算对应时间
        5. 可以参考`smooth_utils.py`的注释或者 https://theorangeduck.com/page/creating-looping-animations-motion-capture
    
    '''
    res = bvh_motion.raw_copy()
    ### Your code here
    fps = 60
    # rotations变化
    rotations = res.joint_rotation
    avel = quat_to_avel(rotations, 1 / fps)
    
    # 第一帧和最后一帧的旋转差
    rot_diff = (R.from_quat(rotations[-1]) * R.from_quat(rotations[0].copy()).inv()).as_rotvec()
    avel_diff = (avel[-1] - avel[0])
    
    for i in range(res.motion_length):
        offset1 = decay_spring_implicit_damping_rot(
            (1 - ratio) * rot_diff, (1 - ratio) * avel_diff, half_life, i / fps
        )
        offset2 = decay_spring_implicit_damping_rot(
            -(1 - ratio) * rot_diff, (1 - ratio) * avel_diff, half_life, (res.motion_length - i - 1) / fps
        )
        offset_rot = R.from_rotvec(offset1[0] + offset2[0])
        res.joint_rotation[i] = (offset_rot * R.from_quat(rotations[i])).as_quat() 
    
    # positions变化
    pos_diff = res.joint_position[-1] - res.joint_position[0]
    pos_diff[:,[0,2]] = 0
    # 第一帧和最后一帧的速度差
    vel1 = res.joint_position[-1] - res.joint_position[-2]
    vel2 = res.joint_position[1] - res.joint_position[0]
    vel_diff = (vel1 - vel2) / fps
    
    for i in range(res.motion_length):
        offset1 = decay_spring_implicit_damping_pos(
            (1 - ratio) * pos_diff, (1 - ratio) * vel_diff, half_life, i / fps
        )
        offset2 = decay_spring_implicit_damping_pos(
            -(1 - ratio) * pos_diff, (1 - ratio) * vel_diff, half_life, (res.motion_length - i - 1) / fps
        )
        offset_pos = offset1[0] + offset2[0]
        res.joint_position[i] += offset_pos
    ### Your code here
    return res


# part3
def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, mix_frame1:int, mix_time:int):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第mix_frame1帧
          mix_time表示用于混合的帧数
    输出: 平滑地连接后的动作，类型为BVHMotion
    
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    res = bvh_motion1.raw_copy()
    
    ### Your code here
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
    
    
    
    # # linear blending
    # # 对齐
    # pos = bvh_motion1.joint_position[mix_frame1 + mix_time, 0, [0, 2]]
    # rot = bvh_motion1.joint_rotation[mix_frame1 + mix_time, 0]
    # facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
    # new_bvh_motion2 = bvh_motion2.translation_and_rotation(0, pos, facing_axis)

    # # 动画拼接
    # blending_joint_position = np.zeros((mix_time, new_bvh_motion2.joint_position.shape[1], new_bvh_motion2.joint_position.shape[2]))
    # blending_joint_rotation = np.zeros((mix_time, new_bvh_motion2.joint_rotation.shape[1], new_bvh_motion2.joint_rotation.shape[2]))
    # blending_joint_rotation[..., 3] = 1.0
    
    # for i in range(mix_time):
    #     t = i / mix_time
    #     blending_joint_position[i] = (1-t) * res.joint_position[mix_frame1] + t * new_bvh_motion2.joint_position[0]
    #     for j in range(len(res.joint_rotation[mix_frame1])):
    #         rotation_walk = res.joint_rotation[mix_frame1, j,:]
    #         rotation_run = new_bvh_motion2.joint_rotation[0, j, :]
    #         cos_theta = np.dot(rotation_walk, rotation_run)
            
    #         if cos_theta < 0.:
    #             cos_theta = -cos_theta
    #             rotation_walk = -rotation_walk
            
    #         theta = np.arccos(cos_theta)
    #         sin_theta = np.sin(theta)
            
    #         if sin_theta > 0.0001:
    #             alpha_1 = np.sin((1 - t) * theta) / sin_theta
    #             alpha_2 = np.sin(t * theta) / sin_theta
    #         else:
    #             alpha_1 = 1 - t
    #             alpha_2 = t
    #         blending_joint_rotation[i, j] = alpha_1 * rotation_walk + alpha_2 * rotation_run
    #         blending_joint_rotation[i, j] /= np.linalg.norm(blending_joint_rotation[i, j])
            
    # new_bvh_motion2.joint_position = np.concatenate([blending_joint_position,  new_bvh_motion2.joint_position], axis=0)
    # new_bvh_motion2.joint_rotation = np.concatenate([blending_joint_rotation,  new_bvh_motion2.joint_rotation], axis=0)


    res.joint_position = np.concatenate([res.joint_position[:mix_frame1],  new_bvh_motion2.joint_position], axis=0)
    res.joint_rotation = np.concatenate([res.joint_rotation[:mix_frame1],  new_bvh_motion2.joint_rotation], axis=0)
    ### Your code here

    return res

