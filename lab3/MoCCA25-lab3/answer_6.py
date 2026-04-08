##############
# 姓名：刘星云
# 学号：2300012297
##############
import numpy as np
from scipy.spatial.transform import Rotation

from joints import Joint

from typing import List, Tuple

from answer_5 import PD_control
from answer_4 import gravity_compensation


def end_effector_track_control(
    m:np.ndarray, I:np.ndarray, g:np.ndarray, 
    x:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray, 
    joints:List[Joint], 
    end_effector_body:int, end_effector_from_body:np.ndarray,
    target_end_effector_pos:np.ndarray, target_end_effector_vel:np.ndarray,
    ) -> np.ndarray:
    ''' 末端点位置控制器
    
    计算每个关节的控制力矩，使得机械臂的末端点位置跟踪目标位置。

    输入：
        m: (num_bodies)       刚体的质量
        I: (num_bodies, 3, 3) 刚体的惯性矩阵
        g: (3)                重力加速度
        x: (num_bodies, 3)    刚体（质心）的位置
        R: (num_bodies, 3, 3) 刚体的朝向，表示为一组矩阵
        v: (num_bodies, 3)    刚体的线速度
        w: (num_bodies, 3)    刚体的角速度
        joints:               关节定义列表，包含 num_joints 个关节
        
        end_effector_body: int       末端点所在的刚体索引
        end_effector_from_body: (3)  末端点在刚体坐标系下的位置
        target_end_effector_pos: (3) 目标末端点位置
        target_end_effector_vel: (3) 目标末端点速度
    
    输出：
        joint_torques: (num_joints, 3) 每个关节的控制力矩
        
        注意，我们假设 bodyA 是父刚体，bodyB 是子刚体。joint_torques[j] 是施加在 bodyB 上的力矩，
        bodyA 获得的力矩为 -joint_torques[j]。
        
    提示：
        * 你可以综合使用 PD_control 和 gravity_compensation 来实现末端点位置控制器，也可以使用你认为合适的其他方法
        * 可以考虑的实现思路：
            1. 基于 Jacobian transpose 方法
                a. 计算重力补偿力矩 tau_g，使得机械臂保持静止
                b. 根据当前末端点位置和目标末端点位置，利用 PD 控制计算虚拟力 f_end，使得末端点位置跟踪目标位置
                c. 利用 Jacobian transpose 方法将 f_end 转换为关节力矩 tau
            2. 基于 PD 控制方法
                a. 根据目标末端点位置，利用 IK 计算目标关节旋转
                b. 计算关节 PD 控制力矩 tau，跟踪 IK 得到的目标关节旋转
            3. 基于 PD + 重力补偿方法
                a. 在 2 中的基础上，计算重力补偿力矩 tau_g，使得机械臂保持静止
            4. 其他方法
                a. 你可以使用其他方法来实现末端点位置控制器，比如基于模型预测控制的方法等
    '''
    
    num_bodies = x.shape[0]
    num_joints = len(joints)
        
    # 计算当前末端点位置和速度
    end_effector_from_body_global = R[end_effector_body] @ end_effector_from_body
        
    current_end_effector_pos = x[end_effector_body] + end_effector_from_body_global
    current_end_effector_vel = v[end_effector_body] + np.cross(w[end_effector_body], end_effector_from_body_global)
    
    # 需要施加的关节力矩
    joint_torques = np.zeros((num_joints, 3))
    
    ######## 你的回答 #######
    
    # a. 重力补偿力矩
    tau_gravity = np.zeros((num_joints, 3))
    tau_gravity = gravity_compensation(m, g, x, R, joints)
    
    # b. 末端点 PD 控制力
    kp = 400.0 
    kd = 20.0 
    pos_error = target_end_effector_pos - current_end_effector_pos
    vel_error = target_end_effector_vel - current_end_effector_vel
    f_end = kp * pos_error + kd * vel_error 
    
    # c. Jacobian transpose
    tau_pd = np.zeros((num_joints, 3))
    for j in range(num_joints):
        p = x[joints[j].bodyA] + R[joints[j].bodyA] @ joints[j].from_bodyA
        tau_pd[j] = np.cross((current_end_effector_pos - p), f_end)
    
    
    joint_torques = tau_gravity + tau_pd

            
    ########################
    
    return joint_torques