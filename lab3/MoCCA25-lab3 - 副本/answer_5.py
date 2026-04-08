import numpy as np
from scipy.spatial.transform import Rotation

from joints import Joint

from typing import List, Tuple

def PD_control(x:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray, joints:List[Joint],
               kp:np.ndarray|float, kd:np.ndarray|float, target_joint_rotations:np.ndarray
               ) -> np.ndarray:
    ''' 关节 PD 控制
    
    给出当前刚体状态、关节定义，计算每个关节的控制力矩，使得机械臂的关节旋转接近目标旋转。
    
    输入：
        x: (num_bodies, 3)    刚体（质心）的位置
        R: (num_bodies, 3, 3) 刚体的朝向，表示为一组矩阵
        v: (num_bodies, 3)    刚体的线速度
        w: (num_bodies, 3)    刚体的角速度
        joints:               关节定义列表，包含 num_joints 个关节
        
        kp: (num_joints) | float 每个关节的比例增益
        kd: (num_joints) | float 每个关节的微分增益
        target_joint_rotations: (num_joints, 3, 3) 每个关节的目标旋转，用旋转矩阵表示
    
    输出：
        joint_torques: (num_joints, 3) 每个关节的控制力矩
        
        注意，我们假设 bodyA 是父刚体，bodyB 是子刚体。joint_torques[j] 是施加在 bodyB 上的力矩，
        bodyA 获得的力矩为 -joint_torques[j]。
    
    提示：
        * 你需要计算每个关节的控制力矩 tau，使得关节旋转接近目标旋转
        * 可以使用 PD 控制公式：tau = kp * (q_target - q_current) - kd * qdot        
        * 注意对于三维旋转，该如何计算 (q_target - q_current) 并将其转换为 3D 向量？
            提示：一种可能方案是：用 R_target * R_current.T 计算偏差，并转换为轴角
        * 注意最终关节力矩应该为世界坐标系下表示
            思考：用上一个提示中计算的偏差，是在哪个坐标系下的表示？
            思考：如果偏差使用上一个提示中的方法计算，关节旋转速度 qdot 应该如何计算？
    '''
    
    num_bodies = x.shape[0]
    num_joints = len(joints)
    
    bodyA = [jnt.bodyA for jnt in joints]
    bodyB = [jnt.bodyB for jnt in joints]
            
    R_bodyA, R_bodyB = R[bodyA,:,:], R[bodyB,:,:]
    w_bodyA, w_bodyB = w[bodyA,:], w[bodyB,:]
    
    joint_torques = np.zeros((num_joints, 3))
    
    ######## 你的回答 #######
    
    #########################
    
    return joint_torques
    