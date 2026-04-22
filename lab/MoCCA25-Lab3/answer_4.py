import numpy as np
from scipy.spatial.transform import Rotation

from joints import Joint

from typing import List, Tuple

def gravity_compensation(m:np.ndarray, g:np.ndarray, x:np.ndarray, R:np.ndarray, joints:List[Joint]) -> np.ndarray:
    ''' 重力补偿
    
    给出当前的刚体位置、朝向以及关节定义，计算每个关节的重力补偿力矩，使得整个机械臂保持静止
    
    输入：
        m: (num_bodies)       刚体的质量
        g: (3)                重力加速度
        x: (num_bodies, 3)    刚体（质心）的位置
        R: (num_bodies, 3, 3) 刚体的朝向，表示为一组矩阵
        joints:               关节定义列表，包含 num_joints 个关节
    
    输出：
        joint_torques: (num_joints, 3)   每个关节的重力补偿力矩
        
        注意，我们假设 bodyA 是父刚体，bodyB 是子刚体。joint_torques[j] 是施加在 bodyB 上的力矩，
        bodyA 获得的力矩为 -joint_torques[j]。
        
    提示：
        * 你需要计算每个关节的重力补偿力矩 tau，使得整个机械臂保持静止
        * 可以使用逆动力学或者基于Jacobian Transpose的方法来计算重力补偿力矩
    '''
    
    
    num_bodies = x.shape[0]
    num_joints = len(joints)
    
    bodyA = [jnt.bodyA for jnt in joints]
    bodyB = [jnt.bodyB for jnt in joints]
    
    from_bodyA = np.array([jnt.from_bodyA for jnt in joints])
    from_bodyB = np.array([jnt.from_bodyB for jnt in joints])
        
    x_bodyA, x_bodyB = x[bodyA,:], x[bodyB,:]    
    R_bodyA, R_bodyB = R[bodyA,:,:], R[bodyB,:,:]
    
    joint_torques = np.zeros((num_joints, 3))
    
    ######## 你的回答 #######
    
    ########################
    
    return joint_torques