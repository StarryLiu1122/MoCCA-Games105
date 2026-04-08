##############
# 姓名：刘星云
# 学号：2300012297
##############

import utils
from joints import Joint

import numpy as np

from typing import List, Tuple



def hinge_Jacobian(x:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray, joints:List[Joint]) -> Tuple[np.ndarray, np.ndarray]:
    ''' 计算 hinge 关节约束的 Jacobian 矩阵
    
        给出一组刚体的位置、朝向，以及一组关节的定义（参见 joints.py 中 Joint 类的定义）
        
        计算每个关节 hinge 约束 Jacobian 矩阵
            
        输入：
            x: (num_bodies, 3)    刚体（质心）的位置
            R: (num_bodies, 3, 3) 刚体的朝向，表示为一组矩阵
            v: (num_bodies, 3)    刚体的（质心）线速度
            w: (num_bodies, 3)    刚体的角速度
            joints:             关节定义列表，包含 num_joints 个关节
            
        x,R,v,w 均为世界坐标系表示
                    
        输出：
            J: (num_joints, 2, 12)  所有 hinge 关节约束的Jacobian矩阵，对应速度向量 [bodyA_v, bodyA_w, bodyB_v, bodyB_w]
            rhs: (num_joints, 2)    所有 hinge 关节约束的右端项，对应关节约束的修正项
                
        提示：            
            * 你可能不需要用到所有的输入信息
            * 关节位置的约束已经在 ball_Jacobian 函数中完成，本函数只需要提供额外的旋转自由度约束
            * hinge关节要求两个刚体的相对旋转只能发生在 hinge_axis 方向，也即在与 hinge_axis 垂直的方向上的相对转速分量必须是 0
              思考一下该如何实现？
            * 同样的，你可以计算对应的 rhs 向量，表示关节约束的修正项
    '''
    
    num_bodies = x.shape[0]
    num_joints = len(joints)
    
    bodyA = [jnt.bodyA for jnt in joints]
    bodyB = [jnt.bodyB for jnt in joints]
    
    from_bodyA = np.array([jnt.from_bodyA for jnt in joints])
    from_bodyB = np.array([jnt.from_bodyB for jnt in joints])
    
    hinge_axes = np.array([jnt.hinge_axis for jnt in joints])
    
    x_bodyA, x_bodyB = x[bodyA,:], x[bodyB,:]    
    R_bodyA, R_bodyB = R[bodyA,:,:], R[bodyB,:,:]    
        
    J = np.zeros((num_joints, 2, 12))
    rhs = np.zeros((num_joints, 2))
    
    ####### 你的回答 #######
    for i in range(num_joints):
        A = bodyA[i]
        B = bodyB[i]
        axis_world = R[A] @ hinge_axes[i]

        # 构造两个正交方向
        if np.abs(axis_world[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        n1 = np.cross(axis_world, ref)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(axis_world, n1)
        n2 /= np.linalg.norm(n2)

        # 设置 Jacobian
        J[i, 0, 3:6] = n1      # bodyA_w
        J[i, 0, 9:12] = -n1    # bodyB_w
        J[i, 1, 3:6] = n2
        J[i, 1, 9:12] = -n2

        # 设置 rhs
        alpha = 1
        rel_w = w[A] - w[B]
        rhs[i, 0] = -alpha * np.dot(n1, rel_w)
        rhs[i, 1] = -alpha * np.dot(n2, rel_w)
            

        # # 计算旋转轴误差和相对角速度
        # hinge_axis_A = R_bodyA[i] @ hinge_axes[i]  
        # hinge_axis_B = R_bodyB[i] @ hinge_axes[i]
        # axis_error = np.cross(hinge_axis_A, hinge_axis_B)
        # rel_w = w[A] - w[B]

        # # 设置 rhs（修正旋转偏差 + 阻尼）
        # alpha = 1
        # beta = 0   
        # rhs[i, 0] = -beta * np.dot(n1, axis_error) - alpha * np.dot(n1, rel_w)
        # rhs[i, 1] = -beta * np.dot(n2, axis_error) - alpha * np.dot(n2, rel_w)
    
    #######################
    
    return J, rhs