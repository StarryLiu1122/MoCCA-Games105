
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
    
    #######################
    
    return J, rhs