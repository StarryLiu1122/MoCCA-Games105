##############
# 姓名：刘星云
# 学号：2300012297
##############

import scipy.linalg
import utils
from joints import Joint

import numpy as np
import scipy

from typing import List, Tuple
from answer_1 import forward_dynamics
from answer_3 import hinge_Jacobian

def create_joint(x:np.ndarray, R:np.ndarray, joint_pos:np.ndarray, bodyA:int, bodyB:int) -> Joint:
    ''' 根据输入信息创建关节定义
    
        给出一组刚体的位置、朝向，以及一组关节的信息，创建关节实例（参见 joints.py 中 Joint 类的定义）
        
        输入：
            x: (num_bodies, 3)      刚体（质心）的位置
            R: (num_bodies, 3, 3)   刚体的朝向，表示为一组矩阵
            joint_pos: (3)          关节的位置
            bodyA: int              关节连接的刚体A的index
            bodyB: int              关节连接的刚体B的index
                    
        x,R,joint_pos 均为世界坐标系表示
        
        输出：
            一个 Joint 实例，包含关节的定义信息
            
        提示：
            * 你需要计算每个关节的 from_bodyA, from_bodyB，即关节在对应刚体局部坐标系下的位置
            * from_bodyA, from_bodyB 表示同一个关节在两个刚体局部坐标系下的位置
            * 通过 joint_def = Joint(bodyA, bodyB, from_bodyA, from_bodyB) 生成关节定义
    '''
    
    
    ####### 你的回答 #######
    # 提示：需要计算每个关节的 from_bodyA, from_bodyB，即关节在对应刚体局部坐标系下的位置
    # 通过 joint_def = Joint(bodyA, bodyB, from_bodyA, from_bodyB) 生成关节定义
    
    xA = x[bodyA]
    xB = x[bodyB]
    RA = R[bodyA]
    RB = R[bodyB]
    
    from_bodyA = RA.T @ (joint_pos - xA)
    from_bodyB = RB.T @ (joint_pos - xB)
    joint = Joint(bodyA, bodyB, from_bodyA, from_bodyB)
    
    #######################
    
    return joint
        

def ball_Jacobian(x:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray, joints:List[Joint]) -> Tuple[np.ndarray, np.ndarray]:
    ''' 计算关节约束的 Jacobian 矩阵
    
        给出一组刚体的位置、朝向，以及一组关节的定义（参见 joints.py 中 Joint 类的定义）
        
        计算每个关节的约束 Jacobian 矩阵
            
        输入：
            x: (num_bodies, 3)    刚体（质心）的位置
            R: (num_bodies, 3, 3) 刚体的朝向，表示为一组矩阵
            v: (num_bodies, 3)    刚体的（质心）线速度
            w: (num_bodies, 3)    刚体的角速度
            joints:               关节定义列表，包含 num_joints 个关节
            
        x,R,v,w 均为世界坐标系表示
                    
        输出：
            J: (num_joints, 3, 12)  所有关节约束的Jacobian矩阵，对应速度向量 [bodyA_v, bodyA_w, bodyB_v, bodyB_w]
            rhs: (num_joints, 3)     关节约束的右端项，表示为一个修正项，用于修正仿真误差导致的关节错位
                
        提示：
            * 你需要计算每个关节的 Jacobian 矩阵，参考 ppt 中相关内容
            * 关节约束的速度表示为 J v = rhs，其中 rhs=0 或者是一个修正项，用于修正仿真误差导致的关节错位，参考 ppt 中关于 ERP 的内容
            * 你可能不需要用到所有的输入信息
    '''
    
    num_bodies = x.shape[0]
    num_joints = len(joints)
    
    bodyA = [jnt.bodyA for jnt in joints]
    bodyB = [jnt.bodyB for jnt in joints]
    
    from_bodyA = np.array([jnt.from_bodyA for jnt in joints])
    from_bodyB = np.array([jnt.from_bodyB for jnt in joints])
    
    x_bodyA, x_bodyB = x[bodyA,:], x[bodyB,:]    
    R_bodyA, R_bodyB = R[bodyA,:,:], R[bodyB,:,:]    
        
    J = np.zeros((num_joints, 3, 12))
    rhs = np.zeros((num_joints, 3))
    
    ####### 你的回答 #######
    for i, joint in enumerate(joints):
        A = bodyA[i]
        B = bodyB[i]

        rA = R[A] @ from_bodyA[i]
        rB = R[B] @ from_bodyB[i]
        
        skew_rA = np.array([
            [0, -rA[2], rA[1]],
            [rA[2], 0, -rA[0]],
            [-rA[1], rA[0], 0]
        ])
        skew_rB = np.array([
            [0, -rB[2], rB[1]],
            [rB[2], 0, -rB[0]],
            [-rB[1], rB[0], 0]
        ])

        # Jacobian 矩阵： [ I, -[rA]_x, -I, [rB]_x ]
        J[i, :, 0:3] = np.eye(3)      
        J[i, :, 3:6] = -skew_rA      
        J[i, :, 6:9] = -np.eye(3)     
        J[i, :, 9:12] = skew_rB        

        alpha = 1
        h = 0.001   
        pA = x[A] + rA  
        pB = x[B] + rB  
        position_error = pB - pA   
        rhs[i] = (alpha / h) * position_error
        
        # alpha = 0.2
        # vA = v[A] + np.cross(w[A], rA)
        # vB = v[B] + np.cross(w[B], rB)
        # position_error = vA - vB   
        # # rhs[i] = (alpha / h) * position_error
        # rhs[i] = alpha * position_error
        
    #######################
    
    return J, rhs

def forward_dynamics_with_constraints(
    m:np.ndarray, I:np.ndarray, inv_m:np.ndarray, inv_I:np.ndarray,
    x:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray, 
    f:np.ndarray, tau:np.ndarray, h:float,
    joints:List[Joint]
    ) -> Tuple[np.ndarray, np.ndarray]:
    ''' 带约束的刚体前向运动学函数
    
        给出一组刚体的位置、朝向，以及当前速度、角速度，
        给出一组关节的定义，利用欧拉积分计算下一时刻刚体的位置和朝向
    
        输入：
            m: (num_bodies, )         刚体的质量
            I: (num_bodies, 3, 3)     刚体的转动惯量
            inv_m: (num_bodies, )     刚体的质量倒数
            inv_I: (num_bodies, 3, 3) 刚体的转动惯量倒数
            
            x: (num_bodies, 3)        刚体（质心）的位置
            R: (num_bodies, 3, 3)     刚体的朝向，表示为一组矩阵
            v: (num_bodies, 3)        刚体的（质心）线速度
            w: (num_bodies, 3)        刚体的角速度
            
            f: (num_bodies, 3)        施加在每个刚体质心上的力
            tau: (num_bodies, 3)      施加在每个刚体上的力矩
            h: ()                     时间步长
            
            joints:             关节定义列表，包含 num_joints 个关节
        
        所有量均为世界坐标系表示
        
        输出：
            v_next: (num_bodies, 3)        刚体下一时刻的速度
            w_next: (num_bodies, 3)        刚体下一时刻的角速度
            
        提示：
            * 使用带约束的 Newton-Euler 方程来求解下一时刻刚体的状态，参考ppt中相关内容
            * 输入包括了场景中所有的刚体，你可以使用 for 循环来处理每个刚体（不推荐），或者
              使用 numpy 的向量化操作来处理所有刚体
            * 你可能不需要用到所有的输入信息
            * 你需要使用 ball_Jacobian 函数来计算关节约束的 Jacobian 矩阵
    '''
        
        
    num_bodies = x.shape[0]
    num_joints = len(joints) 
    
    hinge_joints = [jnt for jnt in joints if jnt.hinge_axis is not None]
    num_hinge_joints = len(hinge_joints)
    
    v_next, w_next = v.copy(), w.copy()
    
    ####### 你的回答 #######
    #------------------------------------------------------------------
    # 以下是一个示例框架，你可以在其基础上修改，也可以删除改为你自己的实现
    # 注意：这个框架的效率比较低，更高效的计算需要考虑质量矩阵、约束 Jacobian 矩阵的稀疏性
    
    # 运动方程的矩阵形式： 由牛顿欧拉方程推导而来，考虑加速度的近似： a = (v_next - v) / h
    #   M * [v_next, w_next] = M * [v, w] - h * [0, w x Iw] + h * [f, tau] + h * J^T * lambda
    # 可以得到
    #   [v_next, w_next] = ([v, w] + h * M^-1 * ([f, tau] - [0, w x Iw])) + h * M^-1 * J^T * lambda
    #                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                    = >[v0, w0]< + h * M^-1 * J^T * lambda
    # 如果你完成了问题1中的 forward_dynamics函数，你会发现第一项 [v0, w0] 就是 forward_dynamics 函数的输出
    # 约束方程为：
    #   J * [v_next, w_next] = rhs
    # 带入上面的结果，并整理可以得到
    #   A lambda = b
    # 其中 A = h * J * M^-1 * J^T
    #      b = rhs - J * [v0, w0]
    # 为了数值稳定，我们可以对 A 进行修正： diag(A) += cfms
    # 进而求解得到 
    #   lambda = A^-1 b
    # 然后带入上面的结果，得到 [v_next, w_next] = [v0, w0] + h * M^-1 * J^T * lambda    
    #------------------------------------------------------------------
        
    #------------------------------------------------------------------
    # [1]. 构造所有刚体的质量矩阵，M = diag([m_1, I_1,... m_n, I_n])，及其逆矩阵
    #------------------------------------------------------------------
    M = np.zeros((num_bodies*6, num_bodies*6))
    inv_M = np.zeros((num_bodies*6, num_bodies*6))
    # 在对角线上填充质量和转动惯量及其倒数
    # 注意转动惯量 I 及其逆 inv_I 需要随着旋转进行相应的变换
    # 你可以考虑使用 scipy.linalg.block_diag 来构造对角矩阵
    
    ##### 你的代码 #####
    for i in range(num_bodies):
        mass = m[i]
        inv_mass = inv_m[i]
        R_i = R[i]

        # 变换到世界坐标系
        I_world = R_i @ I[i] @ R_i.T
        inv_I_world = R_i @ inv_I[i] @ R_i.T

        idx = i * 6
        M[idx:idx+3, idx:idx+3] = np.eye(3) * mass
        M[idx+3:idx+6, idx+3:idx+6] = I_world
        inv_M[idx:idx+3, idx:idx+3] = np.eye(3) * inv_mass
        inv_M[idx+3:idx+6, idx+3:idx+6] = inv_I_world
    ###################
    
    
    #------------------------------------------------------------------
    # [2]. 计算无约束下刚体的速度和角速度 [v0, w0]
    #------------------------------------------------------------------
    # 你可以使用 forward_dynamics 来帮你计算，也可以复制代码到这里，因为后面的计算可能需要用到一些中间变量
    
    ##### 你的代码 #####
    v0, w0 = forward_dynamics(m, I, inv_m.reshape(-1, 1), inv_I, x, R, v, w, f, tau, h)
    ###################    
    
    
    #------------------------------------------------------------------
    # [3]. 构造所有约束的 Jacobian 矩阵 J 和右端项 rhs
    #------------------------------------------------------------------
    constraint_dim = num_joints * 3
    constraint_dim += num_hinge_joints * 2 # hinge 关节的额外约束
    
    J = np.zeros((constraint_dim, num_bodies*6))
    rhs = np.zeros((constraint_dim, 1))
    
    # 计算每个约束 Jacobian 矩阵 J 和右端项 rhs
    J_ball, rhs_ball = ball_Jacobian(x, R, v, w, joints)
    if num_hinge_joints > 0:
        J_hinge, rhs_hinge = hinge_Jacobian(x, R, v, w, hinge_joints)
    
    # 将 J_ball, rhs_ball, J_hinge, rhs_hinge 填充到 J 和 rhs 中
    # 注意 J_ball, J_hinge 的大小都是 (nj, dof, 12), 如果你的实现正确，每个约束矩阵的最后一维的前 6 列应该对应 bodyA，后 6 列应该对应 bodyB
    #   其中对于 J_ball， dof = 3；对于 J_hinge， dof = 2，你需要将对应的矩阵块填充到 J 中的合适位置
    # 注意 rhs_ball, rhs_hinge 的大小都是 (nj, dof)，你可能需要 reshape 调整它们的形状    
    
    ##### 你的代码 #####
    row_cursor = 0

    # 处理 ball joints
    for i in range(num_joints):
        dof = 3 
        # J_ball[i]: (3, 12)，前 6 列是 bodyA，后 6 列是 bodyB
        bodyA = joints[i].bodyA
        bodyB = joints[i].bodyB
        
        # if bodyA == -1:
        #     bodyA = num_bodies - 1
        # if bodyB == -1:
        #     bodyB = num_bodies - 1

        # 将 J_ball[i] 的前 6 列填入对应的 bodyA 列
        if bodyA != -1:
            J[row_cursor:row_cursor+dof, bodyA*6:bodyA*6+6] = J_ball[i][:, 0:6]
        # 将 J_ball[i] 的后 6 列填入对应的 bodyB 列
        if bodyB != -1:
            J[row_cursor:row_cursor+dof, bodyB*6:bodyB*6+6] = J_ball[i][:, 6:]
        # 填充 rhs
        rhs[row_cursor:row_cursor+dof, 0] = rhs_ball[i].reshape(-1)

        row_cursor += dof

    # 处理 hinge joints
    if num_hinge_joints > 0:
        for i in range(num_hinge_joints):
            dof = 2  # hinge joint 每个有 2 个角度自由度约束
            bodyA = hinge_joints[i].bodyA
            bodyB = hinge_joints[i].bodyB
            
            # if bodyA == -1:
            #     bodyA = num_bodies - 1
            # if bodyB == -1:
            #     bodyB = num_bodies - 1
            
            if bodyA != -1:
                J[row_cursor:row_cursor+dof, bodyA*6:bodyA*6+6] = J_hinge[i][:, 0:6]
            if bodyB != -1:
                J[row_cursor:row_cursor+dof, bodyB*6:bodyB*6+6] = J_hinge[i][:, 6:]
            rhs[row_cursor:row_cursor+dof, 0] = rhs_hinge[i].reshape(-1)

            row_cursor += dof
    
    ###################
        
    #------------------------------------------------------------------
    # [4]. 构造约束方程的矩阵 A 和右端项 b
    #------------------------------------------------------------------
    A = np.zeros((constraint_dim, constraint_dim))
    b = np.zeros((constraint_dim, 1))
    
    ##### 你的代码 #####
    v0w0 = np.stack([v0, w0], axis=1).reshape(-1, 1)
    A = h * J @ inv_M @ J.T
    b = rhs - J @ v0w0
    ###################
    
    #------------------------------------------------------------------
    # [5]. 求解约束方程 A lambda = b，得到 lambda
    #------------------------------------------------------------------
    A += np.diag(np.ones(constraint_dim) * 1e-8) # 数值稳定性修正 cfm = 1e-8
    lambd = np.linalg.solve(A, b) 
    
    #------------------------------------------------------------------
    # [6]. 计算约束对速度和角速度的影响，得到 [v_next, w_next] = [v0, w0] + h * M^-1 * J^T * lambda
    #------------------------------------------------------------------
    
    ##### 你的代码 #####
    delta_vw = h * inv_M @ J.T @ lambd
    vw_next = v0w0 + delta_vw
    vw_next = vw_next.reshape(num_bodies, 6)  
    v_next = vw_next[:, :3]
    w_next = vw_next[:, 3:]

    ###################
    
    
    #######################
    
    return v_next, w_next