import numpy as np
import pyvista as pv

import utils

from answer_2 import create_joint

from typing import List, Tuple

def create_triple_pendulum(args, *, zero_gravity:bool=False, no_root:bool=False):
    ''' 创建一个三摆及其初始状态
    '''
    
    # 三个刚体的初始状态。注意第四个刚体是一个不动的虚拟刚体，用于表示空间本身，连接到虚拟刚体的关节将会固定在初始位置旋转
    x = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0], [7.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    R = np.stack([np.eye(3)]*4, axis=0)
    v = np.zeros((4, 3))
    w = np.zeros((4, 3))
    
    # 刚体的质量和转动惯量
    m = np.array([1.0, 1.5, 2.0, 0.0]) # 第四个刚体的质量为0，表示空间本身
    I = np.stack([np.diag(np.array([1.0, 2.0, 4.0]) * i) for i in [0.1, 0.2, 0.3, 0.0]], axis=0) # 第四个刚体的转动惯量为0，表示空间本身
    
    # 以下两个量是为了方便后续计算
    inv_m = np.concatenate((1/m[:-1], np.zeros((1,))), axis=0) # 第四个刚体的质量为0，表示空间本身
    inv_I = np.concatenate((np.linalg.inv(I[:-1,:,:]), np.zeros((1, 3, 3))), axis=0) # 第四个刚体的转动惯量为0，表示空间本身
    
    # 关节的初始位置及定义
    joints = [
        create_joint(x, R, [0.0, 0.0, 0.0], -1, 0), # 固定在世界坐标系
        create_joint(x, R, [4.0, 0.0, 0.0], 0, 1), # 连接刚体0和刚体1
        create_joint(x, R, [6.0, 0.0, 0.0], 1, 2), # 连接刚体1和刚体2
        ]
    
    if no_root:
        joints.pop(0) # 删除第一个关节
        # 这样就只剩下两个关节了，分别连接刚体0和刚体1，以及刚体1和刚体2
    
    # 时间步长
    h = 0.001
    
    # 力和力矩，刚体只受到重力作用
    g = np.array([0, 0, -9.81])
    if zero_gravity:
        g = np.zeros(3)
        
    f = m.reshape(-1, 1) * g.reshape(1, 3) # 重力作用在质心上
    tau = np.zeros((4, 3))
    
    # 各刚体的形状，用于可视化
    box_meshes = [pv.Box() for i in range(3)]
    # 默认形状为2x2x2的立方体
    box_vertices = np.stack([ box.points for box in box_meshes ], axis=0).reshape(3, -1, 3, 1)
    
    box_vertices[...,1:,0] *= 0.1 # 让每个刚体的长、宽为0.1
    box_vertices[0,:,0,0] *= 2 # 让第一个刚体的长为4
    
    # 可视化：
    plotter = pv.Plotter()
    for box in box_meshes:
        plotter.add_mesh(box, color="tan", show_edges=True)        
        
    ground_box = pv.Box(bounds=[-5,5,-5,5,-8,-9])
    plotter.add_mesh(ground_box, color='#eefefe', show_edges=True)    
    
    # 一些显示设置
    plotter.add_axes()    
    plotter.show(auto_close=False, interactive_update=True)
    plotter.view_xz(bounds=[-6, 6, -0.5, 0.5, -6, 1])
        
    utils._update_boxes(x[:-1], R[:-1], box_meshes, box_vertices, plotter)
        
    if args.video is not None:
        if args.video.endswith('.gif'):
            plotter.open_gif(args.video, fps=30)
        else:
            plotter.open_movie(args.video, framerate=30, quality=10)
    
    
    # 返回构建的信息
    return (m, I, inv_m, inv_I, x, R, v, w, f, tau, h, g, joints, box_meshes, box_vertices, plotter)


def forward_kinematics(num_body, x_0, joint_rots, joints) -> Tuple[np.ndarray, np.ndarray]:
    ''' 根据关节的相对旋转，计算每个刚体的位置和朝向
    
    输入：
        x_0: 世界坐标系下的根节点位置
        joint_rots: 各关节旋转
        joints: 关节列表
        
    输出：
        x：各刚体在世界坐标系下的位置
        R：各刚体在世界坐标系下的朝向
    
    注意，我们假设 bodyA 是父关节，bodyB 是子关节
    '''
    
    num_joints = len(joints)
    
    bodyA = [jnt.bodyA for jnt in joints]
    bodyB = [jnt.bodyB for jnt in joints]
    
    x = np.zeros((num_body, 3))
    R = np.stack([np.eye(3)] * num_body, axis=0)
    
    jnt_pos = x_0
    jnt_ori = joint_rots[0]
    
    for j, jnt in enumerate(joints):
        if jnt.bodyA >= 0:
            jnt_pos = x[jnt.bodyA] + jnt_ori @ jnt.from_bodyA
            jnt_ori = jnt_ori @ joint_rots[j]
        
        i = jnt.bodyB
        
        x[i] = jnt_pos - jnt_ori @ jnt.from_bodyB
        R[i] = jnt_ori
        
    return x, R

