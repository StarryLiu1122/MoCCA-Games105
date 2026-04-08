##############
# 姓名：刘星云
# 学号：2300012297
##############

import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHMotion
from physics_warpper import PhysicsInfo


def part1_cal_torque(pose, physics_info: PhysicsInfo, **kargs):
    '''
    输入： pose: (20, 4)的numpy数组，表示每个关节的目标旋转(相对于父关节的)
           physics_info: PhysicsInfo类，包含了当前的物理信息，参见physics_warpper.py
           **kargs: 指定参数，可能包含kp,kd
    输出： global_torque: (20, 3)的numpy数组，表示每个关节的全局坐标下的目标力矩，因为不需要控制方向，根节点力矩会被后续代码无视
    '''
    # ------一些提示代码，你可以随意修改------------#
    kp = kargs.get('kp', 1000) # 需要自行调整kp和kd！ 而且也可以是一个数组，指定每个关节的kp和kd
    kd = kargs.get('kd', 20) 
    parent_index = physics_info.parent_index
    joint_name = physics_info.joint_name
    # 注意关节没有自己的朝向和角速度，这里用子body的朝向和角速度表示此时关节的信息
    joint_orientation = physics_info.get_body_orientation()
    # print(physics_info.get_root_pos_and_vel())
    parent_index = physics_info.parent_index
    joint_avel = physics_info.get_body_angular_velocity()
    target_rotation = pose

    global_torque = np.zeros((20,3))
    # 你的代码
    
    # orientation_error = np.zeros((20, 3))
    # global_error = np.zeros((20, 3))

    # q_current = R.from_quat(joint_orientation[0])
    # q_target = R.from_quat(pose[0])
    # q_error = q_target * q_current.inv()
    # orientation_error[0] = q_error.as_rotvec()  
    # global_error[0] = orientation_error[0]
    # # print(parent_index[0])
    
    # for i in range(1, len(joint_orientation)):
    #     # 当前旋转和目标旋转的差值
    #     q_current = R.from_quat(joint_orientation[parent_index[i]]).inv() * R.from_quat(joint_orientation[i])
    #     q_target = R.from_quat(pose[i])

    #     # 旋转误差
    #     q_error = q_target * q_current.inv()
    #     orientation_error[i] = q_error.as_rotvec()  
    #     global_error[i] = R.from_quat(joint_orientation[parent_index[i]]).as_matrix() @ orientation_error[i]
    
    # # PD控制力矩
    # # local_torque = kp * global_error - kd * joint_avel
    # max_torque = 500
    # local_torque = np.clip(kp * global_error - kd * joint_avel, -max_torque, max_torque)
    
    # global_torque = local_torque
    
    
    n = joint_orientation.shape[0]
    global_error = np.zeros((n, 3))

    R_joint = R.from_quat(joint_orientation)
    R_target = R.from_quat(pose)
    R_parent = R_joint[parent_index]

    R_current_local = R_parent.inv() * R_joint
    R_current_local[0] = R_joint[0]

    R_error = R_target * R_current_local.inv()

    orientation_error_local = R_error.as_rotvec() 

    R_parent_matrix = R_parent.as_matrix()   
    global_error = np.einsum('nij,nj->ni', R_parent_matrix, orientation_error_local)

    max_torque = 500
    local_torque = kp * global_error - kd * joint_avel
    local_torque = np.clip(local_torque, -max_torque, max_torque)
    global_torque = local_torque
    
    # 抹去根节点力矩
    global_torque[0] = np.zeros_like(global_torque[0])
    return global_torque


def part2_cal_float_base_torque(target_position, pose, physics_info, **kargs):
    '''
    输入： target_position: (3,)的numpy数组，表示根节点的目标位置，其余同上
    输出： global_root_force: (3,)的numpy数组，表示根节点的全局坐标下的辅助力，在后续仿真中只会保留y方向的力
           global_root_torque: (3,)的numpy数组，表示根节点的全局坐标下的辅助力矩，用来控制角色的朝向，实际上就是global_torque的第0项
           global_torque: 同上
    注意：
        1. 你需要自己计算kp和kd，并且可以通过kargs调整part1中的kp和kd
        2. global_torque[0]在track静止姿态时会被无视，但是track走路时会被加到根节点上，不然无法保持根节点朝向
        3. 可以适当将根节点目标位置上提以产生更大的辅助力，使角色走得更自然
    '''
    # ------一些提示代码，你可以随意修改------------#
    global_torque = part1_cal_torque(pose, physics_info)
    kp = kargs.get('root_kp', 4000) # 需要自行调整root的kp和kd！
    kd = kargs.get('root_kd', 20)
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    global_root_force = np.zeros((3,))
    global_root_torque = global_torque[0]
    # 你的代码
    
    kp1 = kargs.get('kp', 1000) 
    kd1 = kargs.get('kd', 20) 
    joint_orientation = physics_info.get_body_orientation()
    joint_avel = physics_info.get_body_angular_velocity()

    global_error = np.zeros((20, 3))
    q_current = R.from_quat(joint_orientation[0])
    q_target = R.from_quat(pose[0])
    q_error = q_target * q_current.inv()
    global_error[0] = q_error.as_rotvec()  
    local_torque = kp1 * global_error[0] - kd1 * joint_avel[0]
    max_torque = 100
    local_torque = np.clip(local_torque, -max_torque, max_torque)
    global_torque[0] = local_torque
    global_root_torque = global_torque[0]
    
    
    target_pos = target_position.copy()
    target_pos[1] += 0.1  # 上提，有助于角色保持站立/步行时不沉下去

    # PD控制
    position_error = target_pos - root_position
    velocity_error = -root_velocity
    global_root_force = kp * position_error + kd * velocity_error
    
    ########
    return global_root_force, global_root_torque, global_torque



frame_cnt = 0
def part3_cal_static_standing_torque(bvh: BVHMotion, physics_info):
    '''
    输入： bvh: BVHMotion类，包含了当前的动作信息，参见bvh_loader.py
    输出： 带反馈的global_torque: (20, 3)的numpy数组，因为不需要控制方向，根节点力矩会被无视
    Tips: 
        只track第0帧就能保持站立了
        为了保持平衡可以把目标的根节点位置适当前移，比如把根节点位置和左右脚的中点加权平均，但要注意角色还会收到一个从背后推他的外力
        可以定义一个全局的frame_count变量来标记当前的帧数，在站稳后根据帧数使角色进行周期性左右摇晃，如果效果好可以加分（0-20），可以考虑让角色站稳后再摇晃
    '''
    # ------一些提示代码，你可以随意修改------------#
    tar_pos = bvh.joint_position[0][0]
    pose = bvh.joint_rotation[0]
    joint_name = physics_info.joint_name
    
    joint_positions = physics_info.get_joint_translation()
    # 适当前移，这里的权重需要你自己调整
    tar_pos = tar_pos * 0.8 + joint_positions[9] * 0.1 + joint_positions[10] * 0.1
    # tar_pos = tar_pos * 0.9 + joint_positions[7] * 0.05 + joint_positions[8] * 0.05
    
    global frame_cnt
    # 修改根节点目标位置让角色摇摆起来，可以角色先站稳之后再周期性摇摆
    # tar_pos += 0.1 * np.sin((frame_cnt)/400)
    frame_cnt += 1
    
    torque = np.zeros((20,3))
    # 你的代码
    
    global_torque = part1_cal_torque(pose, physics_info, kp=300, kd=20) 
    joint_velocity = physics_info.get_body_velocity()
    joint_mass = physics_info.get_body_mass()
    # compute center of mass and COM velocity
    CoM = np.zeros(3)
    CoM_vel = np.zeros(3)
    mass = 0
    # 加权平均
    for i in range(len(joint_mass)):
        CoM += joint_mass[i] * joint_positions[i]
        CoM_vel += joint_mass[i] * joint_velocity[i]
        mass += joint_mass[i]
    CoM /= mass
    CoM_vel /= mass
    desired_CoM = tar_pos

    Kp = 200
    Kd = 20
    virtual_force = Kp * (desired_CoM - CoM) - Kd * CoM_vel
    
    torque = global_torque
    for i in range(0, len(torque)):
        torque[i] -= np.cross( CoM - joint_positions[i], -virtual_force)
    
    max_torque = 5000
    torque = np.clip(torque, -max_torque, max_torque)
    
    
    # 抹去根节点力矩
    torque[0] = np.zeros_like(torque[0])
    return torque



















def part3_cal_static_standing_torque(bvh: BVHMotion, physics_info):
    '''
    输入： bvh: BVHMotion类，包含了当前的动作信息，参见bvh_loader.py
    输出： 带反馈的global_torque: (20, 3)的numpy数组，因为不需要控制方向，根节点力矩会被无视
    Tips: 
        只track第0帧就能保持站立了
        为了保持平衡可以把目标的根节点位置适当前移，比如把根节点位置和左右脚的中点加权平均，但要注意角色还会收到一个从背后推他的外力
        可以定义一个全局的frame_count变量来标记当前的帧数，在站稳后根据帧数使角色进行周期性左右摇晃，如果效果好可以加分（0-20），可以考虑让角色站稳后再摇晃
    '''
    # ------一些提示代码，你可以随意修改------------#
    tar_pos = bvh.joint_position[0][0]
    pose = bvh.joint_rotation[0]
    joint_name = physics_info.joint_name
    
    joint_positions = physics_info.get_joint_translation()
    joint_orientations = physics_info.get_body_orientation()
    # 适当前移，这里的权重需要你自己调整
    # tar_pos = tar_pos * 0.8 + joint_positions[9] * 0.1 + joint_positions[10] * 0.1
    # tar_pos = tar_pos * 0.1 + joint_positions[9] * 0.45 + joint_positions[10] * 0.45
    # tar_pos = joint_positions[9] * 0.5 + joint_positions[10] * 0.5
    
    rend_site = joint_positions[9] + R.from_quat(joint_orientations[9]).apply([0.010, 0.002, 0.060])
    lend_site = joint_positions[10] + R.from_quat(joint_orientations[10]).apply([-0.010, 0.002, 0.060])
    
    # tar_pos = joint_positions[9] * 0.4 + joint_positions[10] * 0.4 + rend_site * 0.1 + lend_site * 0.1
    tar_pos = joint_positions[9] * 0.25 + joint_positions[10] * 0.25 + rend_site * 0.25 + lend_site * 0.25
    
    tar_pos[1] = bvh.joint_position[0][0][1] - 0.1
    # tar_pos[1] = bvh.joint_position[0][0][1]
    
    global frame_cnt
    # 修改根节点目标位置让角色摇摆起来，可以角色先站稳之后再周期性摇摆
    tar_pos += 0.1 * np.sin((frame_cnt)/400)
    frame_cnt += 1
    
    if frame_cnt == 1:
        print(joint_name)
    
    torque = np.zeros((20,3))
    # 你的代码
    lower_body_joints = [
        'RootJoint', 
        'rHip', 'lHip', 
        'rKnee', 'lKnee', 
        'rAnkle', 'lAnkle', 
        'rToeJoint', 'lToeJoint'
    ]
    
    global_torque = part1_cal_torque(pose, physics_info, kp=800, kd=31) 
    joint_velocity = physics_info.get_body_velocity()
    joint_mass = physics_info.get_body_mass()
    # compute center of mass and COM velocity
    CoM = np.zeros(3)
    CoM_vel = np.zeros(3)
    mass = 0
    # 加权平均
    for i in range(len(joint_mass)):
        CoM += joint_mass[i] * joint_positions[i]
        CoM_vel += joint_mass[i] * joint_velocity[i]
        mass += joint_mass[i]
    CoM /= mass
    CoM_vel /= mass
    desired_CoM = tar_pos

    Kp = 200
    Kd = 20
    virtual_force = Kp * (desired_CoM - CoM) - Kd * CoM_vel
    
    torque = global_torque
    # for i in range(0, len(torque)):
    #     torque[i] += np.cross( CoM - joint_positions[i], virtual_force)
        
    for i, name in enumerate(joint_name):
        if name in lower_body_joints:
            torque[i] -= np.cross( CoM - joint_positions[i], virtual_force)
    
    max_torque = 500
    torque = np.clip(torque, -max_torque, max_torque)
    
    
    # 抹去根节点力矩
    torque[0] = np.zeros_like(torque[0])
    return torque

























frame_cnt = 0
def part3_cal_static_standing_torque(bvh: BVHMotion, physics_info):
    '''
    输入： bvh: BVHMotion类，包含了当前的动作信息，参见bvh_loader.py
    输出： 带反馈的global_torque: (20, 3)的numpy数组，因为不需要控制方向，根节点力矩会被无视
    Tips: 
        只track第0帧就能保持站立了
        为了保持平衡可以把目标的根节点位置适当前移，比如把根节点位置和左右脚的中点加权平均，但要注意角色还会收到一个从背后推他的外力
        可以定义一个全局的frame_count变量来标记当前的帧数，在站稳后根据帧数使角色进行周期性左右摇晃，如果效果好可以加分（0-20），可以考虑让角色站稳后再摇晃
    '''
    # ------一些提示代码，你可以随意修改------------#
    tar_pos = bvh.joint_position[0][0]
    pose = bvh.joint_rotation[0]
    joint_name = physics_info.joint_name
    
    joint_positions = physics_info.get_joint_translation()
    joint_orientations = physics_info.get_body_orientation()
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    # 适当前移，这里的权重需要你自己调整
    # tar_pos = tar_pos * 0.8 + joint_positions[9] * 0.1 + joint_positions[10] * 0.1
    # tar_pos = tar_pos * 0.1 + joint_positions[9] * 0.45 + joint_positions[10] * 0.45
    # tar_pos = joint_positions[9] * 0.5 + joint_positions[10] * 0.5
    
    rend_site = joint_positions[9] + R.from_quat(joint_orientations[9]).apply([0.010, 0.002, 0.060])
    lend_site = joint_positions[10] + R.from_quat(joint_orientations[10]).apply([-0.010, 0.002, 0.060])
    
    # rend_site = bvh.joint_position[0][9] + np.array([0.010, 0.002, 0.060])
    # lend_site = bvh.joint_position[0][10] + np.array([-0.010, 0.002, 0.060])
    
    tar_pos = joint_positions[9] * 0.1 + joint_positions[10] * 0.1 + rend_site * 0.4 + lend_site * 0.4
    # tar_pos = joint_positions[9] * 0.25 + joint_positions[10] * 0.25 + rend_site * 0.25 + lend_site * 0.25
    
    # tar_pos[1] = bvh.joint_position[0][0][1] + 0.05
    tar_pos[1] = bvh.joint_position[0][0][1]
    
    global frame_cnt
    # 修改根节点目标位置让角色摇摆起来，可以角色先站稳之后再周期性摇摆
    # tar_pos += 0.05 * np.sin((frame_cnt)/400)
    frame_cnt += 1
    
    if frame_cnt == 1:
        print(joint_name)
    
    torque = np.zeros((20,3))
    # 你的代码
    lower_body_joints = [
        'RootJoint', 
        'rHip', 'lHip', 
        'rKnee', 'lKnee', 
        'rAnkle', 'lAnkle', 
        'rToeJoint', 'lToeJoint'
    ]
    hip_joints = ['rHip', 'lHip']
    
    # global_torque = part1_cal_torque(pose, physics_info, kp=400, kd=20) 
    global_torque = part1_cal_torque(pose, physics_info, kp=300, kd=10) 
    joint_velocity = physics_info.get_body_velocity()
    joint_mass = physics_info.get_body_mass()
    # compute center of mass and COM velocity
    CoM = np.zeros(3)
    CoM_vel = np.zeros(3)
    mass = 0
    # 加权平均
    for i in range(len(joint_mass)):
        CoM += joint_mass[i] * joint_positions[i]
        CoM_vel += joint_mass[i] * joint_velocity[i]
        mass += joint_mass[i]
    CoM /= mass
    CoM_vel /= mass
    desired_CoM = tar_pos

    # Kp = 4000
    # Kd = 50
    Kp = 4000
    Kd = 40
    virtual_force = Kp * (desired_CoM - root_position) - Kd * root_velocity
    
    torque = global_torque
    # for i in range(0, len(torque)):
    #     torque[i] += np.cross( CoM - joint_positions[i], virtual_force)
        
    for i, name in enumerate(joint_name):
        if name in lower_body_joints:
            torque[i] -= np.cross( root_position - joint_positions[i], virtual_force)
    
    max_torque = 1000
    torque = np.clip(torque, -max_torque, max_torque)

    
    # 抹去根节点力矩
    torque[0] = np.zeros_like(torque[0])
    return torque