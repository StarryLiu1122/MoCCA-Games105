##############
# 姓名：刘星云
# 学号：2300012297
##############
import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, input_joint_positions, input_joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """    
    joint_positions = input_joint_positions.copy()
    joint_orientations = input_joint_orientations.copy()
    
    joint_parent = meta_data.joint_parent
    joint_offset = [meta_data.joint_initial_position[i] - meta_data.joint_initial_position[joint_parent[i]] for i in
                    range(len(joint_positions))]
    joint_offset[0] = np.array([0., 0., 0.])
    # 迭代计算IK链条上各个关节的旋转
    local_rotation = [R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i]) for i
                      in range(len(joint_orientations))]
    local_rotation[0] = R.from_quat(joint_orientations[0])

    joint_ik_path, _, _, _ = meta_data.get_path_from_root_to_end()
    ik_path_set = set(joint_ik_path)
    num_joints = len(joint_ik_path)
    
    # 最大迭代次数
    max_iter = 100
    threshold = 1e-2
    
    for _ in range(max_iter):
        # 目标关节与末端关节的当前距离
        end_joint_pos = joint_positions[joint_ik_path[-1]]
        distance = np.linalg.norm(target_pose - end_joint_pos)
        
        # 终止条件
        if distance < threshold:
            break
        
        # 从末端向根部迭代
        for i in range(num_joints - 2, -1, -1):
            current_joint = joint_ik_path[i]   # 当前关节索引
            
            # 计算当前指向末端的向量
            current_to_end = joint_positions[joint_ik_path[-1]] - joint_positions[current_joint]
            current_to_target = target_pose - joint_positions[current_joint]
            if np.linalg.norm(current_to_end) < 1e-6 or np.linalg.norm(current_to_target) < 1e-6:
                continue
            
            # 归一化
            current_to_end = current_to_end / np.linalg.norm(current_to_end)
            current_to_target = current_to_target / np.linalg.norm(current_to_target)
            
            # 叉积求旋转轴
            axis = np.cross(current_to_end, current_to_target)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                continue
            axis = axis / axis_norm
            
            # 点积求旋转角度
            angle = np.arccos(np.clip(np.dot(current_to_end, current_to_target), -1.0, 1.0))
            
            # 生成旋转四元数
            rotation = R.from_rotvec(axis * angle)
            
            # 更新关节的旋转和位置
            # current关节 更新旋转
            current_orientation = R.from_quat(joint_orientations[current_joint])
            new_orientation = rotation * current_orientation
            joint_orientations[current_joint] = new_orientation.as_quat()
            
            # 路径上其它关节 更新旋转和位置
            for j in range(i + 1,num_joints):
                index = joint_ik_path[j]
                parent_index = joint_parent[index] 
                if parent_index == joint_ik_path[j - 1]:
                    joint_positions[index] = joint_positions[parent_index] + R.from_quat(joint_orientations[parent_index]).apply(joint_offset[index])
                    joint_orientations[index] = (rotation * R.from_quat(joint_orientations[index])).as_quat()
                else:
                    if parent_index == -1:
                        joint_orientations[index] = (rotation * R.from_quat(joint_orientations[index])).as_quat()
                        joint_positions[index] = joint_positions[joint_ik_path[j-1]] - R.from_quat(joint_orientations[index]).apply(joint_offset[joint_ik_path[j-1]])
                    else:
                        joint_orientations[index] = (rotation * R.from_quat(joint_orientations[index])).as_quat()
                        joint_positions[index] = joint_positions[joint_ik_path[j-1]] - R.from_quat(joint_orientations[index]).apply(joint_offset[joint_ik_path[j-1]])
                        
                    
            # 非路径关节
            for i in range(len(joint_positions)):
                if i not in ik_path_set:
                    joint_orientations[i] = (R.from_quat(joint_orientations[joint_parent[i]]) * local_rotation[i]).as_quat()
                    joint_positions[i] = joint_positions[joint_parent[i]] + R.from_quat(joint_orientations[joint_parent[i]]).apply(joint_offset[i])

    return joint_positions, joint_orientations



def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入左手相对于根节点前进方向的xz偏移，以及目标高度，lShoulder到lWrist为可控部分，其余部分与bvh一致
    注意part1中只要求了目标关节到指定位置，在part2中我们还对目标关节的旋转有所要求
    """
    # joint_positions = None
    # joint_orientations = None
    
    joint_parent = meta_data.joint_parent
    joint_offset = [meta_data.joint_initial_position[i] - meta_data.joint_initial_position[joint_parent[i]] for i in
                    range(len(joint_positions))]
    # 迭代计算IK链条上各个关节的旋转
    local_rotation = [R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i]) for i
                      in range(len(joint_orientations))]
    
    joint_ik_path, _, _, _ = meta_data.get_path_from_root_to_end()
    ik_path_set = set(joint_ik_path)
    num_joints = len(joint_ik_path)
    
    root_index = joint_ik_path[0]
    end_index = joint_ik_path[-1]
    target_pose= joint_positions[root_index] + np.array([relative_x, 0, relative_z])
    target_pose[1] = target_height - np.linalg.norm(joint_offset[end_index])  


    # 最大迭代次数
    max_iter = 100
    threshold = 1e-2
    
    for _ in range(max_iter):
        # 目标关节与末端关节的当前距离
        end_joint_pos = joint_positions[joint_ik_path[-2]]
        distance = np.linalg.norm(target_pose - end_joint_pos)
        
        # 终止条件
        if distance < threshold:
            break
        
        # 从末端向根部迭代
        for i in range(num_joints - 2, -1, -1):
            current_joint = joint_ik_path[i]   # 当前关节索引
            
            # 计算当前指向末端的向量
            current_to_end = joint_positions[joint_ik_path[-2]] - joint_positions[current_joint]
            current_to_target = target_pose - joint_positions[current_joint]
            if np.linalg.norm(current_to_end) < 1e-6 or np.linalg.norm(current_to_target) < 1e-6:
                continue
            
            # 归一化两个向量
            current_to_end = current_to_end / np.linalg.norm(current_to_end)
            current_to_target = current_to_target / np.linalg.norm(current_to_target)
            
            # 叉积求旋转轴
            axis = np.cross(current_to_end, current_to_target)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                continue
            axis = axis / axis_norm
            
            # 点积求旋转角度
            angle = np.arccos(np.clip(np.dot(current_to_end, current_to_target), -1.0, 1.0))
            
            # 生成旋转四元数
            rotation = R.from_rotvec(axis * angle)
            
            # 更新关节的旋转和位置
            # current关节 更新旋转
            current_orientation = R.from_quat(joint_orientations[current_joint])
            new_orientation = rotation * current_orientation
            joint_orientations[current_joint] = new_orientation.as_quat()
            
            # 路径上其它关节 更新旋转和位置
            for j in range(i + 1,num_joints - 1):
                index = joint_ik_path[j]
                parent_index = joint_parent[index] 
                joint_positions[index] = joint_positions[parent_index] + R.from_quat(joint_orientations[parent_index]).apply(joint_offset[index])
                joint_orientations[index] = (rotation * R.from_quat(joint_orientations[index])).as_quat()
                
    
    joint_orientations[end_index] = R.from_euler('Z', 90, degrees=True).as_quat()
    joint_orientations[joint_ik_path[-2]] = R.from_euler('Z', 90, degrees=True).as_quat()
    joint_positions[end_index] = joint_positions[joint_ik_path[-2]]
    joint_positions[end_index][1] += np.linalg.norm(joint_offset[end_index])  
    
    return joint_positions, joint_orientations