##############
# 姓名：刘星云
# 学号：2300012297
##############
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
    
    stack = []  
    index = -1
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("MOTION"):
            break
        
        if line.startswith("{"):
            if stack:
                joint_parent.append(stack[-1])  
            else:
                joint_parent.append(-1)
                
            index += 1
            stack.append(index)  
            
        elif line.startswith("}"):
            stack.pop()
        
        elif line.startswith("ROOT") :
            name = 'RootJoint'
            joint_name.append(name)
            
        elif line.startswith("JOINT"):
            parts = line.split()
            name = parts[1]
            joint_name.append(name)
        
        elif line.startswith("OFFSET"):
            parts = line.split()
            offset = list(map(float, parts[1:4]))
            joint_offset.append(offset)
        
        elif line.startswith("End Site"):
            joint_name.append(f"{joint_name[-1]}_end")
    
    joint_offset = np.array(joint_offset)

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []
    
    num = len(joint_name)
    data = motion_data[frame_id]
    
    # 根关节
    root_position = data[0:3]
    root_orientation = R.from_euler('XYZ', data[3:6], degrees=True).as_quat() 
    joint_positions.append(np.array(root_position).reshape(1, 3))
    joint_orientations.append(root_orientation.reshape(1, 4))
    
    # 非根关节
    now = 1
    for i in range(1,num):
        parent = joint_parent[i]                         # 父关节序号
        parent_position = joint_positions[parent]        # 父关节的全局位置
        parent_orientation = joint_orientations[parent]  # 父关节的全局旋转
        position = parent_position + R.from_quat(parent_orientation).apply(joint_offset[i])
        
        if not joint_name[i].endswith('end'):
            orientation = (R.from_quat(parent_orientation) * R.from_euler('XYZ', data[3 * now + 3:3 * now + 6], degrees=True)).as_quat()
            now += 1
            
    
        joint_positions.append(np.array(position).reshape(1, 3))  
        joint_orientations.append(orientation.reshape(1, 4)) 

    # 拼接
    joint_positions = np.concatenate(joint_positions, axis=0)       
    joint_orientations = np.concatenate(joint_orientations, axis=0)  
    
    return joint_positions, joint_orientations




def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    
    A_motion_data = load_motion_data(A_pose_bvh_path)
    
    # 计算 A-pose 到 T-pose 的旋转变换矩阵
    retarget_matrices = {}
    
    for i, joint in enumerate(T_joint_name):
        if joint in A_joint_name:
            
            A_offset = A_joint_offset[i]      # A-pose 偏移量
            T_offset = T_joint_offset[i]      # T-pose 偏移量
                
            if np.array_equal(T_offset, A_offset):
                retarget_matrices[joint] = R.identity()
            else:
                retarget_matrices[joint] = R.align_vectors([T_offset], [A_offset])[0]


    # 重新计算运动数据
    motion_data = []
    for i in range(A_motion_data.shape[0]):
        data = []
        now = -1
        for joint in T_joint_name:
            if 'end' not in joint:
                now += 1
            else:
                continue   # 叶节点跳过

            if joint == "RootJoint":
                data += list(A_motion_data[i][0:6]) 
            else:
                index = T_joint_name.index(joint)
                    
                if retarget_matrices[joint] == R.identity() and joint != 'lShoulder' and joint != 'rShoulder' and joint != 'lHip' and joint != 'rHip':
                    T_rotation = R.from_euler('XYZ', A_motion_data[i][now * 3 + 3: now * 3 + 6], degrees=True)
                    data += list(T_rotation.as_euler('XYZ', degrees=True))
                    continue
                else:
                    son_index = index + 1
                    
                    son_name = T_joint_name[son_index]
                    A_rotation = R.from_euler('XYZ', A_motion_data[i][now * 3 + 3: now * 3 + 6], degrees=True)
                
                    Qp_rotation = retarget_matrices[joint]   
                    Q_rotation = retarget_matrices[son_name]
                    
                    T_rotation = Qp_rotation * A_rotation * (Q_rotation.inv())
                    data += list(T_rotation.as_euler('XYZ', degrees=True))
                        

        motion_data.append(np.array(data).reshape(1, -1))

    motion_data = np.concatenate(motion_data, axis=0)
    return motion_data