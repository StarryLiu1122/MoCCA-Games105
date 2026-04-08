import numpy as np
from scipy.spatial.transform import Rotation as R

# def part1_calculate_T_pose(bvh_file_path):
#     joint_name = []
#     joint_parent = []
#     joint_offset = []
#     stack = []
    
#     with open(bvh_file_path, 'r') as file:
#         lines = file.readlines()
    
#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()
#         if line.startswith("ROOT") or line.startswith("JOINT"):
#             joint_name.append(line.split()[-1])
#             joint_offset.append([0.0, 0.0, 0.0])
#             if stack:
#                 joint_parent.append(stack[-1])
#             else:
#                 joint_parent.append(-1)
#             stack.append(len(joint_name) - 1)
#         elif line.startswith("End Site"):
#             joint_name.append(f"EndSite_{joint_name[-1]}")
#             joint_offset.append([0.0, 0.0, 0.0])
#             joint_parent.append(stack[-1])
#             stack.append(len(joint_name) - 1)
#         elif line.startswith("OFFSET"):
#             offset = list(map(float, line.split()[1:4]))
#             joint_offset[stack[-1]] = offset
#         elif line.startswith("}"):
#             stack.pop()
#         i += 1
    
#     joint_offset = np.array(joint_offset)
    
#     return joint_name, joint_parent, joint_offset



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
            joint_parent.append(index)  
            stack.append(index)  
            index += 1
            
        
        elif line.startswith("}"):
            stack.pop()
            index -= 1
        
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

joint_name, joint_parent, joint_offset = part1_calculate_T_pose(r'C:\Users\Lenovo\Desktop\MoCCA\MoCCA25-Lab1\data\dance.bvh')

print(len(joint_name))
print(joint_parent)
print(joint_offset)




def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    
    Tips:
        两个bvh的joint name顺序可能不一致哦
        as_euler时也需要大写的XYZ
    """
    # 解析 T-pose 和 A-pose 的 BVH 文件
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    
    # 加载 A-pose 的运动数据
    A_motion_data = load_motion_data(A_pose_bvh_path)
    num_frames = A_motion_data.shape[0]
    num_joints = len(A_joint_name)   # 25
    
    # 创建一个映射，将 A-pose 的关节名称映射到 T-pose 的关节索引
    joint_name_to_index = {name: i for i, name in enumerate(T_joint_name)}
    
    # 计算每个关节的初始旋转差异（A-pose 到 T-pose 的旋转）
    initial_rotations = []
    now = 0
    for i in range(num_joints):
        if A_joint_name[i] in joint_name_to_index and not A_joint_name[i].endswith('end'):
            # 获取 T-pose 和 A-pose 的初始旋转
            T_index = joint_name_to_index[A_joint_name[i]]
            T_rotation = R.from_euler('XYZ', [0, 0, 0], degrees=True)  # T-pose 的初始旋转   改
            A_rotation = R.from_euler('XYZ', A_motion_data[0, 3 + 3 * now:3 + 3 * (now + 1)], degrees=True)  # A-pose 的初始旋转
            now += 1
            # 计算旋转差异
            rotation_diff = T_rotation * A_rotation.inv()
            initial_rotations.append(rotation_diff)
        else:
            # 如果关节在 T-pose 中不存在，则使用单位四元数
            initial_rotations.append(R.from_euler('XYZ', [0, 0, 0], degrees=True))
            print("error")
    
    # 重定向运动数据
    motion_data = np.zeros_like(A_motion_data)
    
    for frame_id in range(num_frames):
        # 复制根关节的平移数据
        motion_data[frame_id, 0:3] = A_motion_data[frame_id, 0:3]
        now = 0
        # 重定向每个关节的旋转数据
        for i in range(num_joints):
            if A_joint_name[i] in joint_name_to_index and not A_joint_name[i].endswith('end'):
                # 获取 A-pose 的旋转数据
                A_euler = A_motion_data[frame_id, 3 + 3 * now:3 + 3 * (now + 1)]
                
                A_rotation = R.from_euler('XYZ', A_euler, degrees=True)
                
                # 应用初始旋转差异
                T_rotation = initial_rotations[i] * A_rotation
                
                # 将旋转转换回欧拉角
                T_euler = T_rotation.as_euler('XYZ', degrees=True)
                motion_data[frame_id, 3 + 3 * now:3 + 3 * (now + 1)] = T_euler
                now += 1
                
    return motion_data



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
    # 用于迭代计算IK链条上各个关节的旋转
    local_rotation = [R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i]) for i
                      in range(len(joint_orientations))]
    local_rotation[0] = R.from_quat(joint_orientations[0])

    joint_ik_path, _, _, _ = meta_data.get_path_from_root_to_end()
    ik_path_set = set(joint_ik_path)
    num_joints = len(joint_ik_path)
    
    # 最大迭代次数
    max_iter = 500
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
            
            # 1. 计算当前指向末端的向量
            current_to_end = joint_positions[joint_ik_path[-1]] - joint_positions[current_joint]
            current_to_target = target_pose - joint_positions[current_joint]
            if np.linalg.norm(current_to_end) < 1e-6 or np.linalg.norm(current_to_target) < 1e-6:
                continue
            
            # 2. 归一化两个向量
            current_to_end = current_to_end / np.linalg.norm(current_to_end)
            current_to_target = current_to_target / np.linalg.norm(current_to_target)
            
            # 3. 通过叉积求旋转轴
            axis = np.cross(current_to_end, current_to_target)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                continue
            axis = axis / axis_norm
            
            # 4. 通过点积求旋转角度
            angle = np.arccos(np.clip(np.dot(current_to_end, current_to_target), -1.0, 1.0))
            
            # 5. 生成旋转四元数
            rotation = R.from_rotvec(axis * angle)
            
            # 6. 更新关节的旋转和位置
            # current关节 更新旋转
            current_orientation = R.from_quat(joint_orientations[current_joint])
            new_orientation = rotation * current_orientation
            joint_orientations[current_joint] = new_orientation.as_quat()
            
            # 路径上其它关节 更新旋转和位置
            for j in range(i + 1,num_joints):
                index = joint_ik_path[j]
                parent_index = joint_parent[index] 
                joint_positions[index] = joint_positions[parent_index] + R.from_quat(joint_orientations[parent_index]).apply(joint_offset[index])
                joint_orientations[index] = (rotation * R.from_quat(joint_orientations[index])).as_quat()

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
    joint_offset[0] = np.array([0., 0., 0.])
    # 用于迭代计算IK链条上各个关节的旋转
    local_rotation = [R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i]) for i
                      in range(len(joint_orientations))]
    local_rotation[0] = R.from_quat(joint_orientations[0])
    
    joint_ik_path, _, _, _ = meta_data.get_path_from_root_to_end()
    ik_path_set = set(joint_ik_path)
    num_joints = len(joint_ik_path)
    
    root_index = joint_ik_path[0]
    end_index = joint_ik_path[-1]
    target_pose= joint_positions[root_index] + np.array([relative_x, 0, relative_z])
    target_pose[1] = target_height - np.linalg.norm(joint_offset[end_index])  


    # 最大迭代次数
    max_iter = 500
    threshold = 1e-2
    
    for _ in range(max_iter):
        # 目标关节与末端关节的当前距离
        end_joint_pos = joint_positions[joint_ik_path[-2]]
        distance = np.linalg.norm(target_pose - end_joint_pos)
        
        # 终止条件
        if distance < threshold:
            break
        
        # 从末端向根部迭代
        for i in range(num_joints - 3, -1, -1):
            current_joint = joint_ik_path[i]   # 当前关节索引
            
            # 1. 计算当前指向末端的向量
            current_to_end = joint_positions[joint_ik_path[-2]] - joint_positions[current_joint]
            current_to_target = target_pose - joint_positions[current_joint]
            if np.linalg.norm(current_to_end) < 1e-6 or np.linalg.norm(current_to_target) < 1e-6:
                continue
            
            # 2. 归一化两个向量
            current_to_end = current_to_end / np.linalg.norm(current_to_end)
            current_to_target = current_to_target / np.linalg.norm(current_to_target)
            
            # 3. 通过叉积求旋转轴
            axis = np.cross(current_to_end, current_to_target)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                continue
            axis = axis / axis_norm
            
            # 4. 通过点积求旋转角度
            angle = np.arccos(np.clip(np.dot(current_to_end, current_to_target), -1.0, 1.0))
            
            # 5. 生成旋转四元数
            rotation = R.from_rotvec(axis * angle)
            
            # 6. 更新关节的旋转和位置
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
    
    joint_orientations[end_index] = R.from_quat([0., 0., 0., 1.]).as_quat()
    joint_positions[end_index] = joint_positions[joint_ik_path[-2]] + np.linalg.norm(joint_offset[end_index])  
    return joint_positions, joint_orientations



elif joint == 'lShoulder':
                Rot = (R.from_euler('XYZ', list(A_motion_data[i][index * 3 + 3: index*3 + 6]), degrees=True) * R.from_euler('XYZ', [0., 0., -45.], degrees=True)).as_euler('XYZ',True)
                data += list(Rot)
            elif joint == 'rShoulder':
                Rot = (R.from_euler('XYZ', list(A_motion_data[i][index * 3 + 3: index*3 + 6]), degrees=True) * R.from_euler('XYZ', [0., 0., 45.], degrees=True)).as_euler('XYZ',True)
                data += list(Rot)
            elif joint == 'lHip':
                Rot = (R.from_euler('XYZ', list(A_motion_data[i][index * 3 + 3: index*3 + 6]), degrees=True) * R.from_euler('XYZ', [0., 0., 45.], degrees=True)).as_euler('XYZ',True)
                data += list(Rot)
            elif joint == 'rHip':
                Rot = (R.from_euler('XYZ', list(A_motion_data[i][index * 3 + 3: index*3 + 6]), degrees=True) * R.from_euler('XYZ', [0., 0., -45.], degrees=True)).as_euler('XYZ',True)
                data += list(Rot)
                
                
                
                
                
                
                
                
                
                
for frame in range(n3):
        walk_frame = n1/n3 * frame
        run_frame = n2/n3 * frame
        walk_frame_floor = int(np.floor(walk_frame))
        walk_frame_ceil = int(np.ceil(walk_frame))
        walk_lerp_alpha = walk_frame_ceil - walk_frame
        run_frame_floor = int(np.floor(run_frame))
        run_frame_ceil = int(np.ceil(run_frame))
        run_lerp_alpha = run_frame_ceil - run_frame
        
        if walk_frame_ceil >= n1 or run_frame_ceil >= n2:
            walk_pos = bvh_motion1.joint_position[walk_frame_floor]
            run_pos = bvh_motion2.joint_position[run_frame_floor]
            res.joint_position[frame] = alpha[frame] * walk_pos + (1 - alpha[frame]) * run_pos
            
            rot_walk = bvh_motion1.joint_rotation[walk_frame_floor]
            rot_run = bvh_motion2.joint_rotation[run_frame_floor]
            res.joint_rotation[frame]  = alpha[frame] * rot_walk + (1 - alpha[frame]) * rot_run
            continue
        
        walk_pos = walk_lerp_alpha * bvh_motion1.joint_position[walk_frame_floor] + (1 - walk_lerp_alpha) * bvh_motion1.joint_position[walk_frame_ceil]
        run_pos = run_lerp_alpha * bvh_motion2.joint_position[run_frame_floor] + (1 - run_lerp_alpha) * bvh_motion2.joint_position[run_frame_ceil]
        res.joint_position[frame] = alpha[frame] * walk_pos + (1 - alpha[frame]) * run_pos
        

        # rot_walk = Slerp(bvh_motion1.joint_rotation[walk_frame_floor], bvh_motion1.joint_rotation[walk_frame_ceil], walk_lerp_alpha)
        # rot_run  = Slerp(bvh_motion2.joint_rotation[run_frame_floor], bvh_motion2.joint_rotation[run_frame_ceil], run_lerp_alpha)
        # res.joint_rotation[frame] = Slerp(rot_walk, rot_run, alpha)
        
        # rot_walk = walk_lerp_alpha * bvh_motion1.joint_rotation[walk_frame_floor] + (1 - walk_lerp_alpha) * bvh_motion1.joint_rotation[walk_frame_ceil]
        # rot_run = run_lerp_alpha * bvh_motion2.joint_rotation[run_frame_floor] + (1 - run_lerp_alpha) * bvh_motion2.joint_rotation[run_frame_ceil]
        # res.joint_rotation[frame]  = alpha * rot_walk + (1 - alpha) * rot_run
        # Walk motion 插值
        
        for joint in range(len(bvh_motion1.joint_name)):
            
            key_times = [0, 1]
            key_rots_walk = R.from_quat([
                bvh_motion1.joint_rotation[walk_frame_floor][joint], 
                bvh_motion1.joint_rotation[walk_frame_ceil][joint]
            ])
            slerp_walk = Slerp(key_times, key_rots_walk)
            rot_walk = slerp_walk([walk_lerp_alpha])[0].as_quat()

            # Run motion 插值
            key_rots_run = R.from_quat([
                bvh_motion2.joint_rotation[run_frame_floor][joint], 
                bvh_motion2.joint_rotation[run_frame_ceil][joint]
            ])
            slerp_run = Slerp(key_times, key_rots_run)
            rot_run = slerp_run([run_lerp_alpha])[0].as_quat()
            
            # 再对两个动作之间插值
            key_rots_blend = R.from_quat([rot_walk, rot_run])
            slerp_blend = Slerp([0, 1], key_rots_blend)
            res.joint_rotation[frame][joint] = slerp_blend([alpha[frame]])[0].as_quat()