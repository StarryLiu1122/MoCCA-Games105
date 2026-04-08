##############
# 姓名：刘星云
# 学号：2300012297
##############
import numpy as np

# part 0
def load_meta_data(bvh_file_path):
    """
    请把lab1-FK-part1的代码复制过来
    请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        channels: List[int]，整数列表，joint的自由度，根节点为6(三个平动三个转动)，其余节点为3(三个转动)
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量
    Tips:
        joint_name顺序应该和bvh一致 
    """
    

    joints = None
    joint_parents = None
    channels = None
    joint_offsets = None

    ### Your code here
    joints = []
    joint_parents = []
    channels = []
    joint_offsets = []
    
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
                joint_parents.append(stack[-1])  
            else:
                joint_parents.append(-1)
                
            index += 1
            stack.append(index)  
            
        elif line.startswith("}"):
            stack.pop()
        
        elif line.startswith("ROOT") :
            name = 'RootJoint'
            joints.append(name)
            
        elif line.startswith("JOINT"):
            parts = line.split()
            name = parts[1]
            joints.append(name)
        
        elif line.startswith("CHANNELS"):
            parts = line.split()
            channel = parts[1]
            channels.append(int(channel))
            
        elif line.startswith("OFFSET"):
            parts = line.split()
            offset = list(map(float, parts[1:4]))
            joint_offsets.append(offset)
        
        elif line.startswith("End Site"):
            joints.append(f"{joints[-1]}_end")
            channels.append(int(0))
    
    joint_offsets = np.array(joint_offsets)
    ### Your code here
    
    return joints, joint_parents, channels, joint_offsets