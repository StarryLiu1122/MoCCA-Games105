import numpy as np

from dataclasses import dataclass

from typing import List

@dataclass
class Joint:
    r''' 关节定义
    
        一个关节连接两个刚体，关节和刚体的关系如下（*代表关节）：
        
        -------------------    --------------------
        |                  \   /                  |
        |     bodyA -------> * <------ bodyB      |    
        |            ^^^^^      ^^^^^             |
        |       from_bodyA /   \ from_bodyB       |
        --------------------   --------------------
    '''
    bodyA: int  # 刚体A的index
    bodyB: int  # 刚体B的index
    from_bodyA: np.ndarray  # (3,) 初始状态下（也即局部坐标系下）关节相对于刚体A质心的位置
    from_bodyB: np.ndarray  # (3,) 初始状态下（也即局部坐标系下）关节相对于刚体B质心的位置
    
    hinge_axis: np.ndarray|None=None  # (3,) 或 None，如果为None，本关节为 ball joint
                                      # 如果不为None，则给出hinge转轴在bodyA坐标系下的表示
                                      # 默认为None
    
    
    
def joint_torque_converter(num_bodies:int, joints:List[Joint]):
    ''' 创建一个函数，可以将一组关节力矩转换为施加在刚体上的力矩
    '''
        
    bodyA = np.array([jnt.bodyA for jnt in joints])
    bodyB = np.array([jnt.bodyB for jnt in joints])
    
    def apply_joint_torques(torques) -> np.ndarray:
        ''' 将关节力矩转换为施加在刚体上的力矩
        
            输入：
                torques: (num_joints, 3) 关节力矩
                
            输出：
                body_torques: (num_bodies, 3) 刚体力矩
        '''
        
        body_torques = np.zeros((num_bodies, 3))
        body_torques[bodyB] += torques
        body_torques[bodyA] -= torques
            
        return body_torques
    
    return apply_joint_torques