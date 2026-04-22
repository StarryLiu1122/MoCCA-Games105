
import utils

import numpy as np
from scipy.spatial.transform import Rotation

from typing import Tuple

def integration(x:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray, h:float) -> Tuple[np.ndarray, np.ndarray]:
    ''' 刚体状态更新函数
    
        给出一组刚体的位置、朝向，以及当前速度、角速度，利用欧拉积分计算下一时刻刚体的位置和朝向
    
        输入：
            x: (num_bodies, 3)        刚体（质心）的位置
            R: (num_bodies, 3, 3)     刚体的朝向，表示为一组矩阵
            v: (num_bodies, 3)        刚体的（质心）线速度
            w: (num_bodies, 3)        刚体的角速度
            h:                      时间步长
        
        所有量均为世界坐标系表示
        
        输出：
            x_next: (num_bodies, 3)        刚体下一时刻的位置
            R_next: (num_bodies, 3, 3)     刚体下一时刻的朝向
            
        提示：
            * 旋转矩阵需要正交化，你可以选择合适的正交化方法
            * 也可以考虑先将旋转转化为四元数，做积分后再转回旋转矩阵
            * 可以使用scipy中的相关函数做相应的转换
            * 输入包括了场景中所有的刚体，你可以使用 for 循环来处理每个刚体（不推荐），或者使用 numpy 的向量化操作来处理所有刚体
            * 你可能不需要用到所有的输入信息
    '''
    
    num_bodies = x.shape[0]
    x_next, R_next = x.copy(), R.copy()
    
    ####### 你的回答 #######
    
    
    ######################
    
    return x_next, R_next

def forward_dynamics(
    m:np.ndarray, I:np.ndarray, inv_m:np.ndarray, inv_I:np.ndarray,
    x:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray, 
    f:np.ndarray, tau:np.ndarray, h:float
    ) -> Tuple[np.ndarray, np.ndarray]:
    ''' 独立刚体的前向运动学函数
    
        给出一组刚体的位置、朝向，以及当前速度、角速度，利用欧拉积分计算下一时刻刚体的位置和朝向
    
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
            h:                        时间步长
        
        所有量均为世界坐标系表示
        
        输出：
            v_next: (num_bodies, 3)        刚体下一时刻的速度
            w_next: (num_bodies, 3, 3)     刚体下一时刻的角速度
            
        提示：
            * 使用 Newton-Euler 方程来求解下一时刻刚体的状态，参考ppt中相关内容
            * 输入包括了场景中所有的刚体，你可以使用 for 循环来处理每个刚体（不推荐），或者使用 numpy 的向量化操作来处理所有刚体
            * 你可能不需要用到所有的输入信息
    '''
    num_bodies = x.shape[0]    
    v_next, w_next = v.copy(), w.copy()
    
    ####### 你的回答 #######
    
    #######################
    
    return v_next, w_next
    