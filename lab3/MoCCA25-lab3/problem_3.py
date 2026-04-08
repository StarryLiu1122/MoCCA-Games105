import numpy as np
import pyvista as pv

import utils
from joints import Joint
from triple_pendulum import create_triple_pendulum

from answer_1 import integration
from answer_2 import forward_dynamics_with_constraints
    
import argparse

np.set_printoptions(formatter={'float': lambda x: f'{x:9.6f}'})

# 在这个任务中，我们在problem_2的基础上，为每个关节增加了一个 hinge 约束
def problem_3(args):
    if args.debug_mode:
        pendulum = create_triple_pendulum(args, zero_gravity=True, no_root=True)
    else:
        pendulum = create_triple_pendulum(args)
    (m, I, inv_m, inv_I, x, R, v, w, f, tau, h, g, joints, box_meshes, box_vertices, plotter) = pendulum
    
    if args.debug_mode:
        w[2,:] = np.array([5.0, 5.0, 5.0])
    else:
        w[2,:] = np.array([0.0, 0.0, 10.0]) # 将第三个刚体的初始角速度为10
    
    # 为每个关节设置一个 hinge 约束
    if args.debug_mode:
        joints[0].hinge_axis = np.array([0, 1, 0])
        joints[1].hinge_axis = np.array([0, 0, 1])
    else:
        joints[0].hinge_axis = np.array([0, 0, 1])
        joints[1].hinge_axis = np.array([0, 1, 0])
        joints[2].hinge_axis = np.array([0, 0, 1])
    
    ###########################
    # 开始仿真
    # 可以通过在控制台 ctrl+C 中止
    if args.debug_mode:
        h = 0.0001 # 我们使用更小的仿真步长来做debug
    fps = 60
    sub_steps = max(1, int(1/fps/h))
    
    for t in range(args.frames): 
        for i in range(sub_steps): 
            v_next, w_next = forward_dynamics_with_constraints(m, I, inv_m, inv_I, x, R, v, w, f, tau, h, joints)
            x_next, R_next = integration(x, R, v_next, w_next, h)
            
            x[:-1], R[:-1], v[:-1], w[:-1] = x_next[:-1], R_next[:-1], v_next[:-1], w_next[:-1]
        
        try:
            with pv.VtkErrorCatcher(True) as error_catcher:
                utils._update_boxes(x[:-1], R[:-1], box_meshes, box_vertices, plotter)
                
            if args.video is not None:
                plotter.write_frame()
        except:
            exit(0)
        
        if args.debug_mode:
            print('angular momentum:', utils.angular_momentum(m, I, x, R, v, w))
        
    plotter.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem 3')
    parser.add_argument('--video', type=str, default=None, help='video name')
    parser.add_argument('--frames', type=int, default=600, help='number of frames')
    parser.add_argument('--debug-mode', action='store_true', help='debug mode')
    
    args = parser.parse_args()
    problem_3(args)