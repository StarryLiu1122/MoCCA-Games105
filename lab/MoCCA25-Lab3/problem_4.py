import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv

import utils
from joints import Joint, joint_torque_converter
from triple_pendulum import create_triple_pendulum, forward_kinematics

from answer_1 import integration
from answer_2 import forward_dynamics_with_constraints
from answer_4 import gravity_compensation
    
import argparse

# 在这个任务中，我们将problem_2中创建的三摆视为一个机械臂，通过关节力矩来实现重力补偿，从而使得机械臂保持在一个静止的姿态
def problem_4(args):                
    (m, I, inv_m, inv_I, x, R, v, w, f, tau, h, g, joints, box_meshes, box_vertices, plotter) = create_triple_pendulum(args)
        
    # 随机机械臂姿势
    init_joint_rots = np.stack([np.eye(3)]*3, axis=0)
    # np.random.seed(0)
    def set_random_pose():
        nonlocal x, R, v, w, init_joint_rots
        init_joint_rots = Rotation.from_euler('XYZ', np.random.uniform(-np.pi, np.pi, (3,3))).as_matrix()
        x, R = forward_kinematics(num_body=4, x_0=np.zeros(3), joint_rots=init_joint_rots, joints=joints)
        utils._update_boxes(x[:-1], R[:-1], box_meshes, box_vertices, plotter)
        v[:] = 0
        w[:] = 0
        
    set_random_pose()
    
    apply_joint_torques = joint_torque_converter(m.shape[0], joints)
    
    ###########################
    # 开始仿真
    # 可以通过在控制台 ctrl+C 中止
    fps = 60
    sub_steps = max(1, int(1/fps/h))
    
    for t in range(args.frames):
        if t > 0 and t % 90 == 0:
            set_random_pose()
            
        for i in range(sub_steps): 
            joint_torques = gravity_compensation(m, g, x, R, joints)
            
            # 将关节力矩施加在刚体上
            tau = apply_joint_torques(joint_torques)
            
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
            
    plotter.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem 4')
    parser.add_argument('--video', type=str, default=None, help='video name')
    parser.add_argument('--frames', type=int, default=900, help='number of frames')
    
    args = parser.parse_args()
    problem_4(args)