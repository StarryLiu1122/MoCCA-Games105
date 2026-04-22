import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv

import utils
from joints import Joint, joint_torque_converter
from triple_pendulum import create_triple_pendulum, forward_kinematics

from answer_1 import integration
from answer_2 import forward_dynamics_with_constraints
from answer_6 import end_effector_track_control
    
import argparse

# 结合前面所有的回答，实现一个简单的机械臂控制器，实现末端点位置控制
def problem_5(args):                
    (m, I, inv_m, inv_I, x, R, v, w, f, tau, h, g, joints, box_meshes, box_vertices, plotter) = create_triple_pendulum(args)
    # 更改初始pose，更加接近轨迹一些
    x, R = forward_kinematics(num_body=4, x_0=np.zeros(3), joint_rots=np.stack([Rotation.from_euler('y', np.pi/3).as_matrix()]*3, axis=0), joints=joints)
    utils._update_boxes(x[:-1], R[:-1], box_meshes, box_vertices, plotter)
    
    # 末端点定义
    end_effector_body = 2
    end_effector_from_body = np.array([1.0, 0.0, 0.0])
    
    current_end_effector_pos = x[end_effector_body] + R[end_effector_body] @ end_effector_from_body
    
    # target_trajectory() 的定义见下面
    trajectory_gen = target_trajectory(center=np.array([0, 0, -4]), normal=np.array([1, -1, 1]), radius=3, omega=2*np.pi/4, plotter=plotter)
        
        
    apply_joint_torques = joint_torque_converter(m.shape[0], joints)
    ###########################
    # 开始仿真
    # 可以通过在控制台 ctrl+C 中止
    fps = 60
    sub_steps = max(1, int(1/fps/h))
    
    for n in range(args.frames):            
        for i in range(sub_steps): 
            t = (n * sub_steps + i) * h
            target_pos, target_vel = trajectory_gen(t)
            joint_torques = end_effector_track_control(m, I, g, x, R, v, w, joints, 
                                                       end_effector_body, end_effector_from_body,
                                                       target_pos, target_vel)
            
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

# 目标末端点轨迹是一个倾斜的圆形
def target_trajectory(center:np.ndarray, normal:np.ndarray, radius:float, omega:float, plotter:pv.Plotter):
    # 预计算圆的参数
    normal = normal / np.linalg.norm(normal)  # 单位化法线方向

    # 定义圆的平面上的两个正交向量
    tangent1 = np.cross(normal, [0, 0, 1])
    if np.linalg.norm(tangent1) < 1e-6:  # 如果法线接近平行于z轴
        tangent1 = np.cross(normal, [0, 1, 0])
    tangent1 = tangent1 / np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)

    def compute_position_velocity(t):
        # 计算圆上的位置和速度
        angle = omega * t
        position = center + radius * (np.cos(angle) * tangent1 + np.sin(angle) * tangent2)
        velocity = radius * omega * (-np.sin(angle) * tangent1 + np.cos(angle) * tangent2)
        return position, velocity
    
    
    if plotter is not None:        
        # 可视化目标轨迹
        num_points = 100
        angles = np.linspace(0, 2 * np.pi, num_points)
        ts = angles / omega  # 将角度转换为时间
        positions = np.array([compute_position_velocity(t)[0] for t in ts])
        plotter.add_points(positions, style='points_gaussian', point_size=5, color='red')

    return compute_position_velocity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem 4')
    parser.add_argument('--video', type=str, default=None, help='video name')
    parser.add_argument('--frames', type=int, default=1500, help='number of frames')
    
    args = parser.parse_args()
    problem_5(args)