
import utils
from answer_1 import integration, forward_dynamics

import numpy as np
from scipy.spatial.transform import Rotation
import argparse

# 在这个任务中，我们创建一个具有一定初速度/角速度的刚体方块，如果你的回答正确，你应该可以看到它在缓慢的移动并旋转
def problem_1(args):    
    
    num_bodies = 3
        
    # 三个刚体的质量和转动惯量
    m = np.array([1.0, 2.0, 3.0])
    I = np.stack([np.diag(np.array([1.0, 2.0, 4.0]) * i) for i in [0.1, 0.2, 0.3]], axis=0) # 第四个刚体的转动惯量为0，表示空间本身
    
    # 以下两个属性是为了方便后续计算
    inv_m = 1/m
    inv_I = np.linalg.inv(I)
    
    # 三个刚体的初始位置
    x = np.array([
        [ 0, 0, 0],
        [-1, 0, 0],
        [ 1, 0, 0]
        ])
    
    # 三个刚体的初始朝向
    R = np.stack([np.eye(3)]*num_bodies, axis=0)
    
    # 三个刚体的速度
    v = np.array([
        [ 0, 0, 0],
        [ 0.05, 0, 0],
        [-0.05, 0, 0]
        ])
    
    # 三个刚体的角速度
    w = np.array([
        [3, 2, 1],
        [0, 0.2, 0.1],
        [0, 0, 1]
        ])

    # 施加在刚体上的力和力矩
    f = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0.]
    ])
    
    tau = np.array([
        [0,0,0],
        [0,0,0],
        [0.,0,0]
    ])

    # 时间步长
    h = 0.001
    # h = 0.016
    
    # 一些显示设置
    import pyvista as pv
    plotter = pv.Plotter()
    box_meshes = [pv.Box([-0.1, 0.1, -0.08, 0.08, -0.15, 0.15]) for i in range(num_bodies)]
    box_vertices = np.stack([box.points for box in box_meshes], axis=0).reshape(num_bodies, -1, 3, 1)
    for i, box in enumerate(box_meshes):
        plotter.add_mesh(box, color='rcy'[i%3])
        
    plotter.add_axes()    
    plotter.show(auto_close=False, interactive_update=True)
    plotter.view_xz(bounds=[-0.8, 0.8, -0.5, 0.5, -0.5, 0.5])
        
    utils._update_boxes(x, R, box_meshes, box_vertices, plotter)
    
    if args.video is not None:
        if args.video.endswith('.gif'):
            plotter.open_gif(args.video, fps=30)
        else:
            plotter.open_movie(args.video, framerate=30, quality=10)
    
    # 开始仿真
    # 可以通过在控制台 ctrl+C 中止
    fps = 60
    sub_steps = max(1, int(1/fps/h))
    
    for t in range(args.frames):
        for i in range(sub_steps):
            f[2, 2] = np.cos((t*sub_steps+i)*h)
            tau[2, 1] = np.cos((t*sub_steps+i)*h)
                    
            v_next, w_next = forward_dynamics(m, I, inv_m, inv_I, x, R, v, w, f, tau, h)
            x_next, R_next = integration(x, R, v_next, w_next, h)
        
            x, R, v, w = x_next, R_next, v_next, w_next
        
        try:
            with pv.VtkErrorCatcher(True) as error_catcher:
                utils._update_boxes(x, R, box_meshes, box_vertices, plotter)
                
            if args.video is not None:
                plotter.write_frame()
        except:
            exit(0)
            
        if args.debug_mode:
            print('angular momentum:', utils.angular_momentum(m[:1], I[:1], x[:1], R[:1], v[:1], w[:1]))        
        
    plotter.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem 1')
    parser.add_argument('--video', type=str, default=None, help='video name')
    parser.add_argument('--frames', type=int, default=600, help='number of frames')
    parser.add_argument('--debug-mode', action='store_true', help='debug mode')
    
    args = parser.parse_args()
    problem_1(args)