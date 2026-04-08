import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHMotion
from physics_warpper import PhysicsInfo
from answer_project import *
from viewer.viewer_new import SimpleViewer

def compute_angular_velocity_from_quat(quats: np.ndarray, dt: float) -> np.ndarray:
    """
    计算角速度（轴角）: v = log(q_{t}^{-1} * q_{t+1}) / dt
    输入：
        quats: shape (T, J, 4)，T 帧数，J 关节数，四元数为 (x, y, z, w)
        dt: 每一帧之间的时间间隔（s）
    输出：
        ang_vels: shape (T-1, J, 3)，角速度向量
    """
    T, J, _ = quats.shape
    ang_vels = np.zeros((T - 1, J, 3))

    for t in range(T - 1):
        R1 = R.from_quat(quats[t])      # shape (J,)
        R2 = R.from_quat(quats[t + 1])  # shape (J,)
        delta = R2 * R1.inv()           # 相对旋转
        log_rot = delta.as_rotvec()     # 旋转向量
        ang_vels[t] = log_rot / dt      # 角速度 = 旋转角 / 时间
    return ang_vels


class CharacterWalkEnv:
    def __init__(self, viewer):
        self.viewer = viewer
        self.physics = PhysicsInfo(viewer)
        self.num_joints = 20
        self.action_dim = self.num_joints * 3
        self.max_steps = 300
        self.reference_motion = BVHMotion("motion_material/walk.bvh")
        self.reference_motion.adjust_joint_name(self.physics.joint_name)
        # self.frame_id = 0
        self.max_frame = self.reference_motion.joint_position.shape[0]
        self.reference_motion_joint_angular_velocity = compute_angular_velocity_from_quat(
            self.reference_motion.joint_rotation, dt=0.010000
        )
        self.reset()

    def reset(self):
        # self.cur_step = 0
        # self.frame_id = 0
        self.idx = np.random.randint(0, self.reference_motion.num_frames - 2)
        self.cur_step = self.idx
        # idx = 0
        self.target_pose = self.reference_motion.joint_rotation[self.idx]
        return self.get_obs()

    # action -> state,reward
    def step(self, action):
        action = action.reshape(self.num_joints, 3)
        self.viewer.set_torque(action)
        self.viewer.simulationTask()
        
        self.cur_step += 1

        reward = self.get_reward()
        done = self.cur_step >= self.max_steps or self.cur_step >= self.max_frame-1 or self.is_fallen()
        
        return self.get_obs(), reward, done, {}

    def get_obs(self):
        joint_rot = self.physics.get_body_orientation()    # 全局还是局部？
        root_pos, _ = self.physics.get_root_pos_and_vel()
        root_rot = joint_rot[0]
        return np.concatenate([root_pos, root_rot, joint_rot[1:].flatten()])

    def get_reward(self):
        w_p, w_v, w_e, w_c = 0.65, 0.1, 0.15, 0.1
        def compute_pose_reward(ref_quat: np.ndarray, curr_quat: np.ndarray) -> float:
            """
            计算 pose reward（关节朝向差异）
            :param ref_quat: (N, 4) 的参考四元数数组，BVH pose
            :param curr_quat: (N, 4) 的当前模拟器角色的四元数数组
            :return: pose reward
            """
            ref_R = R.from_quat(ref_quat)     # shape: (N,)
            curr_R = R.from_quat(curr_quat)   # shape: (N,)
            q_diff = ref_R * curr_R.inv()     # shape: (N,)
            rotvec = q_diff.as_rotvec()       # shape: (N, 3) 转成轴角表示
            angle_diffs = np.linalg.norm(rotvec, axis=1)  # (N,)
            total_squared_angle = np.sum(angle_diffs ** 2)

            pose_reward = np.exp(-2 * total_squared_angle)
            return pose_reward
        def compute_avelocity_reward(ref_avel: np.ndarray, curr_avel: np.ndarray) -> float:
            diff = ref_avel - curr_avel
            return np.exp(-0.1 * np.sum(diff ** 2))

        def compute_end_effector_reward(ref_ee_pos: np.ndarray, curr_ee_pos: np.ndarray) -> float:
            diff = ref_ee_pos - curr_ee_pos
            return np.exp(-40 * np.sum(diff ** 2))

        def compute_com_reward(ref_com: np.ndarray, curr_com: np.ndarray) -> float:
            diff = ref_com - curr_com
            return np.exp(-10 * np.sum(diff ** 2))

        i = self.cur_step
        # pose reward
        joint_rotation = self.reference_motion.joint_rotation
        joint_position = self.reference_motion.joint_position
        # _, target_pose = self.reference_motion.batch_forward_kinematics(joint_position, joint_rotation, i)
        joint_translation, target_poses = self.reference_motion.batch_forward_kinematics()
        target_pose = target_poses[i]
        current_pose = self.physics.get_body_orientation()
        rp = compute_pose_reward(target_pose, current_pose)

        # angular velocity reward
        target_avel = self.reference_motion_joint_angular_velocity[i]
        current_avel = self.physics.get_body_angular_velocity()
        rv = compute_avelocity_reward(target_avel, current_avel)

        # end-effector reward（末端执行器：左右手脚）
        ee_names = ["lToeJoint", "rToeJoint", "lWrist", "rWrist"]
        name_to_index = {name: idx for idx, name in enumerate(self.reference_motion.joint_name)} # 构建关节名到索引的映射表
        ee_indices = [name_to_index[name] for name in ee_names]  # 找到这些末端执行器在数组中的索引
        
        target_ee = self.reference_motion.joint_position[i][ee_indices]
        current_ee = self.physics.get_body_position()[ee_indices]
        re = compute_end_effector_reward(target_ee, current_ee)

        # COM reward 姑且认为是根结点
        target_com = joint_translation[i][0]
        current_com = self.physics.get_joint_translation()
        rc = compute_com_reward(target_com, current_com)

        reward = w_p * rp + w_v * rv + w_e * re + w_c * rc
        
        return reward

    def is_fallen(self):
        root_pos = self.physics.get_joint_translation()[0]
        return root_pos[1] < 0.5