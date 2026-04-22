# MoCCA-Games105: 角色动画与物理仿真

![PKU](https://img.shields.io/badge/University-PKU-red)
![Course](https://img.shields.io/badge/Course-Games105-blue)
![Field](https://img.shields.io/badge/Field-Computer_Graphics-green)

> **北京大学《角色动画与物理仿真》课程 (Games105 / MoCCA) 实验与项目代码仓库**

---

## 📌 课程简介

本仓库记录了在北京大学修读《角色动画与物理仿真》课程期间的学习成果。课程内容涵盖了从传统动画技术到前沿物理仿真及 AI 驱动动画的核心算法。

**核心技术栈：**
* **角色运动学：** FK/IK、雅可比矩阵、CCD 算法。
* **动画合成：** 关键帧插值、运动匹配 (Motion Matching)。
* **物理仿真：** 刚体/柔体动力学、质点弹簧系统、碰撞处理。
* **前沿技术：** 强化学习 (RL) 驱动的角色控制。

---

## 📂 仓库结构

```text
MoCCA-Games105/
├── 📄 CharacterAnimation.pdf         # 课程核心参考资料
├── 📂 Lecture slides/                # 课程讲义 (L01 - L13)
│   ├── 01-02 Math Background         # 数学基础与引言
│   ├── 03-06 Animation Tech          # 运动学、关键帧与数据驱动
│   └── 07-13 Simulation & RL         # 物理仿真、强化学习与路径规划
├── 📂 lab1/                          # 【实验一】角色运动学基础 (FK/IK实现)
├── 📂 lab2/                          # 【实验二】关键帧动画与插值算法
├── 📂 lab3/                          # 【实验三】数据驱动动画与运动匹配
└── 📂 lab4/                          # 【实验四】物理仿真与动力学模拟
```

## 🧪 实验内容详述

| 项目 | 主题 | 核心算法与任务 | 交付物 |
| :--- | :--- | :--- | :--- |
| **Lab 1** | **角色运动学基础** | 实现正向运动学 (FK) 与逆向运动学 (IK)；解析 BVH 动作文件；处理关节链层级变换。 | 💻 核心代码 |
| **Lab 2** | **关键帧动画与插值** | 实现线性插值 (LERP)、球面线性插值 (Slerp)；应用 Catmull-Rom 样条曲线进行平滑处理。 | 📄 [实验报告](./lab2/lab2实验报告_2300012297_刘星云.docx) |
| **Lab 3** | **数据驱动动画** | 运动库检索与特征匹配；实现运动匹配 (Motion Matching) 技术；确保动作切换的自然性。 | 📄 [实验报告](./lab3/lab3实验报告_2300012297_刘星云.pdf) |
| **Lab 4** | **物理仿真** | 刚体/柔体动力学模拟；质点弹簧系统 (Mass-Spring System)；实现高效的碰撞检测与响应。 | 📄 [实验报告](./lab4/lab4实验报告_2300012297_刘星云.docx) |

## 🚀 运行实验

### 1. 环境准备

本项目建议在 `Python 3.8+` 环境下运行。推荐使用 [Conda](https://www.anaconda.com/) 管理虚拟环境以避免依赖冲突：

```bash
# 创建并激活环境
conda create -n games105 python=3.10
conda activate games105

# 安装核心依赖
pip install numpy scipy matplotlib glfw PyOpenGL
```

### 2. 实验执行

每个实验文件夹（lab1-lab4）中均包含对应的启动脚本。请进入相应目录后运行，例如启动 Lab 1 的运动学演示：

```Bash
cd lab1/MoCCA25-Lab1
python task1_forward_kinematics.py
```

## 💻 核心技术点

本项目的代码实现深度涉及以下计算机图形学与动画核心领域：

- 数学基础 (Mathematics): 深入应用四元数 (Quaternions) 计算旋转、齐次坐标变换及 Jacobian 矩阵。

- 运动学 (Kinematics): 实现了基于层级骨骼的 FK，以及基于 CCD 和 FABRIK 的逆运动学解算。

- 动画合成 (Synthesis): 涵盖 BVH 动作捕捉数据解析、运动混合 (Blending) 以及运动匹配 (Motion Matching)。

- 物理仿真 (Physics): 包含基于位置的动力学 (PBD)、隐式/显式欧拉积分、质点弹簧系统及碰撞检测算法。

- 智能控制 (Control): 探索了强化学习算法（如 PPO）在物理骨骼轨迹跟踪与平衡控制中的应用。

## 📚 参考文献与致谢

在学习与实验过程中，参考了以下优秀的资源与教材：

- Games105 课程组： 诚挚感谢北京大学 MoCCA 实验室提供的实验框架与专业教学。

- Rick Parent: Computer Animation: Algorithms and Techniques —— 提供了扎实的算法理论基础。

- GAMES 平台： 感谢 GAMES 系列课程提供的学术交流平台。

## ⚖️ 许可与规范

1. 学术诚信： 本仓库的所有代码及实验报告仅供个人学习记录与学术交流使用。严禁北京大学或其他高校同学直接搬运本项目代码作为课程作业提交，请务必维护学术诚实。

2. 版权声明： 课程讲义及实验原始框架的知识产权归 MoCCA 课程组 所有。
   - 本项目中的具体算法实现、代码修改部分及实验报告文字归作者所有。

## 👤 联系作者

作者： 刘星云 (Xingyun Liu)

邮箱： i2793521817@outlook.com

研究方向：机器人具身智能

最后更新日期: 2026年4月22日

<center><b>⭐ Star us on GitHub if the repository helps your research!</b></center>

<center>Made with ❤️ by Xingyun Liu</center>