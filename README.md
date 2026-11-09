![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![MuJoCo Warp](https://img.shields.io/badge/MuJoCo_Warp-3.16.0-blue)](https://github.com/google-deepmind/mujoco_warp/releases/tag/v3.16.0)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)
[![PyPI downloads](https://img.shields.io/pypi/dm/mjlab?color=blue)](https://pypistats.org/packages/mjlab)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).
The framework provides composable building blocks for environment design,
with minimal dependencies and direct access to native MuJoCo data structures.

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/main/source/installation.html).

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions. See the [motion imitation guide](https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html) for preprocessing setup.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.


## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

## Citation

mjlab is used in published research and open-source robotics projects. See the [Research](https://mujocolab.github.io/mjlab/main/source/research.html) page for publications and projects, or share your own in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell).

If you use mjlab in your research, please consider citing:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## L20 HORA rotation

The L20 task follows the same layout as the manipulation tasks:

- `src/mjlab/tasks/rotation/mdp/` contains actions, resets, observations,
  rewards, terminations, and per-domain metrics.
- `src/mjlab/tasks/rotation/config/l20/` contains environment variants,
  PPO configuration, and task registration.
- `src/mjlab/tasks/rotation/rl/` contains the privileged model and runner.
- `src/mjlab/tasks/rotation/assets/` contains model specifications and grasp banks.

The shared official-size sphere/cylinder/box task is
`Mjlab-Rotation-L20-Official-Mixed`. Its shape allocation is equal, and each
reset restores the complete cached state with zero velocity and joint targets
equal to the restored positions. The 9D privileged input is encoded into 8D
for the actor and critic. Training statistics and configuration are saved in
the run directory under `logs/rsl_rl/l20_hora_stage1_official_mixed/`.

Local WSL entry:

```bash
PYTHONPATH=src uv run --no-project ../teleop/.venv/bin/python -m mjlab.scripts.train Mjlab-Rotation-L20-Official-Mixed
```

Only `Mjlab-Rotation-L20-Official-Mixed` is registered. Earlier training logs
and checkpoints are retained. Superseded cache copies have been archived under
`logs/legacy_rotation_assets/`; the task reads only `assets/official_mixed_v1/`.
The current grasp-generation tools are in the sibling `hora/scripts/` directory.

### HORA 对齐范围

对照本地 HORA 提交 `410d95824dd28b198f6d6910cb4c09330a6303be` 的
`train_s1.sh`、任务配置、`allegro_hand_hora.py`、`models.py` 和 `ppo.py`。
当前代码定位为 L20/MuJoCo Warp 上的 HORA Stage 1 多形状扩展。

| 项目 | 原版 HORA 公开实现 | 当前 L20 | 状态 |
|---|---|---|---|
| 训练分布 | 默认多尺寸、多高径比圆柱，一个共享策略 | 球/圆柱/方块各 1/3，一个策略 | 用户要求的扩展 |
| 尺寸与 DR | .70–.86 九档，中心附近 ±.025；质量 .01–.25 kg、COM 每轴 ±.01 m、摩擦 .3–3 | 范围和创建时采样/分档直接复位机制保留 | 已对齐机制与主要范围 |
| 抓姿缓存 | 标准手型附近 ±.25 rad，球体生成，每尺寸档 50,000 条 | L20 各形状真实几何求解，±.25 rad，99 档共 4552 条，离线标准更严格 | 跨硬件适配，覆盖规模有差异 |
| 复位 | 缓存关节/物体位置与朝向，零速度，目标等于恢复位置 | 完整 qpos，包括 L20 mimic 从动关节；同样零速度与位置目标 | 语义对齐，少数缓存存在回放敏感性 |
| 本体观测 | 3帧 ×（16关节归一化位置 + 16目标）=96D | 同样 96D，L20 为五指、16主动关节及从动约束 | 维度/形式对齐，关节语义不同 |
| 特权输入 | 位置3、缩放1、质量1、摩擦1、COM3；9→256→128→8、ELU+tanh | 同样维度、结构，读取实际物理参数，Actor 使用编码 | 已对齐主结构 |
| Actor/Critic | 一个共享编码器/主干，再接 mu 与 value 头 | HoraPPO 将 Actor/Critic 的编码器与512/256/128主干绑定为同一组参数，动作/价值头独立 | Stage 1结构对齐 |
| 动作 | 20 Hz，动作裁至 ±1，目标每步加 action/24 rad | 相同；物理步 2 ms、25子步，原版约8.33 ms、6子步 | 控制接口对齐，积分/接触不同 |
| 低层控制 | 显式 PD 力矩，P=2.9–3.1、D=.09–.11，力矩裁 ±.5 N·m | MuJoCo 位置伺服，P约19.33–20.67、D=.9–1.1，URDF关节限力 ±1 N·m | L20 适配，未对齐绝对值 |
| 奖励 | 目标轴角速度截断 ±.5、线速度L1、姿态平方、力矩平方、功率平方；权重1/−.3/−.3/−.1/−2 | 同类公式与权重；目标轴由 −Z 映射到手坐标 −X | 主形式对齐，力矩/速度采样见下文 |
| 终止 | 物体世界高度低于阈值，400控制步超时 | 偏离复位点3 cm，包括横向滑移；400步超时 | 掉落定义不同 |
| 噪声 | 关节 ±.02 rad；复位首帧历史用无噪声位置回填 | 同样幅度，复位首帧也经过噪声函数 | 首帧历史存在差异 |
| 外力 | 尺度2m、概率.25；pre_physics_step按模拟dt计算衰减指数 | 同尺度/概率，按控制周期计算 .9**(step_dt/.08) | 离散时序不同 |
| PPO主要标量 | LR .005、gamma .99、GAE .95、clip .2、5 epochs、horizon8、entropy0 | 相同；任务内 HoraPPO 复用 RSL-RL 存储与分布 | 环境奖励权重不变，采集时按原版乘 .01 |
| value与边界项 | return/value running normalization，bounds_loss_coef=.0001；总loss含 .5×critic_coef | 原版RMS更新/裁剪与反归一化、values后returns更新；value_loss_coef=2；原版bounds公式 | 按公开代码对齐 |
| 输入归一化与更新 | 输入RMS在优化minibatch更新；env-major连续批次，更新old mu/sigma；每epoch调整LR | 同样执行，共享参数只裁剪一次梯度 | 按公开代码对齐 |
| 批量与预算 | 配置默认16384环境、minibatch32768、最多1.5B步 | 默认1539环境、4个minibatch（3078条/批）；121833更新=1500007896步 | 总预算对齐，绝对并行数/批量仍不同 |
| Stage 2 / 真机 | 有本体历史适应模块及部署流程 | 当前任务只接入特权 Stage 1 | 尚缺适应训练与部署验证 |

力矩项仍需精确对照：HORA 使用已经裁剪的命令力矩；当前 `torque`
读取 `actuator_force`，`work` 使用 `qfrc_actuator`。已确认执行器原始输出可以超过关节汇总限力，当前惩罚取样与原版
裁剪后的命令力矩不能视作等价。功率项的速度也不同，原公开代码
保留最后一个物理子步的差分速度，当前使用一个控制周期的关节位置差。

因此，仍不能称为原版物理控制与训练规模的逐项复刻。低层力矩/功率取样、
外力时序、批量规模等差异保留如上。bounds loss 严格保留原代码的
`clamp_max(mu-1.1,0)^2 + clamp_max(-mu+1.1,0)^2`，并非通常的对称越界惩罚。
新的共享网络与RMS状态不兼容此前独立网络检查点，不自动转换或覆盖。
当前训练使用新的共享网络与HoraPPO；原独立网络检查点不能直接续训。
从1026增至1539环境时，保留第800次检查点续训，并按已采样步数扣除剩余预算。
默认从零训练预算适用于1539环境，
手动修改环境数时须按 `ceil(1.5e9/(num_envs*8))` 同步调整更新次数。
还需扩充稀疏抓姿与独立测试分布，完成 Stage 2 和真机验证。

1539环境在本机RTX 4060 Laptop 8GB上实测总显存7103–7107 MiB（约6.94 GiB），
吞吐约9000环境步/秒；环境数不取决于抓姿库条目数。检查点保存在硬盘，
增加并行环境后仍需为桌面及运行时波动保留显存。续训时保留优化器的自适应学习率。

实现检查：RMS与原版逐值对照、bounds公式对照、共享参数检查、3次CPU PPO更新
及检查点恢复通过；27环境MuJoCo Warp完成5次更新（1080条环境步），损失有限，
正常退出。这是实现验证，不代表旋转训练效果。验证日志保存在
`logs/hora_alignment_validation/`，原有基线日志与检查点保留。

原始依据：
[训练入口](https://github.com/HaozhiQi/hora/blob/410d95824dd28b198f6d6910cb4c09330a6303be/scripts/train_s1.sh)、
[环境与控制](https://github.com/HaozhiQi/hora/blob/410d95824dd28b198f6d6910cb4c09330a6303be/hora/tasks/allegro_hand_hora.py)、
[网络](https://github.com/HaozhiQi/hora/blob/410d95824dd28b198f6d6910cb4c09330a6303be/hora/algo/models/models.py)、
[PPO](https://github.com/HaozhiQi/hora/blob/410d95824dd28b198f6d6910cb4c09330a6303be/hora/algo/ppo/ppo.py)。

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
