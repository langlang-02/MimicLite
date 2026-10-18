# 官方 MimicLite PPO-Huge：网络、训练数据与预算

整理日期：2026-10-15。

本文对应本次下载并评测的官方 G1 PPO-Huge，W&B run `4234dd57`，权重 `checkpoint_4000.pt`。这些配置不代表所有 MimicLite 版本，也不是 v1.1 ROA student 的配置。

## 依据与文件

- [官方训练记录](https://wandb.ai/elijahgalahad/mimic_lite/runs/4234dd57)
- [权重附带的完整配置](../../upstream/checkpoints/mimic-lite/ppo/files/cfg.yaml)
- [本地权重](../../upstream/checkpoints/mimic-lite/ppo/checkpoint_4000.pt)
- [官方发布表：网络、环境数与 GPU 小时](https://github.com/EGalahad/mimic-lite/blob/feat/multi-robot-support/README.md#released-checkpoints)（2026-10-15 核对；网页后续可能更新）

## 1. 网络结构

| 项目 | 官方 PPO-Huge | 我们上一轮不带 XY 反馈的 latent tracker |
|---|---|---|
| Actor | 839 → 1024 → 1024 → 1024 → 29 | 1048 → 512 → 512 → 512 → 29 |
| Critic | 1389 → 1024 → 512 → 512 → 2 | 1598 → 1024 → 512 → 256 → 2 |
| 参考动作输入 | 显式参考位置、朝向、关节角 | RVQ latent + phase |

箭头包含输入和输出维度。官方只有 Actor 的三个隐藏层均为 1024；Critic 隐藏层是 `[1024, 512, 512]`。

官方网络是前馈 MLP，使用 LayerNorm 和 Mish，没有循环网络或 Transformer；`use_actor_encoder=false`。Actor 输出 29 个关节动作的高斯均值，并有可学习的标准差。Critic 输出对应两个奖励组的价值。

### Actor 输入：839 维

**自身状态 `policy`：535 维。**

| 内容 | 维度 |
|---|---:|
| 根角速度历史 | 3 × 7 = 21 |
| 投影重力历史 | 3 × 7 = 21 |
| 关节位置历史 | 29 × 7 = 203 |
| 关节速度历史 | 29 × 7 = 203 |
| 前 3 步动作 | 29 × 3 = 87 |

状态历史索引为 `[0, 1, 2, 3, 4, 8, 16]`，控制频率 50 Hz，最远回看 0.32 秒。

**参考动作 `command`：304 维。**

| 内容 | 每个参考时刻的维度 | 配置项 |
|---|---:|---|
| pelvis 参考位置的局部表征 | 3 | `ref_root_pos_future_local` |
| 参考根朝向的相对 6D 表征 | 6 | `ref_root_ori_future_b` |
| 参考关节位置 | 29 | `ref_joint_pos_future` |

参考时间偏移为 `[-8, -4, -2, 0, 1, 2, 3, 4]`，每个时刻 38 维，合计 `8 × 38 = 304` 维。按 50 Hz 计算，覆盖过去 160 ms 至未来 80 ms。

官方 tracker 直接读参考动作，不使用我们训练的 MoMask RVQ 编码。我们的不带 XY 反馈版本是 `535 + 513 = 1048` 维，其中 513 为 latent 与 phase；新增 XY 对照实验则为 1050 维。

### Critic 输入：1389 维

包含 Actor 的 839 维输入，以及 550 维 `priv` 特权信息。配置中的特权信息包括参考根位置/朝向、局部身体空间误差、身体线速度/角速度差，以及实际施加的动作和力矩。它们用于训练价值函数，部署 Actor 时不需要输入这 550 维。

## 2. 训练数据混合配置

下表逐项对应已下载配置的 `task.command.motion_cfgs`。权重合计为 1，是采样权重，不是动作数量或总时长占比。

| 数据池 | 权重 | 数据源 | 文件列表筛选 |
|---|---:|---|---|
| LAFAN | 20% | `hf://elijahgalahad/any4hdmi-g1-lafan` | 未指定 |
| 100STYLE | 20% | `hf://elijahgalahad/any4hdmi-g1-100style` | 未指定 |
| SONIC 筛选子集 | 30% | `../any4hdmi/output/g1/sonic` | `limmt/passed/filenames.txt` |
| Ground motions | 5% | `hf://elijahgalahad/any4hdmi-g1-ground-motions` | 未指定 |
| AMASS hard | 2.5% | `hf://elijahgalahad/any4hdmi-g1-amass-hard` | `views/train_dynamic.txt` |
| Extreme motions | 2.5% | `hf://elijahgalahad/any4hdmi-g1-extreme-motions` | 未指定 |
| Extreme demo | 2.5% | `hf://elijahgalahad/any4hdmi-g1-extreme-demo` | 未指定 |
| High dynamic sequences | 2.5% | `hf://elijahgalahad/any4hdmi-g1-high-dynamic-sequences` | 未指定 |
| SONIC 单腿严格筛选子集 | 10% | `../any4hdmi/output/g1/sonic` | `views/strict_one_leg.txt` |
| Real PICO | 5% | `hf://elijahgalahad/any4hdmi-g1-real-pico` | 未指定 |

所有池均设置 `full_motion: true`，以完整动作序列加载；SONIC 的 30% 池还设置了 `shard: true`。表中 SONIC 路径是作者训练环境的相对路径，不表示本机已有这些数据。

SONIC 两个池合计占 40% 权重，但筛选列表可能重叠。仅凭配置不能确定总动作条数、去重后总时长，也不能确认官方数据与我们的固定评测集无重叠。因此，本次官方 17/20 的结果可作为跟踪能力对照，不能直接解释为严格无重叠的泛化成绩。

## 3. 训练预算

| 参数 | 官方 PPO-Huge | 我们上一轮 128 条动作训练 |
|---|---:|---:|
| GPU | 16 × RTX 4090 | 4 × A800 |
| 每卡环境数 | 16,384 | 8,192 |
| 总环境数 | 262,144 | 32,768 |
| 每轮每环境 rollout 步数 | 32 | 32 |
| 训练迭代数 | 4,000 | 2,288 |
| PPO epochs | 5 | 3 |
| 每 epoch minibatches | 8 | 8 |
| 优化器 | Muon | Muon |
| 初始学习率 | 0.0003 | 0.0003 |
| 本轮采样 transition 数 | 33,554,432,000 | 2,399,141,888 |

官方总采样量为 `16 × 16384 × 32 × 4000`，约 335.54 亿条环境 transition，是我们上一轮新增采样量的约 14 倍；我们的数字不包含热启动 checkpoint 先前的训练量。这里统计的是控制步，不是物理仿真子步。

官方每轮最多执行 `5 × 8 = 40` 次 minibatch 更新；启用了 `epoch_kl_early_stop`，实际次数可能更少。DDP 的多卡共同执行一次同步更新，不应再把更新次数乘以卡数。

配置同时保留 `total_frames: 150000000` 与 `total_iters: 4000`。训练脚本优先使用显式的 `total_iters`，不能把此次预算误读为 1.5 亿步。

官方发布表记录计算预算为 92.3 GPU·小时（RTX 4090）；按 16 卡全程同时运行折算，墙钟时间约 5.77 小时。这是折算值，不能直接预测 A800 上的耗时。

官方配置 `checkpoint_path: null`、`seed: 0`；我们对照的上一轮 128 条动作训练从 small12 checkpoint 热启动。网络、参考信息、动作数据、训练预算和初始化均有差异，不能把效果差异单独归因于网络宽度。

## 4. 相关笔记

- [W&B 指标、rank 0 与输入归一化](wandb-metrics-and-distributed-normalization.md)：各指标含义、期望趋势、分布式统计和历史训练链接。
- [官方 G1 tracker 评测记录](official-g1-tracker-evaluation.md)：本次下载、兼容适配和固定评测集结果。
