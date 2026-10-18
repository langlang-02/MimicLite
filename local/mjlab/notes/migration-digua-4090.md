# digua_4090_48 全量训练迁移记录

迁移已完成，四卡短验证通过，正式训练已经启动。这里记录的是启动验证结果，不代表 2500 轮训练已经结束或策略已经收敛。

## 位置与环境

目标机使用 root，在 `/data/langlang` 下开发：

| 内容 | 路径 |
| --- | --- |
| 仓库 | `/data/langlang/Project/MimicLite` |
| 数据 | `/data/langlang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1` |
| 热启动 checkpoint | `/data/langlang/checkpoints/mimiclite/train1024-entropy-2500.pt` |
| FK 缓存 | `/data/langlang/cache/mimiclite-fk-cache` |
| 训练与历史记录 | `/data/langlang/runs/mimiclite` |
| 历史实验原始文件 | `/data/langlang/archives/a800-experiments` |
| 迁移检查记录 | `/data/langlang/transfer` |

Python 3.12.13，PyTorch 2.10.0+cu128，MuJoCo 3.7.0，mujoco-warp 3.7.0.1，mjlab 1.3.0，Warp 1.12.0。驱动 570.195.03，RTX 4090 每卡 49140 MiB。Python 环境、资产和运行缓存位于数据盘。

## 数据与 Git

数据通过 bcecmd 从 `bos:/langlang/mimiclite/migration-20261006/` 恢复。源机当前数据与归档逐文件对照通过；目标机再次验证了 26 个分包及 187792 个文件的 SHA256，共 10681995403 字节。训练索引保持 `flat_ground_v1/train.txt`，包含 159777 条动作。12/128/1024 条子集及固定评测索引保存在仓库中。

新增归档位于 `bos:/langlang/mimiclite/migration-20261018/`，包括历史运行记录、热启动模型、机器人资产和 Git bundle。FK 缓存在目标机重建，避免复用包含旧机器路径的缓存身份。

三个远端均已推送：

- 主仓库：`langlang-02/MimicLite`，`feat/rvq-latent-tracker`。
- 算法子模块：`langlang-02/mimic-lite`，`feat/rvq-latent-tracker`，运行版本 `d6513bd`。
- 框架及实验记录：`langlang-02/active-adaptation`，`feat/mimiclite-latent-experiments`，运行版本 `ab45406`。

主仓库按本地启动、动作索引、评测分析、迁移适配划分提交；mimic-lite 按指标记录、PPO 诊断、可选 XY 输入划分；active-adaptation 按历史配置结果、评测诊断工具、吞吐测试划分。生成的二进制和原始输出保存在 BOS 归档。可选 sim2real 的懒加载补丁保存在 `local/upstream/lazy-h2-import.patch`。

本次正式训练启动时主仓库代码为 `74d2c21`；之后的迁移记录和锁定清单整理不改变该训练的计算代码。源机两个子模块的历史对象已补齐，mimic-lite 的修复版 bundle 已验证可独立克隆。

## 正式训练

- GPU：物理卡 4、5、6、7。
- 每卡 8192 环境，总共 32768；rollout 32，PPO 3 epoch、8 minibatch。
- 从 1024 条动作的最终 checkpoint 热启动，新增 2500 iteration。
- Actor 不带显式 XY 反馈，隐藏层 512/512/512；Critic 1024/512/256。
- 根位置失败作为 termination：`root_pos_error.is_timeout=false`。
- 熵系数前 1875 轮保持 0.01，后 625 轮降至 0.002。考虑恢复的 `last_iter=5929`，全局衰减区间为 7803–8428。

正式目录：`/data/langlang/runs/mimiclite/full-data-2500-20261018T162211Z`。

W&B：[4hev73p2](https://wandb.ai/langlang-02-ustc/mimic_lite/runs/4hev73p2)。本地同时保留 `metrics.jsonl`、TensorBoard 和 `console.log`。

```bash
ssh -t digua_4090_48 'TMPDIR=/tmp /snap/bin/zellij attach mimiclite-migration-validation'
```

训练 pane 名称为 `full-data-training`，ID 为 `terminal_0`。脚本串联缓存构建、10 轮验证和正式训练；Zellij 独立于聊天连接运行。

机器配置为 `local/mjlab/.env.local`。W&B 通过该环境指定的独立 NETRC 使用 langlang-02 账号；凭据不进入 Git 或 BOS。

## 已完成的验证

- 数据完整性及全量 159777 条动作 FK 缓存构建通过。
- 动作索引测试和 PPO 诊断一致性测试通过。
- 四卡 10 轮验证累计 10485760 帧；41 个初始 Actor/Critic/归一化张量与源 checkpoint 完全一致，15 个 Actor 张量训练后更新，所有记录指标有限。
- 正式训练再次验证热启动张量、完整索引、无 XY 输入、终止条件及熵调度；启动核验时已完成 58 轮，W&B 云端已收到指标。近期每轮约 3.2 秒，GPU 显存约 33 GiB/卡，这是启动阶段测量。

检查报告：`transfer/rvq-dataset-20261006/verification.json`、`transfer/source-data-verification.json`、`transfer/environment-details.json`、`transfer/validation/result.json`、`transfer/training-start-verification.json`。正式目录另有 `warmstart-verification.json` 和实际 PPO 参数副本。
