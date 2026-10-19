# Codex 接手说明：G1 RVQ latent tracker

更新时间：2026-10-19。本文件描述已核验事实；GPU 占用、最新运行和 Git 状态在执行前重新检查。目标是让 digua_4090_48 上的新 Codex 独立开展评测、分析和后续训练。

## 交接同步时发现的目录变化（优先处理）

2026-10-19 最终同步时，目标机主机名仍为 `vm-38gu4g7c-cn-zhangjiakou-ix-p6g5z`，但 `/data/langlang/Project/MimicLite` 只剩 `.cache`，`.git`、代码及原目录内 Python 环境已不在该路径。`git pull` 因而失败。数据目录和本次训练结果目录仍存在。原因未知，可能有其他会话在整理目录，不能假定是数据丢失或擅自覆盖。

下面的仓库及环境路径是训练时已经验证的位置，目前执行命令前必须先定位代码与环境，或与用户确认目录调整情况。不要直接运行下面的 source/train 命令。主仓库远端已经有本说明，可从 `feat/rvq-latent-tracker` 获取；两个子模块远端也已推送。若需要恢复，应先确认新位置并保留现有 `.cache`，避免覆盖其他项目。独立交接副本放在 `/data/langlang/CODEX_HANDOFF.md`，无需仓库即可阅读。

## 1. 项目目的与当前结论

基于 MimicLite，为 Unitree G1 训练以冻结 MoMask/RVQ 动作 latent 为条件的 PPO tracker。仿真后端是 mjlab / MuJoCo Warp，不需要启动 Isaac Sim。

研究路线已从 12 条动作调试，扩展到 128、1024 条，再到全部平地过滤后的训练动作。做过 Actor 显式 XY 位置反馈实验，但用户明确表示它只是对照实验；后续主线仍不带 XY 反馈。不要默认扩大网络或改变输入。

最近全量训练已完成，不是仍在运行：W&B `4hev73p2` 状态 `finished`，2500/2500 iteration、2621440000 环境帧，训练循环约 2 小时 12 分钟，最终 checkpoint 保存并同步成功。2500 行指标没有非有限值。

不能声称已经充分收敛或泛化良好。最后两个 250 轮窗口（2001–2250 → 2251–2500）均值：

| 指标 | 前窗口 | 后窗口 |
| --- | ---: | ---: |
| tracking reward 各项之和 | 2.990238 | 3.239373 |
| reward.tracking_metrics/root_pos | 0.163478 | 0.150182 |
| reward.tracking_metrics/root_ori | 0.377406 | 0.348753 |
| reward.tracking_metrics/body_pos | 0.071628 | 0.065604 |
| reward.tracking_metrics/body_ori | 0.354095 | 0.328231 |
| critic/tracking.explained_var | 0.961532 | 0.967943 |

末期仍改善，同时 entropy 在下降，不能把曲线改善完全归因于确定性策略能力提升。训练环境统计主要来自 rank 0，终止、采样和探索噪声都会影响均值。最新全量 checkpoint 尚未在本轮交接工作中做固定动作评测。

## 2. 目标机环境与目录

目标机 `digua_4090_48`，使用 root；仓库 `/data/langlang/Project/MimicLite`。

| 内容 | 位置 |
| --- | --- |
| Python 环境 | 仓库 `local/mjlab/.venv` |
| 数据根 | `/data/langlang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1` |
| 全量 FK 缓存 | `/data/langlang/cache/mimiclite-fk-cache` |
| 所有运行 | `/data/langlang/runs/mimiclite` |
| 本次全量运行 | `/data/langlang/runs/mimiclite/full-data-2500-20261018T162211Z` |
| 本次最终模型 | 上述目录 `checkpoint_latest.pt`，链接到 `checkpoint_2500.pt` |
| 上次 1024 条基线模型 | `/data/langlang/checkpoints/mimiclite/train1024-entropy-2500.pt` |
| 历史实验归档 | `/data/langlang/archives/a800-experiments` |
| 迁移验证报告 | `/data/langlang/transfer` |

8 张 RTX 4090，每张约 48 GiB；当前全量实验使用物理卡 4–7，每卡峰值观测约 33 GiB。0–3 卡曾有其他任务，任何新任务都先 `nvidia-smi` 检查。用户曾授权停止旧的整个 8 卡 DDP 任务，已经执行；这不授权停止此后出现的新任务。

环境已安装并验证：Python 3.12.13、torch 2.10.0+cu128、MuJoCo 3.7.0、mujoco-warp 3.7.0.1、mjlab 1.3.0、Warp 1.12.0。不用重新安装或升级。

```bash
cd /data/langlang/Project/MimicLite
source local/mjlab/env.sh
source local/mjlab/.env.local
which python
nvidia-smi
```

`.env.local` 已配置数据、缓存、运行目录及 W&B 的独立 NETRC 路径。凭据文件受限访问，禁止输出、提交或上传其内容。`env.sh` 默认 W&B disabled，正式训练必须在 profile 和环境中明确 online。根分区较小，环境、下载、缓存及输出都放 `/data/langlang`。

## 3. 数据与 latent 表征

完整导出包含 187777 个动作窗口（不是 187792 个动作；后者是迁移验证文件数），train 168895 / val 9294 / test 9588。全量训练使用平地筛选索引 `flat_ground_v1/train.txt` 的 **159777 条**，并非未过滤的全部 train。平地筛选细节见 `../LATENT_RL.md`；不要把规则筛选当作每个动作物理可行性的保证。

当前表征 `g1_local_rot6d68_v1`，20 Hz、68 维：29 个关节角、29 个逐帧关节差分、6D 根 roll/pitch 旋转、yaw 增量、当前 heading 坐标系下二维 XY 增量、根高度。没有绝对 XY 或绝对 yaw。按导出 mean/std 归一化后进行 RVQ；六级 512D 残差码本 embedding 求和，token 5 Hz。Actor command 为 512D latent + 1D phase。

Reference 独立保存绝对 XYZ、wxyz 四元数、29 个关节角。reference 20 Hz、latent 5 Hz、控制 50 Hz。latent 运行时 mmap 按动作及时间读取，不参与 PPO 反向传播；训练 tracker 不需要重新加载 MoMask 编码器。离线编码不应未经验证就宣称因果，研究未来信息依赖应另查编码器实现。

固定索引位于 `local/mjlab/motion_sets/`，每行是相对数据根的动作路径：

- `train_small_v1.txt`：12 条，T01–T12。
- `train128_v1/train_128_v1.txt`：128 条。
- `train1024_v1/train_1024_v1.txt`：1024 条。
- `eval_fixed_v1.txt`：20 条固定验证动作，E01–E20，使用 `--split val`。
- 各目录的 `annotations.csv` / selection / README 提供来源和筛选依据。T/E 编号只是展示标签，样本路径/sample_id 才是稳定身份。

不要把评测索引加入训练。12 条与固定评测集做过来源组隔离，但不能推断整个历史 train/val 都来源无泄漏。迁移逐文件 SHA256 校验通过，FK 缓存在目标机重建成功，无需重新搬数据。

## 4. 策略、PPO 和最新训练配置

默认 Actor 输入 `policy` 535D + `command` 513D = 1048D，隐藏层 512/512/512。Critic 输入 `priv + policy + command` 合计 1598D，隐藏层 1024/512/256。`priv` 是特权观测，不能因为 `algo.in_keys` 列出 priv 就误认为 Actor 也使用它。具体观测项、历史栈及噪声以 checkpoint 保存的 cfg 和任务 observation YAML 为准。

可选 XY 实验增加 `actor_feedback` 两维，Actor 1050D，Critic 不变；接口见 `actor-xy-feedback-ablation.md`。旧 XY 实验与某些无 XY 实验初始化方式不同，不能当严格单因素结论。

本次配置：4 卡 × 每卡 8192 环境；rollout 32；每次 iteration 总采集 1048576 帧；PPO 3 epoch、8 minibatch；四卡各自采样和计算梯度，通过项目手动梯度 all-reduce 同步。默认没有启用 torch compile。

从 1024 条模型热启动：恢复 Actor、Critic、VecNorm 和 last_iter；优化器重新初始化，不是完整优化器状态续训。新增 2500 轮。entropy 前 1875 轮保持 0.01，最后 625 轮降到 0.002；生成器根据恢复的 last_iter 设置全局调度区间，不能直接抄旧区间再用于新 checkpoint。

根位置失败 `root_pos_error.is_timeout=false`：作为真正 termination，不做截断 bootstrap，并可进入只针对 terminated 的失败回退采样。它不是移除了 root_pos_error 终止条件。评测关闭终止仅用于诊断，不等于训练也该关闭。

实际配置优先级：本次运行 `.hydra/config.yaml`、`ppo-overrides.txt`、checkpoint 的 cfg；勿把名字带 a800 的历史模板当本次最终参数。

## 5. W&B、日志和持久会话

W&B：https://wandb.ai/langlang-02-ustc/mimic_lite/runs/4hev73p2

本次目录保存 `metrics.jsonl`、`tensorboard/`、`console.log`、实际参数和 `warmstart-verification.json`。W&B `_step=2499` 对应本地 iteration 2500，不是少训一轮。

```bash
TMPDIR=/tmp /snap/bin/zellij attach mimiclite-migration-validation
```

既有会话的 pane 为 `full-data-training`（terminal_0）；任务已结束，会话可能仍保留 shell。新长任务建议新建有语义的 session，日志落盘，脚本串联必要阶段，不用 Codex 一直轮询。远程进入时可用 `ssh -t digua_4090_48 'TMPDIR=/tmp /snap/bin/zellij attach mimiclite-migration-validation'`。

## 6. 下一步建议：先固定评测，再决定续训

先比较新全量模型与 1024 条基线：固定 20 条 val + 12 条训练诊断动作。12 条不能代表 159777 条训练分布，需要时再从全量索引冻结一个有来源/类别覆盖的训练诊断子集。无需直接评测全部 1024 或 159777 条。

默认终止条件评测优先；对失败动作再用相同设置关闭 root/body_ori 终止观察漂移。对比必须保持动作、seed、终止条件一致。汇报完整动作完成率、完成比例、失败原因、根漂移/身体误差和视频，不只报提前失败之前的平均误差。

以下命令需先确认 GPU 4 空闲，并在持久会话中执行。交接文档本身不代表已启动这些任务。

```bash
cd /data/langlang/Project/MimicLite
source local/mjlab/env.sh
source local/mjlab/.env.local
ckpt=/data/langlang/runs/mimiclite/full-data-2500-20261018T162211Z/checkpoint_latest.pt
out=/data/langlang/runs/mimiclite/eval-full-fixed-$(date -u +%Y%m%dT%H%M%SZ)
CUDA_VISIBLE_DEVICES=4 TORCH_COMPILE_DISABLE=1 python local/mjlab/evaluate_latent_motion.py \
  --dataset "$RVQ_DATA_ROOT" --checkpoint "$ckpt" \
  --index-file local/mjlab/motion_sets/eval_fixed_v1.txt --split val --output "$out"
```

训练集 12 条用 `train_small_v1.txt` 和 `--split train`。基线评测把 checkpoint 换成 `/data/langlang/checkpoints/mimiclite/train1024-entropy-2500.pt`，使用新的 output。单动作可用 `--motion 相对路径` 代替 `--index-file`。

宽松诊断追加 `--disable-root-terminations --disable-body-ori-termination`，输出目录另建并注明 relaxed。此时完成率不是严格成功率。

评测脚本自动读取 checkpoint cfg，禁用 domain randomization，确定性动作、从动作起点开始、不自动 reset；逐动作进程顺序执行。输出：`batch-summary.json`、各动作 `summary.json`、`metrics.csv`、`diagnostics.csv`、`trajectory.npz`、`reference_vs_policy.mp4`。索引旁的 annotations 必须完整覆盖选中动作；新子集应提供匹配 CSV 或放到不含旧 annotations 的目录。输出目录要求不存在或为空；中断不会自动续跑，保留已完成结果后另建剩余动作索引。

## 7. 决定继续训练时的入口

评测后再选择预算；下例只是生成新 profile 的模板，不是推荐立即再训 2500 轮。该生成器会重新开启 0.01→0.002 调度，若要低探索微调应明确修改新 profile 并记录原因。

```bash
cd /data/langlang/Project/MimicLite
source local/mjlab/env.sh
source local/mjlab/.env.local
profile=/data/langlang/transfer/continue-full-$(date -u +%Y%m%dT%H%M%SZ).txt
python local/mjlab/migration/full_training.py \
  --checkpoint /data/langlang/runs/mimiclite/full-data-2500-20261018T162211Z/checkpoint_latest.pt \
  --output "$profile" --workers 4 --num-envs 8192 --iterations 2500 --wandb-mode online
# 检查 profile 和 GPU 占用后，在 Zellij 中运行：
CUDA_VISIBLE_DEVICES=4,5,6,7 WANDB_MODE=online PPO_OVERRIDES_FILE="$profile" \
  bash local/mjlab/train_local.sh
```

`train_local.sh` 根据显式 GPU 列表设置 worker 数、创建新运行目录。不要直接使用默认历史 profile。`migration/validate_target.sh` 是迁移验证链，会做 smoke 并可串联正式训练；普通后续实验不必反复跑整个迁移流程。

## 8. 代码地图与 Git

- `mimic-lite/scripts/train.py`：训练与 checkpoint。
- `mimic-lite/mimic_lite_learning/`：PPO、Actor/Critic、指标。
- `mimic-lite/mimic_lite/tasks/command.py`、`rvq_latent.py`：reference/latent 时间对齐及 reset。
- `mimic-lite/mimic_lite/tasks/windowed_cache.py`：窗口缓存。
- `local/mjlab/evaluate_latent_motion.py`、`tracking_diagnostics.py`：评测与误差诊断。
- `local/mjlab/motion_sets/`、`motion_index.py`：固定动作索引。
- `active-adaptation/`：环境框架；其中 experiments 历史脚本可能有源服务器绝对路径，应视为出处记录，不直接运行。

| 仓库 | 远端 | 分支 |
| --- | --- | --- |
| 主仓库 | https://github.com/langlang-02/MimicLite | feat/rvq-latent-tracker |
| mimic-lite | https://github.com/langlang-02/mimic-lite | feat/rvq-latent-tracker |
| active-adaptation | https://github.com/langlang-02/active-adaptation | feat/mimiclite-latent-experiments |

全量运行时代码：主仓库 `74d2c21`，mimic-lite `d6513bd`，active-adaptation `ab45406`。之后的文档提交不改变本次运行代码。当前 HEAD 用 git 查询，不把这里的运行版本当最新 HEAD。子模块改动先在子模块提交/push，再更新主仓库 gitlink。sim2real 非当前训练必需，不需要为评测初始化；可选补丁在 `local/upstream/`。

源机已使用 langlang 的 gh 凭据 push；目标机 GitHub 写入认证是否存在需另查，不复制或打印 token。数据迁移使用 bcecmd/BOS：`bos:/langlang/mimiclite/migration-20261006/` 是数据分包；`migration-20261018/` 是模型、历史记录、资产和 Git 备份。不要重复下载已验证的数据。

## 9. 进一步阅读

- `migration-digua-4090.md`：迁移与启动阶段报告（其中“已完成 58 轮”是历史启动快照，当前已完成 2500 轮）。
- `wandb-metrics-and-distributed-normalization.md`：指标含义、rank 0 范围和输入归一化。
- `actor-xy-feedback-ablation.md`：XY 接口与对照实验限制。
- `../LATENT_RL.md`：数据表征、转换、缓存及平台交付细节，旧绝对路径仅为历史记录。
- `../motion_sets/README.txt`：索引规则和评测语义。

给新 Codex 的第一条任务可以是：先读 AGENTS.md 和本说明，核对 GPU 与文件状态，用最终全量模型和 1024 条基线在相同固定动作、相同默认终止条件下评测，比较失败原因和完成率；暂不启动新训练。
