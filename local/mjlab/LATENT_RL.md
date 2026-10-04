# G1 latent RL：原理、数据与平台交付

## 当前训练结构

Actor 使用 `policy`（本体观测）和 `command`（512D RVQ embedding + 1D phase）。
Critic 使用 `priv + policy + command`。动作输出仍是原来的关节位置控制接口。
PPO 算法没有替换；这次修改的是动作条件、参考数据适配和时间同步。

原始 G1 动作有两条路径：

1. 原始 feature bank 的关节角、倾斜角和逐帧 XYZ/yaw anchor 生成
   `reference_native38`；转换器把 root 6D 旋转换成 wxyz 四元数，得到
   `[xyz(3), quaternion(4), joints(29)]`，供 any4hdmi 重采样和 FK 使用。
   此路径不调用 RVQ decoder。奖励、初始化与终止条件使用原始 reference。
2. 冻结 RVQ encoder/quantizer 生成 6 层 token，以及 6 层码本向量求和的
   512D latent。运行时按样本 ID 和时间读取 latent，RVQ 不参与 PPO 反向传播。

PPO 任务不需要加载 RVQ checkpoint；挂载转换后的 reference/latent 即可。
原 checkpoint、归一化和源 shard 保留用于追溯与重新预处理。

reference 为 20 Hz，latent 为 5 Hz，控制为 50 Hz。
令 `s` 为原始动作的 50 Hz step，则 `frame=floor(s*20/50)`，
`token=frame//4`，`phase=(frame%4)/4`。phase 是离散的 0、0.25、0.5、0.75。
windowed 数据集的 motion_id 实际是环境路由，必须先取原始 motion id，
再加 source_start_t；窗口尾部对 step 截断，避免 latent 越过 reference。
reset 刷新新动作的参考和 latent，同时保留其他环境的 observation 时间锚点。

上游环境 reset 返回零 observation；首个控制步沿用这一既有行为，之后返回正常
的本体与 latent 观测。reset 时内部 reference/latent 已刷新，用于首步奖励。
RVQ encoder 使用完整窗口的非因果上下文；这条管线验证离线动作条件跟踪，
不证明实时因果编码或策略已学会跟踪。

## 代码入口

MoMask 工程：`/DRoboticsAILab/zhaohui.yang/momask-codes`。
RVQ 预处理：`dataset/bones-seed_prepare/g1_preprocess_rvq.py`。

MimicLite 工程：`/DRoboticsAILab/zhaohui.yang/MimicLite`。

- `mimic-lite/scripts/convert_rvq_tracking_dataset.py`：qpos 与 latent mmap 导出。
- `mimic-lite/mimic_lite/tasks/rvq_latent.py`：样本映射、token/phase 和按需读取。
- `mimic-lite/mimic_lite/tasks/command.py`：窗口映射、reset 与 latent 刷新。
- `mimic-lite/mimic_lite/tasks/observations/track.py`：Actor command 观测。
- `mimic-lite/mimic_lite/tasks/motion.py`：windowed 模式保留 mmap，避免全量
  `index_select` 拷贝占用数十 GiB RAM。代价是每个 GPU 窗口保留全部身体字段，
  并行环境数应根据实测显存指定。
- `local/mjlab/validate_rvq_data.py`：全部样本与源 shard 的独立数值比对。
- `local/mjlab/verify_latent_runtime.py`：重载已有 smoke checkpoint 并推理，零 PPO 更新。

## 数据目录

下列相对路径均位于 `/EFS/users/zhaohui.yang/dataset/`：

| 路径 | 用途 |
|---|---|
| `bones-seed-g1/` | 原始数据根目录，来自 feature bank 的 dataset_info.json |
| `BONES-G1-feature-bank-v1/` | 20 Hz 特征、逐帧 anchors、样本 metadata 和 split |
| `momask-checkpoints/bones_g1/bones_g1_rvq_rot6d68_v1_nq6/model/frozen_latest_ep0499_it8998000.tar` | 指定的冻结 RVQ checkpoint |
| `momask-rvq-tracking/rot6d68_ep499_v1/` | RVQ 导出源：93 个 shard、码本、归一化与来源清单 |
| `mimiclite-rvq-tracking/rot6d68_ep499_v1/` | MimicLite 完整 reference + latent |
| `mimiclite-rvq-tracking/rot6d68_ep499_smoke/` | 96 条小规模验证数据 |

完整数据共 187,777 个窗口；train 168,895，val 9,294，test 9,588。
reference 共 15,925,244 帧，latent 共 4,044,952 行。

### 平地描述筛选（2026-10-03）

原始 `train.txt/val.txt/test.txt` 保持不变。正式平台入口现使用同一数据根目录下
`flat_ground_v1/train.txt`，验证/测试使用对应的 `val.txt/test.txt`。
保留原 split 归属及顺序，不改变 motions 或 latent。

| split | 原始窗口 | 排除 | 保留 |
|---|---:|---:|---:|
| train | 168895 | 9118 | 159777 |
| val | 9294 | 540 | 8754 |
| test | 9588 | 588 | 9000 |

规则使用原始 `seed_metadata_v004.csv` 的自然描述 1–3、技术/短描述、动作名称及
源文件中的描述性名称（发现 20 个下箱子窗口的描述被误标为舞蹈），
识别上下箱子、梯子/楼梯、坐椅子、墙体支撑、拐杖等；明确命中时排除该来源的
所有窗口，含混合动作中的普通片段。box-step 舞步、平地跳跃不因名字带 box/jump
而被排除。第四条自然描述存在与动作名称冲突的实例，仅由该字段命中的 338 个
窗口保留并列入 `flat_ground_v1/review.jsonl`，待人工复核。
这只是描述筛选，不证明保留动作全部物理可跟踪。

`flat_ground_v1/excluded.jsonl` 记录样本、来源、规则、命中文本；`report.json`
记录规则、计数及输入/输出 SHA256。验证结果在
`local/mjlab/records/flat-ground-filter-validation.json`。
筛选程序和针对误匹配的测试在 MoMask 工程的
`dataset/bones-seed_prepare/g1_filter_flat_ground.py` 与
`test_g1_filter_flat_ground.py`。

复现到新的输出目录（程序拒绝覆盖已有审核记录）：

```bash
python /DRoboticsAILab/zhaohui.yang/momask-codes/dataset/bones-seed_prepare/g1_filter_flat_ground.py \
  --metadata /EFS/users/zhaohui.yang/dataset/bones-seed-g1/metadata/seed_metadata_v004.csv \
  --feature-bank /EFS/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1 \
  --dataset /EFS/users/zhaohui.yang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1 \
  --output /EFS/users/zhaohui.yang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1/flat_ground_v1_recheck
```

原完整训练集的 FK 缓存仍保留。筛选列表改变缓存指纹，首次使用筛选列表时当前
加载器会构建新的子集缓存；本次未启动该构建或训练，也无需重新运行 RVQ。

MimicLite 数据目录内部：

- `motions/<sample_id>.npz`：`qpos[T,36]`，20 Hz。
- `latent/latent.npy`：float32 `[4044952,512]`，按需 mmap，不全部搬到 GPU。
- `latent/sample_ids.npy`、`latent/token_offsets.npy`：样本和 latent 切片索引。
- `train.txt`、`val.txt`、`test.txt`：相对 motion 文件路径；训练只读 train。
- `manifest.json`：机器人模型、关节顺序和参考采样率。
- `latent/manifest.json`：latent 格式和 RVQ checkpoint SHA256。
- `conversion.json`：转换完成记录；它本身不替代完整性验证报告。

本机小规模 FK cache 在 `active-adaptation/.cache/motion/qpos_online_v2/`，属于
可重建数据。全量验证曾触发 VePFS 目录配额超限（errno 122），不是整个文件系统
容量用尽；失败临时缓存已清理。完整缓存改用
`/EFS/users/zhaohui.yang/dataset/mimiclite-fk-cache/.cache/motion/`。
`MIMICLITE_MOTION_CACHE_BASE` 指定缓存基目录，实际缓存在其 `.cache/motion/` 下。
全量缓存需要数十至上百 GiB，使用有足够配额的共享盘，不放系统盘。
windowed_cache.py 沿用 any4hdmi 的互斥锁，直接构建到目标目录、最后发布 READY，
避免 EFS 目录 rename 的 EXDEV 错误。windowed 的 FK 缓存使用顺序文件写入，
保留上游插值和 FK 算法，每批计算后立即写盘；避免 mmap 写入的高延迟和扩容。
`ANY4HDMI_CACHE_BUILD_BATCH_SIZE=2048` 控制每批目标帧数（单条长动作可超过此数）。
与原写入器的小数据集逐字段比较记录在 `records/latent-cache-equivalence.json`。
设置此变量时 motion dataset 使用绝对路径。

## 验证与报告

截至 2026-10-03，以下本地验证已通过：

- 全部 187,777 个窗口、15,925,244 帧 reference 和 4,044,952 行 latent
  与源导出逐项比对通过；latent 完全一致，旋转矩阵最大误差约 1.03e-7。
- 168,895 条 train 动作的完整 FK 缓存已发布 ready.flag，并成功 mmap 加载。
- 使用已有 smoke checkpoint，在 4 个 mjlab 环境运行 128 步推理；权重重载、
  有限数检查、每步 latent/phase 对齐和 reset 检查通过。本轮 PPO 更新为 0。
- 8 项单元测试通过；顺序缓存与上游缓存的 8 个字段比较通过，身体线速度
  最大差异 5.22e-8，其余字段完全一致；174 个依赖包一致性检查通过。

这验证数据与运行链路，不证明正式训练收敛或跟踪质量。镜像尚未构建、推送，
平台任务尚未提交；正式 PPO 参数仍待用户指定。

从工程根目录运行：

```bash
source local/mjlab/env.sh
python local/mjlab/validate_rvq_data.py \
  --source /EFS/users/zhaohui.yang/dataset/momask-rvq-tracking/rot6d68_ep499_v1 \
  --dataset /EFS/users/zhaohui.yang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1 \
  --report local/mjlab/records/latent-data-validation.json
```

`records/latent-runtime-smoke.json` 与 `latent-runtime-full.json` 分别记录
小数据集和全量 train 加载后的 checkpoint 重载、逐步 latent 对齐和推理。
`records/latent-unit-tests.log` 记录 token 边界、尾部、窗口偏移、partial reset
及旋转转换测试。成功报告只在所有断言通过后写出，检查日期和日志确认对应本次运行。

完整训练集的推理验证（使用已有 smoke 权重，不训练）：

```bash
source local/mjlab/env.sh
export MIMICLITE_MOTION_CACHE_BASE=/EFS/users/zhaohui.yang/dataset/mimiclite-fk-cache
export ANY4HDMI_CACHE_BUILD_BATCH_SIZE=2048
TORCH_COMPILE_DISABLE=1 python local/mjlab/verify_latent_runtime.py \
  --dataset /EFS/users/zhaohui.yang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1 \
  --checkpoint local/mjlab/records/latent-windowed-train/checkpoint_latest.pt \
  --report local/mjlab/records/latent-runtime-full.json
```

本轮验证不启动 PPO，不修改已有策略权重。旧 smoke checkpoint 的两次 PPO 更新
只用于验证 checkpoint 权重确实变化过；正式参数由用户另行指定。

## 交到火山引擎机器学习平台

### 4 卡 A800 起始配置

`local/mjlab/ppo_4xa800_defaults.txt` 固定当前 train.yaml/PPOConfig 默认参数，
仅将每进程环境数设为 2048。单机 4 卡设置 `MIMICLITE_NPROC_PER_NODE=4`，
`PPO_OVERRIDES_FILE=/opt/mimiclite/local/mjlab/ppo_4xa800_defaults.txt`。
入口通过 torch.distributed.run 启动 4 个进程，每卡 2048、合计 8192 个环境；
框架已有 NCCL 梯度同步及 rank seed 偏移。此配置已通过 Hydra 解析和 shell
语法检查，尚未在 A800 或 4 卡上实测，2048 只是容量/吞吐测量的起点。

默认 rollout=32、epochs=3、minibatches=8、lr=3e-4（KL 自适应）、Muon，
Actor=[512,512,512]、Critic=[1024,512,256]，entropy=0.004。
每轮全局采样 262144 步，每卡 minibatch=8192 条；默认总帧数 150000000
在 4 卡下向下取整为 572 轮，不保证收敛。保存间隔为 500 轮，另有最终保存。
正式多卡启动前先单进程构建筛选列表 FK 缓存，再测试多卡加载、同步及显存。
不要让其他 rank 在首次长时间构建时进入 NCCL 集合通信等待。

### 推荐：构建固定镜像并挂载数据

本机 `.venv` 和 Python 位于 VePFS 项目目录中。直接“保存开发机镜像”不能假定
共享挂载里的环境也被包含，而且 editable 安装有绝对路径。
使用 `local/mjlab/Dockerfile` 在 `/opt/mimiclite` 重建 Python 3.12.13 和
requirements.lock，复制本次代码（含尚未提交的改动），并预取固定版本的 G1/Atom
机器人资源。不要只 git clone 原始 commit，否则会遗漏 latent 改动。

```bash
# 在工程根目录；替换为你有权限推送的 CR 仓库和唯一版本标签。
docker build -f local/mjlab/Dockerfile -t YOUR_CR/mimiclite:YOUR_VERSION .
docker push YOUR_CR/mimiclite:YOUR_VERSION
```

这些是交付命令，本轮未执行镜像构建、push 或任务提交。
镜像中的依赖固定为现有锁文件版本：torch 2.10.0、mjlab 1.3.0、MuJoCo 3.7.0、
Warp 1.12.0；any4hdmi 固定到 bc83d6c。原 MoMask 的 Python 3.11 环境不能直接替代。
Dockerfile 默认命令只检查导入，不训练。

在平台“模型训练 → 自定义训练”中：

1. 选择上述自定义镜像的固定版本或 digest，框架选择 Custom。
2. GPU 类型、数量、CPU/RAM 和 PPO 参数由用户确定；本轮验证范围为单 GPU。
3. 将现有 EFS 挂到 `/EFS`，使 dataset 绝对路径保持一致。
4. 将 VePFS 的 `zhaohui.yang/MimicLite` 子目录挂到 `/workspace`，保留写权限。
   训练代码仍从镜像内 `/opt/mimiclite` 使用，此挂载仅供缓存、配置和输出。
5. 设置 `RVQ_DATA_ROOT=/EFS/users/zhaohui.yang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1`，
   `RL_RUN_DIR=/workspace/runs/<unique_run>`，
   `MIMICLITE_MOTION_CACHE_BASE=/EFS/users/zhaohui.yang/dataset/mimiclite-fk-cache`，
   `PPO_OVERRIDES_FILE=/workspace/configs/<approved_params>.txt`。
6. Command 使用 `bash /opt/mimiclite/local/mjlab/platform_entrypoint.sh`。
   若数据路径、mtime、manifest 和机器人 XML 内容一致，这一挂载可复用本机
   已构建的同一动作列表的 FK cache；不一致或改用筛选列表时 any4hdmi 会生成新缓存。

已有 MoMask 任务配置使用的存储 ID 为 VePFS `vepfs-cnbjf7015ef11e9c` 和
EFS `efs-cnbjc8a589e0f6d6f`，可在控制台核对后选择同一存储；挂载路径按上文设置。

PPO 参数文件每行一个 Hydra override，不加 shell 引号；不要填写未经确定的参数。
需确定的参数包括并行环境数、rollout 长度、PPO epochs、minibatches、学习率、
网络维度、总迭代数和 checkpoint 间隔。启动脚本在缺少参数文件时立即退出。
W&B 使用 disabled 模式，临时目录和 checkpoint 均放在持久 run 目录下。

当前 CLI 已确认支持 `mlp job create -f <yaml>`；从控制台导出最终配置再提交。
队列、资源规格、镜像 URL 不复用旧 MoMask 任务的选择。任务环境还需验证 GPU
驱动兼容、挂载和 EGL；本机通过不等于远端任务环境已验证。

当前上游 checkpoint 不保存 optimizer 状态，重载是权重恢复/warm start，不能声称
严格恢复中断训练。正式长跑如需 optimizer 续训，应在启动前补齐并单独验证。

官方说明：
- https://docs.volcengine.com/docs/ml-platform/build-custom-container-image?lang=zh
- https://www.volcengine.com/docs/6459/72350?lang=zh
