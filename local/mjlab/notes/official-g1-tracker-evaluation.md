# 官方 G1 tracker 发布版与固定集评测（2026-10-14）

主仓库没有 GitHub Releases 条目；“发布版”通过 README 中的 Drive / W&B 链接提供。当前主仓库 G1 默认指向 MimicLite v1.1 ROA student。

权威来源：

- https://github.com/Roboparty/MimicLite
- https://github.com/EGalahad/mimic-lite/blob/feat/multi-robot-support/README.md
- https://egalahad.github.io/sim2real/leaderboard/

当前实现仓库列出的 v1.1 Actor 是 512×3，PPO/ROA Huge 是 1024×3。旧版中文 README 与最新模型表不一致时，以当前实现仓库和实际权重为准。

官方榜单默认 Locomotion/Manipulation 汇总中，v1.1 的局部身体位置误差最低（26.68 mm）；PPO Huge 的全局根误差最低（0.255 m，根误差仅统计 Locomotion）。v1.1 并非在所有指标上最好。官方榜单的动作集与本项目固定 20 条 BONES-SEED 评测集不同，数值不能直接横向混用。

## 下载与运行

Google Drive 的 v1.1 下载遇到网络阻塞（本地/volc-test 直连超时、本地代理 EOF）。实际下载并测试的替代版本是官方 PPO Huge：

- W&B：`elijahgalahad/mimic_lite/4234dd57`
- 文件：`checkpoint_4000.pt`
- 本地：`local/upstream/checkpoints/mimic-lite/ppo/`
- Actor：839 → 1024 → 1024 → 1024 → 29
- 输入：535 维本体观测 + 304 维显式参考动作；后者为 8 帧 ×（根局部位置 3 + 根相对朝向 6 + 关节位置 29）。参考帧偏移为 `[-8,-4,-2,0,1,2,3,4]`，50 Hz 下未来跨度 80 ms。
- 未使用 RVQ latent，未重新训练官方模型。

实验目录：`active-adaptation/projects/hdmi/scripts/experiments/20261014T151354Z-official-g1-eval`。
结果目录：`/mnt_scalelab/langlang/home/runs/mimiclite/eval-20261014T151354Z-official-g1-eval`。
Zellij：`mimiclite-official-eval-20261014`。

## 验证与判定

原始 checkpoint 不修改。加载时仅兼容旧接口：去掉训练专用配置字段、拆分新版本 Critic 的合并观测项、采用既有奖励与终止配置。Actor、Critic、VecNorm 的所有保存张量逐一验证完全一致。ONNX 导出及 ORT 输出精度验证通过。

实际跟踪沿用 native mjlab 环境，并在进程内将设备设为 CPU；视频采用 OSMesa 软件渲染。GPU 不分配给本评测。旧训练代码和训练进程不修改。

固定 20 条动作、seed 42、参考第 0 帧起步、确定性 Actor、禁用 domain randomization 和失败回退、不自动 reset；保留既有评测的零首帧观测行为。与旧评测一样，观测组件中配置的传感器噪声仍保留；“确定性”指采用动作分布均值，不表示传感器观测无噪声。官方模型额外保留其 checkpoint 的参考观测噪声设置。

严格失败判据：

- torso 全局位置误差 ≥ 0.4 m，连续 25 步；
- torso 全局朝向误差 ≥ 1.2 rad，连续 25 步；
- 任一跟踪 link 的局部位置误差 ≥ 0.4 m，连续 5 步；
- 任一跟踪 link 的局部朝向误差 ≥ 1.2 rad，连续 5 步。

成功要求动作结束且没有上述失败，50 Hz 控制。机器人、动作控制参数、本体观测及终止配置与上次 latent 评测核对一致。物理步长均为 0.005 s，控制步长 0.02 s。

## 解释限制

CPU/GPU 物理数值和随机数不同；官方与本地策略的输入、网络、训练数据和训练预算不同，因此不是单因素实验。各片段误差仅统计实际运行到失败或结束的部分，不能单靠失败前平均误差判断长程质量。官方模型与固定评测片段的训练数据重合情况尚未确认，不能宣称这是官方模型的严格留出泛化测试。该结果属于 PPO Huge，不能标成 v1.1 的结果。

CPU/GPU 健全性检查：旧 latent checkpoint 的 E01 在 CPU 第 19 步因 body_ori_error 终止，原 GPU 结果为第 20 步、同一原因；官方 PPO Huge 则执行 158 步完成全段。该检查只覆盖一条动作。

## 已完成结果

20 条全部完成，20 个视频解码验证通过。官方 PPO Huge 成功 17/20（85%），平均动作完成率 93.42%；原 latent 策略为 2/20（10%），40.29%。

相同时段的逐片段平均误差再平均：根位置 18.16→5.50 cm；局部身体位置 5.30→2.24 cm；关节 MAE 0.1425→0.0501 rad。

失败项：
- E07 `motions/walk_sideway_left_loop_004__A029_seg001_win000.npz`：完成 78.1%，root_pos_error。
- E08 `motions/walk_sideway_left_loop_003__A023_seg000_win000.npz`：完成 56.1%，root_pos_error。
- E19 `motions/dance_jazz_hands_002__A467_seg000_win000.npz`：完成 34.2%，body_pos_error,body_ori_error。
