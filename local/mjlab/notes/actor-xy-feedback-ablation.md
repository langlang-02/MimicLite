# Actor XY 显式反馈对照实验

本实验只给 Actor 增加当前参考位置的水平误差；默认训练配置仍使用不带 XY 反馈的 latent 策略。不迁移旧 checkpoint，本次从头训练。

## 定义与接口

`root_xy_error` 使用当前参考帧（`obs_current_step_index`，时间偏移 0）的 anchor 位置减去机器人当前 anchor 位置，再转到机器人仅含 yaw 的水平朝向坐标系，取 X、Y 两维。当前任务 anchor 为 `torso_link`，与根位置终止判据一致。原始单位为米，正 X 表示参考在前方，正 Y 表示参考在左侧；高度误差不进入这两维。

启用时同时传入三个 Hydra override：

```text
+task.observation.actor_feedback.root_xy_error={_target_:mimic_lite.root_xy_error}
algo.actor_extra_in_keys=[actor_feedback]
algo.in_keys=[command,policy,priv,actor_feedback]
```

独立 `actor_feedback` 组参与现有 VecNorm 标准化、rollout 保存、PPO 更新与推理；不会加入 Critic。默认 `actor_extra_in_keys=()`，不添加以上 override 即保持原有接口。

| 网络 | 默认输入维度 | XY 实验输入维度 | 隐藏层 |
|---|---:|---:|---|
| Actor | 1048 | 1050 | 512, 512, 512 |
| Critic | 1598 | 1598 | 1024, 512, 256 |

`policy` 535 维和 `command` 513 维都不变。额外信息不修改 RVQ 编码、奖励、终止条件或采样逻辑。

## 训练和复现

128 条动作，GPU 0–3，每卡 8192 环境，共 32768；2288 轮，seed 42，其余 PPO 设置沿用上次。`checkpoint_path=null`。实验目录 `active-adaptation/projects/hdmi/scripts/experiments/20261014T075846Z-train128-actorxy` 包含固定 index、完整配置、启动脚本、代码快照、测试和运行状态。

上一次 8192 环境实验从 12 动作 checkpoint 热启动，本次从头训练，两者初始化不同。因此最终效果差异不能完全归因于 XY 反馈；严格单因素比较需要使用相同初始化方式的无 XY 基线。

新 checkpoint 自带观测配置，评测工具按保存配置构建对应网络。不能把新 checkpoint 强行加载到默认 1048 维 Actor；后续无 XY 实验应使用无 XY 配置和对应 checkpoint。
