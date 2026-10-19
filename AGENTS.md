# MimicLite 项目入口

接手本项目先阅读 `local/mjlab/notes/CODEX_HANDOFF.md`。其中包含 digua_4090_48 的实际环境、最新已完成训练、数据与 checkpoint 路径、评测和继续训练命令。

- 当前研究主线是 G1 RVQ latent tracker，默认 Actor 不带显式 XY 误差反馈。
- 目标机在 `/data/langlang/Project/MimicLite` 使用 root 开发；环境入口为 `local/mjlab/env.sh` 和机器专属的 `local/mjlab/.env.local`。
- 运行 GPU 工作前检查占用，显式指定 GPU；历史上获准停止的旧任务不代表可以停止新的其他任务。
- 长任务使用 Zellij，诊断与正式任务可以脚本串联；不要持续用聊天工具轮询整个训练。
- 评测必须标注使用的 index、checkpoint、终止条件；关闭终止条件的完成率不能当严格跟踪成功率。
- 代码子模块分别提交并先 push，再更新主仓库 gitlink。不要提交凭据、数据、缓存或运行产物。
- 修改 `active-adaptation` 时还应阅读该子模块的 `AGENTS.md`。
