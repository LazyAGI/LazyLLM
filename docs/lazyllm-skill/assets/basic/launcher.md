# Launcher

分布式或集群任务通过 `lazyllm.launchers` 指定启动方式。完整 API 见项目 `docs/zh/API Reference/launcher.md` 或 `docs/en/API Reference/launcher.md`。

## 常用 Launcher

| 名称 | 说明 |
|------|------|
| EmptyLauncher | 空启动器，常用于本地单进程或测试 |
| RemoteLauncher | 远程节点启动，可指定 nnode、nproc、ngpus 等 |
| SlurmLauncher | 基于 Slurm 调度 |
| ScoLauncher | 基于 Sco 调度 |
| K8sLauncher | 基于 Kubernetes |
| Job | 作业封装 |

## 使用示例

```python
from lazyllm import launchers

# 单机多卡
launchers.remote(ngpus=4)

# 微调 / 部署时传入
model = lazyllm.TrainableModule('qwen2-1.5b').finetune_method(
    lazyllm.finetune.llamafactory,
    launcher=launchers.remote(ngpus=4)
)
```

Flow 中带 launcher 的组件也可传入 `launcher=lazyllm.launchers.empty()` 等，见 [references/flow.md](../../references/flow.md) 与 [deploy_framework.md](../finetune/deploy_framework.md)。
