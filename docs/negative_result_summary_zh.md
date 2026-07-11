# ACWM-Phys response-structure 负结果摘要

## 原始想法是什么

最开始的想法是：动作空间里可能存在不同动力学 regime，例如小动作/大动作；如果为不同
regime 训练 specialist teacher，再用 response KD 蒸馏给统一 student，student 也许能比
普通 prediction KD 更好地保留动作响应。

这个想法当时是合理的，因为世界模型只看 factual action 的预测误差并不够。一个模型可以
在真实动作上 MSE 不错，但在 `a`、`0`、`-a` 或局部扰动动作下几乎不改变输出。也就是说，
视觉预测好不等于 action response 学对了。

## 哪些实验杀死了这个方向

Push Cube 首先杀掉了“默认 magnitude specialist”。它的动作幅值基本恒定，所谓 small/large
不是幅值 regime。后来只在 signed target coordinate 上看到一个很窄的 feasibility signal：
response KD 能把 `MSE(pred(a), pred(-a))` 提高约 `136x` ID、`153x` OOD，而且 rollout MSE
没有明显变坏。但这只能说明 paired response 评估有用，不能说明 specialist 方法普适。

Reacher 进一步杀掉了 magnitude partition。Reacher 的 2D torque magnitude 是非退化的，
median split 也平衡，但高/低 torque 组的视觉运动 proxy 只差 `1.0705x`，动作/运动相关只有
`r=0.0546`。官方 dense baseline 本身也已经很 action-responsive，zero/opposite torque 的
paired response 在 `1e-3` 量级。因此 oracle gate 禁止训练 teacher 和 KD。

Robot Arm 是最后的 Gate A 检查。官方 dense checkpoint 已下载并验证：

```text
/root/autodl-tmp/acwm-response-checkpoints/VideoDiT_S_robot_arm_240x240/latest.pt
sha256: 438303c23acfd153ac42ddb8eb84cc412687432242bafcb1276b85f03934c6f7
```

三种子 ResponseProbe 已完成。结果显示 full temporal action shuffle gap 可测，ID 均值
`0.008114`，OOD 均值 `0.007564`，所以不能说 Robot Arm 完全不响应 action。但局部单维
扰动没有通过 Gate A：最高的 `dim_1` response/noise-floor ratio 也只有 ID `0.3131`、
OOD `0.3976`，远低于预注册阈值 `2.0`。随机维度组对照也没有支持稳定各向异性。

因此 Robot Arm Gate A 的机器决策是 `fail`。

## 现在能 claim 什么

可以 claim：

- magnitude partition 不能作为默认 specialist 轴；
- response-first gate 是必要的；
- ACWM-Phys 当前证据不支持 specialist teacher + response KD 作为主线；
- 这个分支产出了可复用的 `ActionSchema`、`ResponseProbe`、`CounterfactualEvaluator`、
  随机分组对照、三种子聚合、heatmap、response distribution 和 Gate A 机器判定。

不能 claim：

- response KD 在高维异构动作空间上有效；
- Robot Arm dense baseline 完全 action-blind；
- full shuffle gap 等于物理正确的 counterfactual controllability；
- 调低阈值或重新搜索 partition 就可以拯救当前假设。

## 下一步应该做什么

ACWM-Phys 上的 specialist/KD 方向应该冻结成负结果，不继续训练专家，不跑 KD，不调阈值。
下一步更合理的研究方向是 response-aware evaluation 和 structured action encoding。

如果未来要重新尝试 specialist 或 response KD，应该换到真正高 DoF、异构动作的数据集，例如
DexJoCo/EgoDex 风格的 arm-hand benchmark。流程必须从官方动作语义开始，先定义
`ActionSchema`，跑 Gate A 和随机分组对照；只有 Gate A 通过后，才允许 teacher admission 和 KD。
