# Robot Arm Gate A 与后续准入

本轮只执行 Gate A。当前判断规则：

1. ID/OOD 评估 split 元数据与视频必须版本一致且完整；train 全量 inventory 不完整时继续补齐并记录；
2. 使用相同扩散初始噪声后，局部扰动响应须超过独立噪声地板；
3. 该结论必须在三种子及至少一个预注册动作组中稳定；
4. 该结论还必须大于三种子随机维度组对照方差；
5. 若失败，结论是修复数据或条件评估，不训练专家。

当前结果文件在 `/root/autodl-tmp/acwm-response-results/robot_arm/`。官方 dense
checkpoint 已可用，ID/OOD eval split 完整，Gate A 已执行并输出 `fail`：

```text
results/response_structure/robot_arm/gate_a/decision.json
```

失败原因不是缺 checkpoint，而是没有任何维度组满足局部响应/噪声比 >= 2 且超过随机分组方差。
因此停止 Robot Arm specialist/KD 路径；下一步只能诊断 action-conditioning 路径、
扰动尺度/评估器，或转为 response-aware evaluation/structured action encoding。
