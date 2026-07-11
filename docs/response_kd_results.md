# Response KD 结果

状态：尚未开始，符合 Gate A 协议。

Robot Arm 现阶段没有专家准入通过，不能创建 prediction-KD 或 response-KD 的结果表。
本机已完成的元数据审计发现训练集有 2,002 条、ID/OOD 各 105 条元数据轨迹，动作为
7 维；ID/OOD 210 个视频已齐全，训练视频仍在下载。Robot Arm 官方 dense checkpoint
已补齐并验证可读，但 Gate A 已输出 **C：fail，停止专家/KD**。

关键观测：full temporal shuffle gap 可测（ID `0.00811`，OOD `0.00756`），说明模型
对全动作时序破坏有响应；但局部单维扰动响应没有高于 intentionally-unpaired 噪声地板。
ID 最高组 `dim_1` 的 response/noise ratio 均值为 `0.3131`，OOD 为 `0.3976`，均低于
预注册阈值 `2.0`，且随机维度组对照没有支持稳定各向异性。

完整机器可读审计和 Gate A 决策位于数据盘：

- `/root/autodl-tmp/acwm-response-results/robot_arm/action_audit/action_audit.json`
- `/root/autodl-tmp/acwm-response-results/robot_arm/gate_a/decision.json`
- `/root/autodl-tmp/acwm-response-results/robot_arm/probe/id_ood_response_degradation.json`

不得将本条目误读为“动作通道完全无响应”：全时序 shuffle gap 存在。正确读法是：
当前 probe 没有测到可用于 specialist/KD 的稳定局部组响应失败轴，因此不得启动专家训练或 KD。
