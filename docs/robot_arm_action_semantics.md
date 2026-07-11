# Robot Arm 动作语义审计

状态：**未解决，禁止推断。**

当前仓库登记 Robot Arm 的动作维度为 7。已固定审计使用的官方数据集修订为
`4017e5f5900a7a8590d4e944b6a84f55df75a0c1`，但数据卡、元数据与当前工作区仍未提供
可核验的逐维控制语义。因此 `ActionSchema` 只使用稳定索引
`dim_0` 至 `dim_6`，动作表示为 `unknown`；没有虚构任何关节、末端执行器或
控制方式名称。

在官方资料确认前，允许的诊断是：按组掩码、全时序/组内时序置换、按训练集
标准差的局部加性扰动、带符号局部方向扰动。零动作、缩放和反向动作不能被解释
为物理反事实，工具会拒绝执行 `zero_action`。

审计命令（结果存数据盘，避免占用系统盘）：

```bash
ACWM_DATA_ROOT=/root/autodl-tmp/acwm-response-data \
python scripts/response_structure/audit_robot_arm.py \
  --data-root /root/autodl-tmp/acwm-response-data/kinematics/robot_arm_64 \
  --dataset-revision 4017e5f5900a7a8590d4e944b6a84f55df75a0c1 \
  --output results/response_structure/robot_arm/action_audit
```

产物 `action_audit.json` 会记录所有元数据路径、缺失视频样例、版本可见性、逐维
分布、协方差/相关、时间自相关、符号跃迁、一阶/二阶差分、有效协方差秩和同时
活跃维度数。ID/OOD 评估 split 不完整会阻断 Gate A；train 全量 inventory 不完整时
继续补齐并记录，不用零帧或缺失样本替代。
