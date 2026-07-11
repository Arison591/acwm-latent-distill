# Robot Arm ResponseProbe

`acwm.action_latent.response` 提供：

- `ActionSchema`：组、归一化统计、表示类型及每条语义证据；
- `ActionPerturbationSampler`：训练集尺度的可重放局部扰动；
- `ResponseProbe` / `CounterfactualEvaluator`：成对响应与噪声地板计算。

每次成对预测固定相同观测、动作历史和初始扩散噪声。噪声地板则特意保持动作不变
而更换初始噪声，用来排除“采样随机性被误当作动作响应”。逐窗口 JSON 不会静默
以零帧替代缺失视频。

三种子执行示例：

```bash
for seed in 0 1 2; do
  python scripts/response_structure/run_response_probe.py \
    --cfg configs/envs/robot_arm.yaml --ckpt /数据盘/robot_arm_dense.pt \
    --schema /root/autodl-tmp/acwm-response-results/robot_arm/action_audit/action_schema.json \
    --seed "$seed" --output "/root/autodl-tmp/acwm-response-results/robot_arm/probe/seed_$seed"
done
```

每个种子必须使用无物理语义前提的五类操作：全时序置换、组内时序置换、组掩码、
局部加性和带符号局部方向扰动。表示为 `unknown` 时，零动作与缩放会被拒绝而不是
被误解释。探针同时记录逐窗口 factual MSE、masked MSE、PSNR、SSIM、响应/噪声地板
比、shuffle gap，以及有限差分方向相关性。

`scripts/response_structure/gate_a.py` 只有在三个种子、至少一个组的响应/噪声地板
比达到预注册阈值（默认 2）、并且三种子随机维度组对照齐全时才可能输出 `pass`；
否则是 `fail` 或 `insufficient_evidence`。通过 Gate A 前不得定义专家或运行 KD。

当前 Robot Arm 运行（2026-07-11）使用官方 dense checkpoint
`/root/autodl-tmp/acwm-response-checkpoints/VideoDiT_S_robot_arm_240x240/latest.pt`
（step 100000, sha256 `438303c23acfd153ac42ddb8eb84cc412687432242bafcb1276b85f03934c6f7`）。
ID/OOD 各 3 seed、每 seed 2 个窗口、10 denoising steps 的 probe 已完成。ID split
最高局部响应/噪声比来自 `dim_1`，均值为 `0.3131`；OOD `dim_1` 均值为 `0.3976`。
二者都远低于 Gate A 阈值 `2.0`。full temporal shuffle gap 可测（ID 均值
`0.00811`，OOD 均值 `0.00756`），但单维局部扰动没有稳定高于 intentionally-unpaired
noise floor。

结果目录 `results/response_structure/` 是指向数据盘
`/root/autodl-tmp/acwm-response-results/` 的符号链接，满足仓库路径约定而不占用系统盘。
