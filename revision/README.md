# OFN 补实验（16 核 Slurm）
在新的目录解压并从解压目录提交。需要带 torch.optim.Muon 的 PyTorch、DeepXDE 及项目依赖。复用原服务器环境时，默认路径为 $HOME/HeatProject/.venv/bin/python，路径不同请先设置 REV_PY。独立环境可使用包内 requirements-linux.txt。Slurm 默认使用 T1 分区、16 CPU、32 GB 内存，单个数组元素的时限为 24 小时。

## 提交
```bash
unzip OFN_revision_server16.zip
cd OFN_revision_server16
sha256sum -c SHA256SUMS > /dev/null
bash revision/submit.sh
squeue -u "$USER"
```
只想先跑服务器短检查时，单独执行 `sbatch revision/server.slurm smoke`。完整提交会先对两道方程分别做短检查，再按顺序提交调参、冻结选择、正式训练和评估。调度使用 afterok 依赖和数组并发上限 1，一次最多占一台 16 核节点。训练阶段每个数组元素启动四个进程，各用四个 CPU 线程，每个进程最多负责两次训练。每个方程最后一个正式数组批次可能有空位。

## 实验口径
- 两道方程分别用种子 289–291 调参。Full 与 No-Flow 各用三种优化方案。split 将三个输出投影矩阵交给独立 Adam，其余参数交给另一个 Adam。hybrid 只把同三个矩阵交给 Muon。两种方案使用同一张联合网格，基础学习率 5 个，投影学习率 6 个。单学习率 Adam 扫 5 个值。这个补充方案采用预先固定的有限网格，解释端点最优时需说明搜索范围。
- 每次调参训练 2 万步。对每个种子取最后三个验证时点的误差中位数，再取三种子的均值。一组候选中只要有一个种子数值发散，该候选即失去入选资格。共同学习率来自两架构、三种方案的等权验证分数。冻结后，正式训练均使用种子 292–301。
- 去归一化和固定门控的结构对照沿用对应原结构的已选学习率。长预算比较保持原 2 万步组的初始学习率，从头运行 4 万步完整余弦调度。相同训练任务在多个比较间共用。每个方程共 390 次调参、最多 220 次正式训练，其中 40 次正式训练为 4 万步。
- 热方程原始九配置的 90 份最终权重随包提供，每份均已核对原始结果哈希和公开测试表的 evaluation_key。两个热方程评估共享预先固定的 1,000 个新测试输入及原有训练预处理。原始模型间的误差差值按测试输入配对，同时区间采用 Bonferroni 校正，以原有十个训练种子为条件。
- 完整作业为长周期实验。参考先前正式记录中 OFN 单个 2 万步任务的墙钟中位数，热方程约 3.3 小时，Burgers 约 3.5 小时。按四路并行粗估两道方程约 40–46 个连续节点日。租金可按这个范围乘 24 再乘节点小时单价。快速发散、硬件差异、排队或中断均会改变实际用时。

## 结果与恢复
- 作业日志为解压目录的 revision-*.out。两道方程各自的选择与正式统计位于 revision_runs/heat/ 和 revision_runs/burgers/，重点查看 selection.json、evaluation.json。运行明细和可恢复检查点保留在各阶段子目录。
- results_fresh_test_revision/analysis/ 包含原始九配置的新测试统计与配对排序区间。revision/burgers_leave_one_out.csv 是由原始正式种子结果计算的留一法分析。
- 同一目录下重新执行同一数组批次时，完成的训练读取缓存，中断的任务从上次完整 checkpoint 恢复。单独重提一个数组元素可用 `sbatch --array=批次号 revision/server.slurm tune heat` 等命令。Slurm 中失败作业的旧 afterok 依赖可能不再触发，如需重新提交整个链，先取消本次链中仍挂起的旧作业，保留结果目录。
