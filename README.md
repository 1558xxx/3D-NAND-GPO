# 3D NAND

这是从原项目中单独整理出来的“本次实际使用文件”目录。

整理原则：

- 只复制本次 3D NAND 迁移学习实验真正用到的文件
- 保留 `Data / Pretrain / GPD` 的相对结构，方便复现
- 不移动原项目文件，原目录保持不变
- 不包含这次没有实际用到的脚本、配置和杂项文件

## 目录说明

- `Data/split_by_model/`
  - 本次实际跑到的 `model_2.csv` 到 `model_8.csv`

- `Pretrain/`
  - 本次预训练、目标域微调、批量跑模和汇总用到的配置与代码
  - `Models/` 中只保留了本次实际使用的 `MLPRegressor`
  - `artifacts/curve_transfer_batch/` 中保存了全部 sample/full 评估结果与 Excel 汇总

- `GPD/`
  - 本次扩散模型用到的主入口、数据准备、日志工具
  - `TimeTransformer/` 和 `denoising_diffusion_pytorch/` 保留为可运行的最小依赖包
  - `Logs/`, `Output/`, `ModelSave/`, `TensorBoardLogs/` 目录已预留，便于后续重跑

## 关键结果

- Excel 汇总：
  - `Pretrain/artifacts/curve_transfer_batch/curve_transfer_batch_results.xls`

- 最终总表：
  - `Pretrain/artifacts/curve_transfer_batch/final_summary.csv`

- 批处理状态：
  - `Pretrain/artifacts/curve_transfer_batch/batch_state.json`

## 说明

- 这份目录是“整理副本”，不是替换原仓库
- 如果后面你还想继续精简，我建议优先保留：
  - `Data/split_by_model`
  - `Pretrain`
  - `GPD`
  - `Pretrain/artifacts/curve_transfer_batch`
