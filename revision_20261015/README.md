# 全量物理分组修订实验代码

本目录保存2026年10月15日完成的全量修订实验实际使用的数值代码。原仓库顶层Data/GPD/Pretrain/analysis及paper_data对应早期协议；本目录采用恢复原始计数后的独立物理分组，不用旧混合数据替代本轮数据。

## 当前实验范围

两芯片TH58TFT1T23BA8H（CT）和MT29F1T08EEHAF（FG）；Bake保持与Read读取干扰分别建模；P1-P7全部28组；种子42/43/44；全部461,736条源域曲线条目和384,780条有效目标曲线。共享基线不是独立采集。

主实验84项、六种传统基线28项、B7正则化敏感性84项（lambda=0复用主实验，另评估1e-6/1e-4/1e-2）、生成成本1项、统计审计1项、人工交付核对1项，共199个要求任务。fresh运行只登记这199项；本地历史249项取消任务不会被重新登记。当前范围以full_physical_run_20261014/scope_profile.json和queue_scope.py为准。config.json里保留的历史扩展字段不表示启用那些扩展。没有目标曲线子集替代全量评价。

## 代码布局

- full_physical_run_20261014/prepare_full_data.py：从原始计数矩阵和工作表恢复28个分组及坐标。
- full_common.py：数据加载、嵌套/默认观测协议、测试尾部与整体指标、原子结果保存。
- gpu_ops.py：609维tanh MLP、批量L-BFGS、确定性随机设置。
- generative_models.py和paper_revision/source_reference/original_generator：实际使用的原Transformer3与v-prediction扩散实现。保持相对目录才能按原代码导入。
- gpu_worker.py：全源拟合、原生成器两遍训练、三种子全目标适配及lambda敏感性。
- classical_worker.py及src/中的传递依赖：普通/对数空间/偏斜高斯、自然三次样条及PCHIP。
- supervisor.py、queue_scope.py：持久化任务调度、范围约束、检查点续跑与交付门。
- generator_cost_retained.py：单609维向量的同步GPU生成成本。
- finalize_retained_run.py及finalize_full_run.py中的共享函数：全量覆盖、种子统计、配对曲线/WL/PEC块区间、稳健指标与失败分类。
- delivery_statistics.py：基于已审计的完整记录生成实验类型级汇总和图数据，不重新拟合。
- cpu_retained_preanalysis.py、monitor_audit.py：本轮使用的增量统计和覆盖监督辅助程序；前者使用Windows进程优先级/锁接口。
- CODE_PROVENANCE.json：每个复制文件的原项目相对路径和SHA-256，数值源码按原字节复制。

## 数据输入

本次push发布运行代码、配置与输入清单。原始测量数据、检查点、全部预测与逐曲线结果保存在作者本地，未由本次push公开发布。metadata/data_manifest.expected.json记录28组数量、有效规则、源/目标物理档位与数据散列，可用于核对；它不是已准备数据的完成标记。不要把它手工改名为data/manifest.json来跳过预处理。

需要在full_physical_run_20261014/raw/准备以下四个目录，每个包含wlcount.mat及对应芯片型号的.xlsx工作表：

```
TH58TFT1T23BA8H_Bake/wlcount.mat
TH58TFT1T23BA8H_Bake/TH58TFT1T23BA8H.xlsx
TH58TFT1T23BA8H_Read/wlcount.mat
TH58TFT1T23BA8H_Read/TH58TFT1T23BA8H.xlsx
MT29F1T08EEHAF_Bake/wlcount.mat
MT29F1T08EEHAF_Bake/MT29F1T08EEHAF.xlsx
MT29F1T08EEHAF_Read/wlcount.mat
MT29F1T08EEHAF_Read/MT29F1T08EEHAF.xlsx
```

mat变量名为count；CT每WL扫描131个坐标、256WL，FG为164个坐标、744WL；11个PEC工作表PE-0至PE-10000。代码直接核对维度、非负计数、工作表及正计数至少12点的既定有效规则，不修改输入。现有顶层Data中的早期CSV不等同于这些输入。

## 环境与运行

运行平台为Windows，Python 3.12.14；AMD Ryzen7 6800H、16GB RAM、RTX3060 Laptop GPU 6GB。CUDA训练使用PyTorch 2.13.0+cu126，CPU统计环境使用2.14.1+cpu。实际包版本见environment_versions.json及两份requirements。GPU环境原来通过路径配置共享部分CPU依赖；在新环境中应安装完整依赖，并选择相同CPU/CUDA发行包。

在本revision_20261015目录建立.venv和.venv_gpu，调度器按这个相对布局寻找两种运行时。例如PowerShell中：

```powershell
python -m venv .venv
python -m venv .venv_gpu
.\.venv\Scripts\python.exe -m pip install -r requirements_cpu.txt
.\.venv_gpu\Scripts\python.exe -m pip install -r requirements_gpu.txt
Set-Location full_physical_run_20261014
..\.venv\Scripts\python.exe -X utf8 -B prepare_full_data.py
..\.venv\Scripts\python.exe -X utf8 -B supervisor.py
```

PyTorch的本地版本后缀依赖相应CPU/CUDA发行源，安装时须使用提供该版本的包源。代码没有捆绑虚拟环境或假设任意新版本均能逐位复现。

调度器会保留已完成检查点并从未完成部分续跑。实际训练每个源向量遍历两遍、batch16、Adam 8e-5；目标适配为固定140外步的批量L-BFGS，批内共享线搜索和曲率。源码还含历史扩展函数，但当前队列不会启动已取消的ablation/extra_gpu/extra_cpu任务。

数值与统计198项完成后，analysis_complete.json标志全28组审计完成；第199项delivery会等待论文与回复PDF的人工逐页检查及真实SHA-256放行。仓库不发布稿件和审稿文件，fresh复现停在此交付门是设计行为，不应伪造delivery_release.json或把缺失交付物算完成。数值结果可在results和analysis_retained核对。

## 证据边界

主lambda=0，固定网格不根据目标测试改默认。B5/7/9查询仅用坐标且共用固定测试集；default训练/验证观测分配另列。每条记录保留失败和未定义指标，不筛有利曲线。第二芯片独立训练不是零样本迁移；PEC与块混杂，bootstrap区间不是独立器件总体区间。两遍训练不证明收敛。已知正计数支持上的尾部误差不等于BER/ECC、真实采集时延或能耗收益。普通高斯之外的强基线和不利预算结果均保留，代码公开不代表所有方法已全面最优或原始数据已获得公开授权。
