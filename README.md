# 重要补丁

现在加入新的两档简易难度，方便大家的分数能够有区分度

- `level = 1`，还是原本的任务

- `level = 2`，舍弃了对qi的约束要求

- `level = 3`，舍弃了所有约束要求

```sh
python vmec_eval.py --boundaries ./submission.jsonl --out result.json --parallel --workers 16 --level 3
```

返回的json文件中可以找*score字段查看自己的评分

**注：不强制针对后两个简单任务进行优化**

# AI-Driven Design and Optimization of Magnetic Field Configurations for Stellarator Fusion Devices

![logo](logo.png)

## 背景

可控核聚变是被寄予厚望的终极清洁能源路线之一。在磁约束聚变（MCF）的两大主流构型
里，**托卡马克（tokamak）** 依靠等离子体自身的环向电流提供旋转变换（rotational
transform），结构简单但天生有破裂、电流驱动等稳态运行难题；**仿星器（stellarator）**
则把扭转完全外推到外部 3D 线圈上，等离子体本身不需要净环向电流，因而原则上可以
连续稳态运行、不存在破裂风险。代价是磁场位形高度三维，线圈形状极其复杂，对设计与
加工都提出了远高于托卡马克的要求。

仿星器设计的核心问题，可以粗略概括为：**给定一组等离子体边界（plasma boundary）
的傅里叶描述**，反演出一个三维磁场位形，并要求它同时满足好的物理性质（MHD 平衡 /
稳定、Quasi-Symmetry 或 Quasi-Isodynamicity 以保证粒子约束、合理的 ι/N、镜比、
磁井 …）与好的工程性质（线圈不要太扭、aspect ratio 不要太极端、伸长率受控 …）。
这是一个高维、非凸、强耦合、单次评估昂贵（每条边界都要跑一次 VMEC 收敛平衡）的
黑箱优化问题，传统上严重依赖物理直觉与逐点梯度方法。

近年来 Proxima Fusion 等机构开源了 **ConStellaration** 基准，把仿星器边界优化
抽象成几个可复现的 benchmark 问题，让 AI / 优化社区能够在统一接口下竞争。本仓库
聚焦其中的 **Problem 2 — Simple-to-Build QI Stellarator**：在保证 Quasi-
Isodynamic 品质和若干几何约束的前提下，寻找尽可能"好造"（aspect ratio 低、
elongation 受控、镜比合理）的边界。我们把它做成一个协作工作区——大家用任意 AI /
优化方法生成候选边界，统一通过本仓库的评估脚本打分、写进 `submission.jsonl`，
并在过程中沉淀对仿星器边界搜索空间的工程经验。

## 任务简介

- **问题**：在 stellarator 对称、给定 `n_field_periods` 的约束下，搜索一组傅里叶
  系数 `r_cos` / `z_sin`，使得 VMEC 平衡满足下面 5 条约束，并尽量降低 aspect ratio。
- **评估对象**：`constellaration.problems.SimpleToBuildQIStellarator`（即
  problem2），feasibility 非负；`feasibility ≤ 1e-2` 视为可行（cons 内置 1%
  容差），越接近 0 越好。
- **约束**（problem2 不评估 MHD 稳定性，那是 problem3 的事）：
  - `aspect_ratio` ≤ 上限
  - `edge_rotational_transform / n_field_periods` ≥ 下限
  - `log10(qi 残差)` ≤ 上限
  - `edge_magnetic_mirror_ratio` ≤ 上限
  - `max_elongation` ≤ 上限

## 环境配置

本仓库的代码只依赖一个外部包：`constellaration`。装完它就能直接跑，不需要
再额外配 VMEC、Nevergrad、scipy、numpy 等——它们都会被 `constellaration`
自动拉成依赖。

**推荐 Python 3.12**（3.10 / 3.11 也能用），原生 venv 一行装好：

```bash
# 用原生 venv 起一个干净环境
python3.12 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install constellaration
```

装完直接跑：

```bash
# 评估自家 submission
python vmec_eval.py --boundaries ./submission.jsonl --out result.json --parallel --workers 16
```

## 目录结构

```
.
├── README.md            # 本文件
├── logo.png             # 仓库 logo
├── vmec_eval.py         # VMEC + problem2 批量评估器（支持并行 / 串行开关）
├── submission.jsonl     # 候选边界提交文件，每行一条 {tag, nfp, boundary}
└── result.json          # 评估脚本的输出样例
```

## 数据集

外部输入只有一份已 VMEC 评估的边界数据集。SII-AI4Fusion 团队把分散在不同来源、
不同字段命名、不同 VMEC 收敛状态的边界统一成可直接喂给 problem 2 评估器的格式，
非常适合用来做生成模型的预训练 / 条件生成、近邻检索、热启动 CMA-ES 的初始种群，
或是直接当 baseline 边界过滤再提交。建议把它作为本仓库的"先验数据池"。

> **SII-Stellarator-Configuration-Dataset**
> https://huggingface.co/datasets/SII-AI4Fusion/SII-Stellarator-Configuration-Dataset

按 NFP 分文件，每个 sample 含 `boundary.r_cos` / `boundary.z_sin`（5×9 矩阵）+
9 个 VMEC metric：

| 文件 | 行数 | NFP |
|---|---|---|
| `nfp1_n14607.parquet` | 14,607 | 1 |
| `nfp2_n19945.parquet` | 19,945 | 2 |
| `nfp3_n68191.parquet` | 68,191 | 3 |
| `nfp4_n27798.parquet` | 27,798 | 4 |
| `nfp5_n28144.parquet` | 28,144 | 5 |
| **总计** | **158,685** | |

[hf-ds]: https://huggingface.co/datasets/SII-AI4Fusion/SII-Stellarator-Configuration-Dataset

## 提交格式 — `submission.jsonl`

每行一条 JSON，字段如下：

```json
{
  "tag": "yourname",                 // 任意标记，用于区分作者 / 版本
  "nfp": 3,                          // 场周期数
  "boundary": {
    "r_cos": [[...], [...], ...],    // 形状 (mpol, 2 * ntor + 1)
    "z_sin": [[...], [...], ...]     // 同上
  }
}
```

`r_sin` / `z_cos` 与 `is_stellarator_symmetric` 由评估脚本自动补齐，默认走
stellarator 对称分支。

## 评估脚本 — `vmec_eval.py`

`vmec_eval.py` 跑的就是和**学院线上评测一模一样**的 problem 2 评估流程
（同一份 `forward_model` + `SimpleToBuildQIStellarator`，同一组 high-fidelity VMEC 设置）。
它直接吃你的标准格式 `submission.jsonl`，给每条边界算出 `feasibility` 和关键 metrics——
**本地跑出什么分，上传到学院平台就是什么分**，没有任何近似或简化。

> ⚠️ **学院评测系统每人每天最多 3 次，且单次只接受一条边界**——不支持并行 / 批量
> 评估，每次提交只能验证一个候选。既然本地评估结果与线上完全一致，请把候选筛选、
> 调试、迭代全部放在本地完成（这边可以并行扫几十几百条），**不要频繁在线提交**——
> 线上 quota 应该留给真正想上榜的那一条最优结果，频繁占用会给评测系统带来不必要
> 的压力，也会浪费自己的提交机会。

所以推荐的用法是：每次迭代完成 `submission.jsonl` 之后，先在本地用它跑一遍，
确认分数稳定后再上传。

### 一句话使用

```bash
# ✅ 推荐：并行评估，16 个进程
python vmec_eval.py --boundaries ./submission.jsonl --out result.json --parallel --workers 16
```

输入就是你即将上传到学院平台的 `submission.jsonl`，输出 `result.json` 里每条记录对应
你 `submission.jsonl` 里的同一行（`index` 字段对齐），包含：

- `success`：VMEC 是否收敛、几何是否合法
- `feasibility`：problem 2 的 feasibility 分数，非负；`feasibility ≤ 1e-2`
  视为可行（cons 内置 1% 容差），越接近 0 越好。这就是榜单分的核心
- `signed_violations`：5 条约束的归一化违背量（> 0 表违反，≤ 0 表满足，方便定位是哪条约束没过）
- `metrics`：`aspect_ratio` / `edge_rotational_transform_over_n_field_periods` / `qi` /
  `edge_magnetic_mirror_ratio` / `max_elongation` / `minimum_normalized_magnetic_gradient_scale_length`
- `error`：失败时的异常信息
- `elapsed`：单条评估耗时

### 强烈建议开并行，能大大加快速度

VMEC 单条边界少则几十秒，多则几分钟。要评估几十几百条候选（CMA-ES 一代就是这量级），
串行跑一晚都未必跑完，**并行几乎线性加速**——脚本基于 `ProcessPoolExecutor` 做进程级
并行，加 `--parallel --workers N` 就行。

为什么是进程级并行而不是线程：VMEC 是 CPU 密集且不彻底释放 GIL 的 Fortran 后端，
线程并行基本无效。脚本会在每个子进程里把 `OMP / MKL / OPENBLAS / NUMEXPR_NUM_THREADS`
强制锁成 1，避免外层进程并行 + 内层 BLAS 多线程互相抢核——这是单机能压满 CPU 的关键。

`--workers` 一般取 CPU 物理核数的 0.5～1 倍：核多内存大就放开，机器吃紧就压低。

### 评估时的进度日志

```
[vmec_eval] start: n=2 mode=parallel workers=16
[vmec_eval] [  1/2] idx=  0 ok  feas=0.1234 t=42.1s (elapsed 42.3s)
[vmec_eval] [  2/2] idx=  1 ok  feas=0.0876 t=58.7s (elapsed 60.0s)
[vmec_eval] done: 2/2 ok, mode=parallel, elapsed=60.0s
```

每条：`[完成数 / 总数] idx=submission.jsonl 中的行号 ok / ERR feas=分数 t=本条耗时 (elapsed 总耗时)`，
失败会带 error 信息，方便你回到 `submission.jsonl` 里定位是哪条边界出了问题。

### 作为库调用嵌进优化循环

如果你的优化算法本身就要反复评估候选边界（CMA-ES 一代打分、生成模型重排），可以
直接 import 进去，省去写中间文件的麻烦：

```python
from vmec_eval import evaluate_boundaries
# boundary_json_list: list[str]，每个元素是 SurfaceRZFourier 的 JSON 字符串
results = evaluate_boundaries(boundary_json_list, parallel=True, workers=16)
# results 与输入等长、顺序对齐
```

返回字段与命令行版完全一致。

### 它具体做了什么（可选展开）

如果你只是想用，看到这里就够了。下面是实现细节，方便排查问题或二次开发：

1. **读边界**：从 `submission.jsonl` 逐行读，兼容两种格式——平台标准格式
   `{"tag", "nfp", "boundary": {"r_cos", "z_sin"}}` 和老格式裸 `SurfaceRZFourier`
   JSON / dict；自动补齐 `n_field_periods`、`is_stellarator_symmetric`、
   `r_sin=None`、`z_cos=None`。
2. **跑 VMEC**：对每条边界调用 `constellaration.forward_model.forward_model`
   （`default_high_fidelity` 设置，与学院平台完全一致），算出 MHD 平衡的 metrics。
3. **打 problem 2 分**：用 `problems.SimpleToBuildQIStellarator()` 算 `feasibility`
   和归一化 signed constraint violations。
4. **抽关键指标**：把上面列的 6 个 metric 拎出来，非有限值（NaN / inf）一律置 None。
5. **失败兜底**：VMEC 不收敛 / 几何非法 / 数值爆炸时不抛异常，把
   `{"success": False, "error": "..."}` 塞进当条结果，整批不会因为一条挂掉。
6. **写结果**：所有结果一次性写到 `--out` 指定的 JSON 文件。


## 参考 Baseline

ConStellaration 官方仓库 [proximafusion/constellaration](https://github.com/proximafusion/constellaration)
在 `optimization_examples/` 下提供了一份针对 problem 2 (`SimpleToBuildQIStellarator`)
的端到端求解脚本，可以直接当作对照基线：

- **`launch_alm_simple_to_build_stellarator.py`** —— 增广拉格朗日 (ALM) 外环
  + Nevergrad 黑箱内环，默认 48 worker 并行评估，适合多核机器、全局搜索、
  强约束场景。

脚本从 NAE 解析初值出发，在 `mpol = ntor = 4` 的傅里叶子空间里搜索；目标
是最小化 `aspect_ratio`，约束就是 problem 2 的 5 条；VMEC 默认走 `low_fidelity`
省时间。

把脚本跑完得到的最终边界转成本仓库 `submission.jsonl` 的格式，再用
`vmec_eval.py` 在 high-fidelity 下复评，就能与你的 AI 方法做横向对比。

### 进阶：DESC —— JAX 写的可微 VMEC 求解器

除了 ConStellaration 自带的两份 baseline，实际优化中还非常推荐用
[PlasmaControl/DESC](https://github.com/PlasmaControl/DESC)。DESC 是一个用
JAX 写的 MHD 平衡求解器，本质上是 **VMEC 的可微替代品**——解的还是同一套理想
MHD 力平衡方程（固定边界），但实现路径完全不同，由此带来几个关键优势：

- **谱方法 + 整体牛顿**：径向 Zernike + 角度 Fourier 投影，把"找平衡"变成
  解一个非线性最小二乘问题，**对 ill-conditioned 的强 3D 几何更稳**，VMEC
  容易卡住的边界 DESC 还能收敛。
- **整段代码可微**：DESC 整套求解器是一个长链 JAX 函数，从「边界傅里叶系数」
  一路到「`aspect_ratio` / `iota` / QI 残差 / 镜比 / 伸长率」全程可微。
  一行 `jax.grad(metric)(boundary)` 就能拿到 metric 对边界的真实导数（一次
  反向传播 ≈ 一次前向求解的代价），而 VMEC 是黑箱、只能数值差分。
- **能解锁的方法**：梯度下降 / L-BFGS / 拉格朗日乘子真实梯度更新 / 二阶
  Hessian / 把 DESC 当 differentiable loss 端到端训生成模型——这些都是
  VMEC 黑箱版下做不了或非常昂贵的事。

⚠️ **使用前请注意**：

- **环境依赖需要自行安装**：DESC 不在 `pip install constellaration` 的依赖里，
  需要单独 `pip install desc-opt`，并且 JAX / GPU 后端可能要按机器额外配置。
- **数值结果与 VMEC 略有差异**：DESC 与 vmecpp 用了不同的离散化方式，收敛到
  的解略有偏差，下游所有 metric 都会跟着偏。
- **打榜分一律以根目录 `vmec_eval.py` 为准**（学院线上也是 vmecpp）。DESC 的
  正确角色是「优化阶段的快速 + 可微代理评估器」，而不是最终评分器——内部循环
  用 DESC 找候选 / 拿梯度，最终最优的几条再用真 VMEC 复评一次。

## 工作流程与求解思路

整体的协作循环只有四步：

1. 用你喜欢的方法（梯度 / 进化 / 贝叶斯 / 生成模型 …）生成一批候选边界。
2. 按 [提交格式](#提交格式--submissionjsonl) 写到 `submission.jsonl`
   （多人多 tag 可以共存）。
3. 跑 `vmec_eval.py` 拿到 feasibility 与 metrics（强烈建议开并行）。
4. 挑出 `feasibility ≤ 1e-2` 且最接近 0 的几条，回到第 1 步迭代下一轮；觉得
   稳了就上传到学院平台参与排行。

第 1 步是最有发挥空间的环节。下面只是起点参考，欢迎尝试任意方法：

- **CMA-ES / 进化策略**：对傅里叶向量直接搜索，搭配 Augmented Lagrangian 处理
  约束。从 [SII 数据集][hf-ds] 里挑几条 feasibility 最优的样本作为初始均值，
  能显著缩短收敛时间。
- **梯度优化**：利用 problem2 的可微替身（代理模型）做局部精修。
- **贝叶斯优化 / TuRBO**：适合预算紧、单次评估昂贵的场景。
- **生成模型 + 重排**：在 [SII 数据集][hf-ds] 上预训练 VAE / Diffusion，
  从隐空间采样得到候选，再用本评估器筛选。
- **多保真**：先用 low-fidelity VMEC 粗筛，再用 high-fidelity 精评。
- **数据集近邻检索**：把目标约束嵌入边界空间，直接在 [SII 数据集][hf-ds] 里检
  最相近的几条作为 baseline 提交，简单但有效。

## 学术诚信与提交方式

ConStellaration 官方 benchmark 上已经有不少团队公开的先进解，里面用到的
搜索策略、初始化技巧、约束处理、超参选择都很值得借鉴，**作为方法学参考阅读
是鼓励的**。

但是请注意：**严禁直接爬取官方榜单上的样本边界，再原样提交到学校平台**。
那不是优化算法的成果，是抄袭。

为了从机制上杜绝作弊，最终的提交方式如下：

- 选手把自己的优化算法代码完整放到容器内的 `/root/fusion/` 目录下；
- 把开发环境打成集群镜像，**镜像名按 `姓名+学号` 命名**（例如
  `创小智CZXSxxxxxx`），方便助教检索与对应；
- **注意：必须在提交截止时间之后才把镜像设为公开**——提前公开会被视为实训
  未完成；并且镜像一旦公开，所有参赛者都能看到里面的代码和实现细节，所以
  提前公开等于把自己的方案直接送给别人；
- 截止时间后，助教会拉取（已公开的）镜像，在统一硬件上重新运行 `/root/fusion/`
  下的代码进行测评打分；
- 因此提交里的边界必须是**镜像内代码当场跑出来的**，而不是预先塞进文件里
  的现成结果。任何企图绕过这一流程的行为都将被认定为作弊；
- 同时把 `/root/fusion/` 下的代码打成压缩包（命名同样为 `姓名+学号.zip`），
  **微信私发给助教**作为双备份，防止镜像出问题时无法评分。

## 评分标准

最终成绩由三部分组成：**榜单得分**、**代码质量分** 和 **实训期间表现**。
三部分都要参与，缺一不可。

### 一、榜单得分（自动打分）

把符合本仓库 [提交格式](#提交格式--submissionjsonl) 的 `submission.jsonl`
上传到学院平台，平台会自动跑评估并出分，根据得分排行——这是最终评分指标之一。
评分核心是 problem 2 的 feasibility / objective 综合得分，越靠前的名次拿到的
榜单得分越高。

> 🚨 **不满足约束直接 0 分**。这是 ConStellaration 评分的硬规则：只要 5 条约束
> 中任何一条违背超过 1% 容差（即 `feasibility > 1e-2`），`score` 直接判 0，
> 不会再看 objective 多漂亮。所以提交前**一定要先在本地用 `vmec_eval.py` 自评
> 一遍**，确认 `feasibility ≤ 1e-2` 再上传，**不要把不可行的边界浪费在每天 3
> 次的提交配额上**。

### 二、代码质量分（人工 / 助教评审）

只有得分高没用，**用什么方法拿到的分**同样重要。助教会拉取你公开的镜像，
进入 `/root/fusion/` 阅读代码，按下面的 rubric 打分（满分 100）。

| 维度 | 分值 | 评分要点 |
|---|---|---|
| **可复现性** | 20 | 一条 `bash run.sh` / `python main.py` 就能从初始化跑到产出 `submission.jsonl`；随机种子固定；依赖说明清楚，没有手动 patch 也没有未提交的二进制文件；多次运行结果稳定可复现 |
| **方法新颖性** | 20 | 不是直接抄官方 baseline 或 SII 数据集里的现成边界；有明确的算法贡献（新的初始化、新的搜索策略、新的代理模型、新的约束处理 …）；与官方 baseline 的差异点描述清楚 |
| **算法合理性** | 15 | 优化器选择 / 约束处理 / 收敛判据 / 超参与问题尺度匹配；目标函数设计能说清为什么；有 ablation 或对比实验支撑结论 |
| **代码工程质量** | 15 | 模块清晰，函数 / 类职责单一；命名可读；没有死代码、没有大段被注释掉的实验残骸；关键 magic number 有注释解释 |
| **文档与可读性** | 10 | `README.md` 写清楚算法思路 / 入口 / 配置项 / 依赖 / 一键运行；关键模块有简明 docstring；曲线 / 中间结果有保留 |
| **算力效率** | 10 | 单次 VMEC 评估失败有兜底；并行 / 多保真等加速手段使用合理；CPU / 内存占用与产出得分匹配（避免无谓暴力扫） |
| **可扩展性** | 5 | 把求解器与 problem 解耦，能轻松迁移到 其他问题 / 不同 NFP / 不同约束上限；超参可配置而不是硬编码 |
| **学术诚信** | 5 | README中明确标注哪些是自己的工作、哪些是改写自他人；无作弊行为（见 [学术诚信](#学术诚信与提交方式)） |

未通过项的硬否决条件（任意一条触发，代码质量分判 0）：

- 提交的 `submission.jsonl` 不是镜像内代码当场跑出来的（夹带预先存好的边界）；
- 镜像在截止时间前公开，或截止时间后拒绝公开；
- 直接搬运官方榜单 / SII 数据集中的边界，无任何后处理或新算法成分；
- `/root/fusion/` 下代码缺失、跑不起来、或助教在统一硬件上无法复现。

### 三、实训期间表现

参与度、沟通协作、对反馈的响应等过程性表现，由助教与老师在实训过程中观察记录，
也会作为最终成绩的一部分。

### 总分

最终成绩由榜单得分、代码质量分和实训期间表现综合得出。建议同学把精力**前期投在算法本身（拉高榜单分）**、**中期投在过程参与和沟通（拉高实训表现分）**、
**后期投在代码整理和 README（拉高代码质量分）**——三边都不能偏废。
