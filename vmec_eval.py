"""vmec_eval.py — VMEC 边界批量评估器(支持并行 / 串行切换)。

设计思路:
  * 输入一组 stellarator 边界(SurfaceRZFourier 的 JSON 字符串列表),
    逐个调用 constellaration 的 forward_model 跑 VMEC,得到 MHD 平衡态;
  * 用 problem2 (`SimpleToBuildQIStellarator`) 计算 feasibility 与归一化约束违背量;
  * 提取若干关键几何/物理指标(aspect_ratio、ι/N、QI 残差、镜比、最大伸长率等)汇总返回。

problem2 (Simple-to-Build QI Stellarator) 的 5 条约束:
  - aspect_ratio                                ≤ 上限
  - edge ι / N_FP                               ≥ 下限
  - log10(QI 残差)                              ≤ 上限
  - edge magnetic mirror ratio                  ≤ 上限
  - max elongation                              ≤ 上限
注意 problem2 并不要求 MHD 稳定(那是 problem3 的事)。

并行模式由 `parallel` 开关控制:
  * parallel=True  -> 用 ProcessPoolExecutor 多进程并发评估(VMEC 是 CPU 密集且
                     释放 GIL 不彻底,进程级并行最稳);
  * parallel=False -> 单进程顺序评估,便于 debug 或在小核数机器上避免开销。
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


def _eval_one(args: tuple[int, str]) -> dict[str, Any]:
    """评估单个边界。返回结构化结果 dict,绝不抛异常(失败信息塞进 error 字段)。

    设计为顶层函数,这样在 ProcessPoolExecutor 下能被 pickle 传给子进程。
    """
    i, boundary_json = args

    # 子进程里强行把 BLAS / OpenMP 的线程数压到 1。
    # VMEC 在外层已经做进程并行了,内层再开多线程只会互相抢核、整体反而变慢。
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    out: dict[str, Any] = {
        "index": i, "success": False, "feasibility": None,
        "metrics": None, "signed_violations": None,
        "error": None, "elapsed": None,
    }
    t0 = time.time()
    try:
        # 延迟 import:确保子进程内独立加载,主进程不必背着这些重模块。
        from constellaration import forward_model, problems
        from constellaration.geometry import surface_rz_fourier

        # 反序列化边界 -> SurfaceRZFourier 对象
        boundary = surface_rz_fourier.SurfaceRZFourier.model_validate_json(boundary_json)

        # high-fidelity 设置:跑真正的 VMEC,精度足够喂给 problem2 评分
        setting = forward_model.ConstellarationSettings.default_high_fidelity()
        m, _ = forward_model.forward_model(boundary=boundary, settings=setting)

        # ===== problem2:Simple-to-Build Quasi-Isodynamic Stellarator =====
        # 只看几何 / QI 残差 / 镜比 / 伸长率,不评估 MHD 稳定性
        prob = problems.SimpleToBuildQIStellarator()
        feas = float(prob.compute_feasibility(m))
        # 归一化的 signed constraint violation:>0 表违反,<=0 表满足
        signed = prob._normalized_constraint_violations(m).tolist()

        # 抽取要回传的指标(只挑会用到的,避免把整个 metrics 对象 pickle 回主进程)
        md: dict[str, Any] = {}
        for name in (
            "aspect_ratio",
            "edge_rotational_transform_over_n_field_periods",
            "qi",
            "edge_magnetic_mirror_ratio",
            "max_elongation",
            "minimum_normalized_magnetic_gradient_scale_length",
        ):
            v = getattr(m, name, None)
            try:
                v = float(v) if v is not None else None
                # NaN / inf 一律视为缺失,后续打分阶段可以直接判 None
                if v is not None and not math.isfinite(v):
                    v = None
            except Exception:
                v = None
            md[name] = v

        out.update(success=True, feasibility=feas,
                   metrics=md, signed_violations=signed)
    except Exception as exc:
        # VMEC 不收敛 / 几何非法 / 数值爆炸 都会落到这里,记下异常类型即可
        out["error"] = f"{type(exc).__name__}: {exc}"

    out["elapsed"] = time.time() - t0
    return out


def evaluate_boundaries(
    boundaries: list[str],
    *,
    parallel: bool = True,
    workers: int = 16,
) -> list[dict[str, Any]]:
    """对一批边界 JSON 做 VMEC + problem2 评估。

    Args:
        boundaries: 边界 JSON 字符串列表(SurfaceRZFourier 的序列化形式)。
        parallel:   True 启用多进程并行;False 走单进程顺序评估。
        workers:    并行模式下的进程数;一般取物理核数的 0.5~1 倍。

    Returns:
        与输入等长的结果列表,顺序严格对齐输入索引。
    """
    n = len(boundaries)
    # 预分配占位符,后面按 index 回填,保证返回顺序与输入一致
    results: list[dict[str, Any] | None] = [None] * n
    tasks = list(enumerate(boundaries))

    mode = "parallel" if (parallel and n > 1) else "serial"
    print(f"[vmec_eval] start: n={n} mode={mode} workers={workers if mode=='parallel' else 1}",
          flush=True)
    t_start = time.time()
    done = 0

    def _log(r: dict[str, Any]) -> None:
        # 每条评估完成时打一行:进度 / 是否成功 / feasibility / 耗时
        nonlocal done
        done += 1
        feas = r["feasibility"]
        feas_s = f"{feas:.4f}" if isinstance(feas, (int, float)) else "  N/A"
        tag = "ok " if r["success"] else "ERR"
        err = f" err={r['error']}" if r["error"] else ""
        print(f"[vmec_eval] [{done:>3d}/{n}] idx={r['index']:>3d} {tag} "
              f"feas={feas_s} t={r['elapsed']:.1f}s "
              f"(elapsed {time.time()-t_start:.1f}s){err}", flush=True)

    if parallel and n > 1:
        # 多进程:适合一次评估 popsize 个候选(CMA-ES / 进化算法批量打分场景)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_eval_one, t) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                results[r["index"]] = r
                _log(r)
    else:
        # 串行:零并发开销,debug 时栈帧清晰,失败也能直接看到 traceback(去掉 try 即可)
        for t in tasks:
            r = _eval_one(t)
            results[r["index"]] = r
            _log(r)

    return results  # type: ignore[return-value]


if __name__ == "__main__":
    # 简单的 CLI:从 JSONL 文件读边界(每行一个 JSON 字符串),把结果写出去
    import argparse

    ap = argparse.ArgumentParser(description="VMEC + problem2 batch evaluator")
    ap.add_argument("--boundaries", required=True,
                    help="JSONL 文件,每行一条 submission 记录:"
                         "{\"tag\":\"problem2\",\"nfp\":N,\"boundary\":{r_cos,z_sin}}")
    ap.add_argument("--out", required=True, help="输出 JSON 文件路径")
    ap.add_argument("--parallel", action="store_true",
                    help="启用多进程并行评估(默认串行)")
    ap.add_argument("--workers", type=int, default=16,
                    help="并行模式下的进程数")
    args = ap.parse_args()

    # 解析 submission.jsonl / submission2.jsonl 这种格式:
    #   每行: {"tag": "problem2", "nfp": <int>, "boundary": {"r_cos": [...], "z_sin": [...]}}
    # SurfaceRZFourier 需要的字段比 boundary 字段多,所以这里补齐:
    #   - 把顶层 nfp 注入 boundary.n_field_periods
    #   - r_sin / z_cos 默认 None(stellarator 对称)
    #   - is_stellarator_symmetric 默认 True
    # 同时也兼容裸 JSON 字符串 / 已经是完整 SurfaceRZFourier dict 的情况。
    boundaries: list[str] = []
    with open(args.boundaries) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if isinstance(obj, str):
                # 已经是 SurfaceRZFourier 的 JSON 字符串,直接收下
                boundaries.append(obj)
                continue

            # tag 只是标记,这里全部保留,不做任何筛选

            # 取出真正的边界 dict(submission 格式) 或就把整条 dict 当边界(老格式)
            b = obj["boundary"] if isinstance(obj, dict) and "boundary" in obj else obj

            # 补齐 SurfaceRZFourier 需要的字段
            b.setdefault("n_field_periods", int(obj["nfp"]) if "nfp" in obj else b.get("n_field_periods"))
            b.setdefault("is_stellarator_symmetric", True)
            b.setdefault("r_sin", None)
            b.setdefault("z_cos", None)

            boundaries.append(json.dumps(b))

    t0 = time.time()
    res = evaluate_boundaries(boundaries,
                              parallel=args.parallel,
                              workers=args.workers)
    dt = time.time() - t0

    # 简要统计输出,真正的明细写到 --out
    ok = sum(1 for r in res if r["success"])
    print(f"[vmec_eval] done: {ok}/{len(res)} ok, "
          f"mode={'parallel' if args.parallel else 'serial'}, "
          f"elapsed={dt:.1f}s")
    with open(args.out, "w") as f:
        json.dump(res, f, default=float, indent=2)
