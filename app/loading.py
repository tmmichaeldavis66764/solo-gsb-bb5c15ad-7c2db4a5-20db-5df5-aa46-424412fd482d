"""已保存配平方案的逐管安全装载次序规划(只读, 不改变求解与保存格式)。

问题模型
--------
方案确定后, n 支试管各自的最终槽位 θ_k 与标称质量 m_k 固定。实际装转子时
从**空转子**开始, 每步只装入**一支**试管(装入其方案槽位, 最终槽位不变),
经过 n 步达到方案的完整布局。第 t 步装入后转子上的试管集合为 S_t, 中间
状态的一阶不平衡量为

    U₁(S_t) = R · Σ_{k∈S_t} m_k · e^{iθ_k}        (g·mm)

全过程峰值 max_t |U₁(S_t)| 取决于装载次序: 次序不当会让中间状态的合力
很大(例如先装一支无对置的重管), 次序得当则始终很小(成对对消)。本模块
精确求使**全过程峰值 |U₁| 最小**的装载次序。

精确求序(子集动态规划)
--------------------
n ≤ 16, 子集状态 2^n ≤ 65536, 用子集动态规划**精确**求解(不做贪心或近似):

    v_k      = R·m_k·e^{iθ_k}                       各管质径积向量
    U[mask]  = Σ_{k∈mask} v_k                       子集向量和(递推 O(2^n))
    dp[mask] = min_{j∈mask} max(dp[mask∖{j}], |U[mask]|)
               —— 以任意次序装完 mask 中试管的最小过程峰值, dp[∅] = 0
    h[mask]  = min_{j∉mask} max(|U[mask∪{j}]|, h[mask∪{j}])
               —— 从 mask 出发装完剩余试管的最小后续峰值, h[全集] = 0

最小全过程峰值 A = dp[全集]。达到最小峰值的次序条数在"子集模长不超过
A"的状态格中数路径得到(一条次序的峰值 ≤ A 当且仅当它经过的每个子集
的模长都 ≤ A)。

**同一份子集向量模长**: 最小峰值 A、最优次序计数、字典序裁决与逐步明细
全部取自同一组 U[mask]/|U[mask]|(dp/h 只是这些模长的 min/max 选择),
因此返回峰值与所选路径的逐步累计峰值**严格相等**, 不会出现"汇总峰值
与逐步峰值不一致"的口径分裂。

最优性的严格比较(不用相对容差)
------------------------------
最优性(某子集/某次序是否达到最小峰值)按**计算值严格比较**, 不使用
1e-9 相对容差把较差路径合并为"并列"——例如两管对径、质量
50.00000001/50 g 时, 先装重管的过程峰值 5000.000001 严格差于先装轻管
的 5000, 后者是唯一最优次序(计数为 1), 不得按相对容差误并为 2 条。
由于 dp/h/计数/裁决使用的都是同一份子集模长的 min/max 选择, 严格比较
是精确的: 真正等值(计算值完全相等)的路径仍全部计入最优次序条数。

并列裁决
--------
达到最小峰值的次序可能不止一条, 此时按**试管编号序列升序**(字典序)裁决:
从第一步起逐位尝试剩余试管中编号最小者 j, 若

    max(当前前缀峰值, |U[T∪{j}]|, h[T∪{j}]) ≤ A        (严格比较, 无容差)

则选定之, 再固定位次继续裁决下一位。

限值判定
--------
用户可填写装载阶段一阶不平衡量限值(g·mm): 逐步比较当步 |U₁| 与限值并
列出超限步骤; 所选路径的累计峰值不超过限值判定为合格, 否则给出超出量。
超限步骤、合格结论与超出量均基于**所选路径**的逐步模长; 限值比较只额外
容许固定的绝对浮点噪声底线 1e-9 g·mm(吸收角度/向量求和的舍入噪声, 与
app/weighing.py、app/evacuation.py 的硬限值口径一致), **不**按限值或
峰值量级放大成相对容差。

拒绝条件(HTTP 400)
------------------
- 方案没有已放置的试管;
- 已放置试管数超过 16(子集动态规划状态数为 2^n, 明确拒绝);
- 摆放明细中试管编号/槽位重复、质量非正、缺少槽位角度;
- 装载阶段限值为负。
"""
from __future__ import annotations

import math
from typing import Any, Optional

# 子集动态规划的状态数上限: 2^16 = 65536
MAX_TUBES = 16
# 限值判定(超限/合格)只容许的固定绝对浮点噪声底线 1e-9 g·mm
# (吸收角度/向量求和的舍入噪声, 与 app/weighing.py、app/evacuation.py
# 的硬限值口径一致); 最优性比较不用任何容差, 严格按计算值
_HARD_LIMIT_TOL = 1e-9


def plan_loading_sequence(plan: dict[str, Any],
                          u1_limit_g_mm: Optional[float] = None) -> dict[str, Any]:
    """对已保存方案精确规划逐管安全装载次序(只读)。

    参数
    ----
    plan: GET /api/plan/{id} 返回结构的方案(含 id/name/request/result);
        仅读取, 不修改。
    u1_limit_g_mm: 装载阶段一阶不平衡量限值(g·mm), 可空; 非空时必须非负。

    返回
    ----
    含最优装载次序、最小全过程峰值、达到最小峰值的次序条数、逐步明细
    (装入管号/槽位/当步 |U₁|/累计峰值/超限标志)与限值判定的字典。
    """
    plan_id = plan.get("id")

    limit: Optional[float] = None
    if u1_limit_g_mm is not None:
        limit = float(u1_limit_g_mm)
        if not math.isfinite(limit):
            raise ValueError("装载阶段限值必须为有限数值。")
        if limit < 0.0:
            raise ValueError("装载阶段限值不能为负。")

    result = plan.get("result") or {}
    cfg = result.get("config") or {}
    radius = cfg.get("radius_mm")
    if radius is None:
        radius = (plan.get("request") or {}).get("radius_mm", 100.0)
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("方案中的转子半径必须为正数。")

    used = [p for p in (result.get("placements") or [])
            if p.get("slot") is not None]
    if not used:
        raise ValueError(
            f"方案 {plan_id} 没有已放置的试管, 无法进行装载规划。"
        )
    if len(used) > MAX_TUBES:
        raise ValueError(
            f"方案 {plan_id} 有 {len(used)} 支已放置试管, 超过逐管装载规划"
            f"上限 {MAX_TUBES} 支(子集动态规划状态数为 2^n), 明确拒绝。"
        )

    tubes: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_slots: set[int] = set()
    for p in used:
        tid = int(p["tube_id"])
        if tid in seen_ids:
            raise ValueError(f"摆放明细中试管 {tid} 重复。")
        seen_ids.add(tid)
        slot = int(p["slot"])
        if slot in seen_slots:
            raise ValueError(f"摆放明细中槽位 {slot} 被重复占用。")
        seen_slots.add(slot)
        mass = float(p["mass"])
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"试管 {tid} 的标称质量必须为正数。")
        angle = p.get("angle")
        if angle is None:
            raise ValueError(f"试管 {tid} 的摆放结果缺少槽位角度。")
        tubes.append({
            "tube_id": tid,
            "slot": slot,
            "angle_deg": float(angle),
            "mass": mass,
        })
    # 按试管编号升序: DP 中的位序与并列裁决的尝试顺序都以此为依据
    tubes.sort(key=lambda t: t["tube_id"])
    n = len(tubes)

    # 各管质径积向量 v_k = R·m_k·e^{iθ_k} (g·mm)
    vecs: list[complex] = []
    for t in tubes:
        a = math.radians(t["angle_deg"])
        vecs.append(complex(t["mass"] * radius * math.cos(a),
                            t["mass"] * radius * math.sin(a)))

    size = 1 << n
    full = size - 1

    # U[mask]: 子集向量和; mag[mask] = |U[mask]|。递推: 取最低位。
    # 最小峰值、最优次序计数、并列裁决与逐步明细统一使用这一份
    # 子集向量/模长, 保证四处结果严格一致(逐步明细不再另行累加)
    mag = [0.0] * size
    u_vec = [0j] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        u = u_vec[mask ^ lsb] + vecs[i]
        u_vec[mask] = u
        mag[mask] = abs(u)

    # dp[mask]: 以任意次序装完 mask 中试管的最小过程峰值
    dp = [0.0] * size
    for mask in range(1, size):
        m = mag[mask]
        best = math.inf
        sub = mask
        while sub:
            lsb = sub & (-sub)
            prev = mask ^ lsb
            cand = dp[prev] if dp[prev] > m else m
            if cand < best:
                best = cand
            sub ^= lsb
        dp[mask] = best

    peak = dp[full]

    # 达到最小峰值的次序条数: 一条次序的峰值 == A 当且仅当它经过的每个
    # 子集的模长都 ≤ A(严格按计算值比较, 不用相对容差合并较差路径;
    # 计算值真正等值的路径仍全部计数)
    cnt = [0] * size
    cnt[0] = 1
    for mask in range(1, size):
        if mag[mask] > peak:
            continue
        c = 0
        sub = mask
        while sub:
            lsb = sub & (-sub)
            c += cnt[mask ^ lsb]
            sub ^= lsb
        cnt[mask] = c

    # h[mask]: 从 mask 出发装完剩余试管的最小后续峰值(倒序递推, 依赖超集)
    h = [0.0] * size
    for mask in range(full - 1, -1, -1):
        best = math.inf
        for j in range(n):
            bit = 1 << j
            if mask & bit:
                continue
            nxt = mask | bit
            cand = mag[nxt] if mag[nxt] > h[nxt] else h[nxt]
            if cand < best:
                best = cand
        h[mask] = best

    # 并列裁决: 正向逐位选取仍能达到最小峰值的、编号最小的试管
    # (与计数同一口径: 严格按计算值 ≤ A, 不用相对容差)
    order: list[int] = []
    cur = 0
    prefix_peak = 0.0
    while cur != full:
        for j in range(n):
            bit = 1 << j
            if cur & bit:
                continue
            nxt = cur | bit
            cand = prefix_peak
            if mag[nxt] > cand:
                cand = mag[nxt]
            if h[nxt] > cand:
                cand = h[nxt]
            if cand <= peak:
                order.append(j)
                cur = nxt
                if mag[nxt] > prefix_peak:
                    prefix_peak = mag[nxt]
                break
        else:  # pragma: no cover - 理论上不发生
            raise AssertionError("装载次序构造失败: 动态规划状态不一致。")
    # 构造出的路径峰值与 DP 求得的最小峰值是同一份子集模长的 min/max
    # 选择, 二者必然严格相等
    assert prefix_peak == peak

    # 逐步明细: 装入管号、槽位、当步 |U₁|、累计峰值与超限标志。
    # 当步向量/模长直接取同一份子集结果 u_vec/mag(按所选路径的前缀
    # 子集索引), 不另行累加, 因此累计峰值与返回峰值严格相等; 超限
    # 判定只容许固定 1e-9 g·mm 绝对噪声底线
    steps: list[dict[str, Any]] = []
    exceeded: list[int] = []
    cur = 0
    run_peak = 0.0
    for seq, j in enumerate(order, start=1):
        t = tubes[j]
        cur |= 1 << j
        u = u_vec[cur]
        ux, uy = u.real, u.imag
        step_mag = mag[cur]
        if step_mag > run_peak:
            run_peak = step_mag
        exceeds: Optional[bool] = None
        if limit is not None:
            exceeds = step_mag > limit + _HARD_LIMIT_TOL
            if exceeds:
                exceeded.append(seq)
        steps.append({
            "seq": seq,
            "tube_id": t["tube_id"],
            "slot": t["slot"],
            "angle_deg": t["angle_deg"],
            "mass_g": t["mass"],
            "u1_mag_g_mm": step_mag,
            "u1_x_g_mm": ux,
            "u1_y_g_mm": uy,
            "u1_angle_deg": math.degrees(math.atan2(uy, ux)) % 360.0,
            "peak_so_far_g_mm": run_peak,
            "exceeds_limit": exceeds,
        })

    # 返回峰值即所选路径的累计峰值; 合格结论与超出量同样基于该路径
    assert run_peak == peak
    within: Optional[bool] = None
    if limit is not None:
        within = run_peak <= limit + _HARD_LIMIT_TOL

    return {
        "method": (
            "子集动态规划(2^n, n≤16)精确求全过程峰值 |U₁| 最小的装载次序"
            "(空转子起步, 每步装入一支试管, 最终槽位与方案一致); "
            "最优性按计算值严格比较(不用相对容差合并较差路径), "
            "并列最优按试管编号序列升序裁决"
        ),
        "tube_count": n,
        "radius_mm": radius,
        "peak_u1_g_mm": peak,
        "final_u1_g_mm": mag[full],
        "optimal_order_count": cnt[full],
        "tie_broken": cnt[full] > 1,
        "order": [tubes[j]["tube_id"] for j in order],
        "steps": steps,
        "u1_limit_g_mm": limit,
        "within_limit": within,
        "exceeded_steps": exceeded,
        "exceeded_count": len(exceeded),
        "excess_g_mm": (None if limit is None else max(0.0, run_peak - limit)),
        "margin_g_mm": (None if limit is None else max(0.0, limit - run_peak)),
        "plan": {
            "id": plan.get("id"),
            "name": plan.get("name"),
            "created_at": plan.get("created_at"),
        },
        "note": "逐管装载次序为只读规划结果, 未修改任何已保存方案。",
    }
