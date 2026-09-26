"""已保存配平方案的现场装管复核(只读, 不改变求解与保存格式)。

问题模型
--------
方案确定后: n 支试管各有标称质量 m_k 与方案槽位 s_k。现场装管后, 操作者
按**启用槽**录入实际有管槽位及天平实测质量 r_j(精度 0.01 g), 并给出一个
**统一允许偏差** e(g)。读数 j 可能对应试管 k 当且仅当

    |m_k - r_j| <= e

复核问题即: 只在**质量差不超过偏差**的边上, 求方案试管与实测读数之间的
**一对一匹配**(二分图完美匹配), 并以**总绝对质量差最小**为准。

精确十进制比较
--------------
标称质量、实测质量与允许偏差一律按其**字符串表示的十进制值**构造 Decimal
精确比较: 标称质量**不舍入到 0.01 g**, 10.0000000004 与 10.0000000007 是
两个不同的值。因此不使用任何浮点相对容差——总绝对质量差相差 1e-10 g 的两个
匹配不会被判为"等代价", 边判定与最优判定都是精确的十进制大小/相等关系。
仅实测读数仍须为 0.01 g 的整数倍(天平读数精度), 这是对录入值本身的校验,
不影响标称质量的精度。

精确求解(子集动态规划)
--------------------
读数最多 12 份, 故完整匹配时方案管数 n <= 12, 用子集 DP **精确**求最小
总绝对质量差(不做贪心、不取近似、不枚举全部排列):

    F[i][S]  前 i 支试管(按管号升序)匹配读数集 S 的最小总绝对质量差(Decimal)
    C[i][S]  达到 F[i][S] 的匹配条数(任意精度整数, 与 F 在同一遍 DP 内递推)
    G[i][S]  第 i 支起的试管匹配读数集 S 的最小总绝对质量差(Decimal)

最小总绝对质量差 C* = F[n][全集]。边 (k, j) 出现在**某个最优匹配**中,
当且仅当存在不含 j、且大小恰为 k 的读数集 S 使

    F[k][S] + |m_k - r_j| + G[k+1][全集 \\ S \\ {j}] = C*   (十进制精确相等)

同一组 F/G 同时给出两个方向的候选: 每个实测读数 j 在全部最优匹配中可能
对应的试管, 以及每支方案试管 k 在全部最优匹配中可能落入的实测槽, 不枚举
全部最优匹配(退化情形下可达 n! 条; 条数由 C[n][全集] 直接得到):

- **候选唯一**: 该槽读数在全部最优匹配中都对应同一支试管 —— 按原方案
  槽位判定"正确"(读数槽即该管方案槽)或"错槽"(读数槽不是该管方案槽);
- **候选多解**: 标为"身份不确定" —— 不能任取一个最优匹配断言错放。

读数数量与方案管数不等、或不存在完整匹配时, 明确说明原因(并给出最多
可匹配对数, 以及在允许偏差内无任何兼容读数的试管/无任何兼容试管的
读数清单)。

拒绝条件(ValueError → HTTP 400)
-------------------------------
- 实测读数少于 1 份或超过 12 份; 槽位重复、在方案中不存在或为禁用槽;
- 实测质量非正、非有限, 或不是 0.01 g 的整数倍(读数精度 0.01 g);
- 统一允许偏差为负或非有限;
- 方案没有已放置试管、缺少槽位状态; 摆放明细编号/槽位重复、质量非正。
"""
from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

# 读数份数上限: 子集 DP 状态数为 2^n, n <= 12 时 4096, 瞬时完成
MAX_READINGS = 12
# 天平读数精度(g): 实测质量必须是该步长的整数倍(十进制精确判定)
MASS_STEP = Decimal("0.01")

_STATUS_LABELS = {
    "correct": "正确",
    "wrong_slot": "错槽",
    "uncertain": "身份不确定",
}


def _as_decimal(value: Any, what: str) -> Decimal:
    """把质量/偏差入参按**字符串表示的十进制值**精确转为 Decimal。

    float 先经 repr 取最短可回显十进制串(如 10.0000000004 对应的字符串),
    从而保留其全部可分辨十进制值, 不引入二进制浮点误差; 非有限 float、
    NaN/Infinity 串或不支持的类型一律拒绝。bool 不按数值接受。
    """
    if isinstance(value, Decimal):
        d = value
    elif isinstance(value, bool):
        raise ValueError(f"{what}必须是数值。")
    elif isinstance(value, int):
        d = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{what}必须为有限数值。")
        d = Decimal(repr(value))
    elif isinstance(value, str):
        try:
            d = Decimal(value.strip())
        except (InvalidOperation, AttributeError, ValueError):
            raise ValueError(f"{what}必须是十进制数值。") from None
        if not d.is_finite():
            raise ValueError(f"{what}必须为有限数值。")
    else:
        raise ValueError(f"{what}必须是十进制数值。")
    return d


def _dec_str(d: Decimal) -> str:
    """十进制值的紧凑展示串(去掉无意义尾零), 用于原因说明。"""
    s = str(d)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def recheck_tube_loading(plan: dict[str, Any],
                         readings: list[dict[str, Any]],
                         tolerance_g: Any) -> dict[str, Any]:
    """对已保存方案做现场装管复核(只读)。

    参数
    ----
    plan: GET /api/plan/{id} 返回结构的方案(含 id/name/request/result),
        仅读取, 不修改。
    readings: 实测读数列表, 每项含 slot(实际有管的启用槽)与 mass(实测
        质量 g, 0.01 精度); 1~12 份, 槽位不得重复。
    tolerance_g: 统一允许偏差(g), 非负有限数; 只有 |标称-实测| <= 偏差
        的试管-读数对才允许匹配。

    返回
    ----
    含匹配可行性、最小总绝对质量差、最优匹配条数、逐实测槽候选试管与
    判定(正确/错槽/身份不确定)、逐方案试管可能槽位及原因说明的字典。
    """
    plan_id = plan.get("id")

    # ---- 统一允许偏差(十进制精确值) ----
    tol = _as_decimal(tolerance_g, "统一允许偏差")
    if tol < 0:
        raise ValueError("统一允许偏差不能为负。")
    # 输出回显: 数值类型原样回显(其十进制字符串即精确值), 其他类型转 float
    tol_out = (tolerance_g if isinstance(tolerance_g, (int, float))
               and not isinstance(tolerance_g, bool) else float(tol))

    # ---- 方案试管(标称质量与方案槽位) ----
    result = plan.get("result") or {}
    used = [p for p in (result.get("placements") or [])
            if p.get("slot") is not None]
    if not used:
        raise ValueError(
            f"方案 {plan_id} 没有已放置的试管, 无法进行装管复核。"
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
        mass_dec = _as_decimal(p["mass"], f"试管 {tid} 的标称质量")
        if mass_dec <= 0:
            raise ValueError(f"试管 {tid} 的标称质量必须为正数。")
        tubes.append({
            "tube_id": tid,
            "plan_slot": slot,
            "mass": mass_dec,                       # 精确十进制值(不舍入)
            "mass_raw": p["mass"],                  # 原始入参, 用于输出回显
        })
    # 管号升序: DP 位序与结果展示都以此为依据
    tubes.sort(key=lambda t: t["tube_id"])
    n = len(tubes)

    # ---- 槽位状态(读数槽必须是方案中的启用槽) ----
    slot_state = result.get("slot_state") or []
    if not slot_state:
        raise ValueError(
            f"方案 {plan_id} 缺少槽位状态信息, 无法进行装管复核。"
        )
    slot_info: dict[int, dict[str, Any]] = {}
    for item in slot_state:
        idx = int(item["slot"])
        slot_info[idx] = {
            "angle": float(item["angle"]),
            "enabled": bool(item.get("enabled", True)),
        }

    # ---- 实测读数(0.01 g 精度, 十进制精确判定) ----
    if not isinstance(readings, (list, tuple)) or not readings:
        raise ValueError("请至少录入 1 份实测读数。")
    if len(readings) > MAX_READINGS:
        raise ValueError(
            f"实测读数最多 {MAX_READINGS} 份(当前 {len(readings)} 份), 明确拒绝。"
        )
    meas: list[dict[str, Any]] = []
    seen_rslots: set[int] = set()
    for raw in readings:
        slot = int(raw["slot"])
        if slot in seen_rslots:
            raise ValueError(f"实测槽位 {slot} 重复录入。")
        seen_rslots.add(slot)
        info = slot_info.get(slot)
        if info is None:
            raise ValueError(f"实测槽位 {slot} 在方案中不存在。")
        if not info["enabled"]:
            raise ValueError(
                f"实测槽位 {slot} 在方案中为禁用槽, 不属于可装管的启用槽。"
            )
        raw_mass = raw["mass"]
        mass = _as_decimal(raw_mass, f"槽位 {slot} 的实测质量")
        if mass <= 0:
            raise ValueError(f"槽位 {slot} 的实测质量必须为正的有限数。")
        if mass % MASS_STEP != 0:
            shown = repr(raw_mass) if isinstance(raw_mass, float) else str(raw_mass)
            raise ValueError(
                f"槽位 {slot} 的实测质量 {shown} g 不符合 0.01 g 读数精度"
                "(实测质量须为 0.01 g 的整数倍)。"
            )
        meas.append({
            "slot": slot,
            "mass": mass,                 # 精确十进制值
            "mass_raw": raw_mass,         # 原始入参(0.01 精度, 输出回显)
            "angle": info["angle"],
        })
    # 槽号升序: 读数位序与展示顺序一致
    meas.sort(key=lambda r: r["slot"])
    m = len(meas)

    def _context() -> dict[str, Any]:
        return {
            "method": (
                "二分图一对一匹配(子集动态规划精确求最小总绝对质量差, "
                f"读数≤{MAX_READINGS} 份): 标称质量、实测质量与偏差按其字符串"
                "表示的十进制值精确比较(标称质量不舍入到 0.01 g, 无浮点容差), "
                "只在 |标称-实测| ≤ 统一允许偏差的边上匹配; 最小代价、最优计数"
                "与双向候选由同一子集 DP 精确给出; 对每个实测槽汇总所有最优"
                "匹配可能对应的试管, 唯一时按原方案槽位判定正确/错槽, 多解时"
                "标为身份不确定"
            ),
            "plan": {
                "id": plan.get("id"),
                "name": plan.get("name"),
                "created_at": plan.get("created_at"),
            },
            "tolerance_g": tol_out,
            "plan_tube_count": n,
            "reading_count": m,
            "complete": False,
            "reason": None,
            "total_abs_deviation_g": None,
            "optimal_matching_count": None,
            "max_matched_pairs": None,
            "tubes_without_candidate": [],
            "readings_without_candidate": [],
            "readings": [],
            "tubes": [],
            "summary": None,
            "note": "现场装管复核为只读结果, 未修改任何已保存方案。",
        }

    # ---- 读数数量与方案管数不等: 说明原因, 不做匹配 ----
    if m != n:
        out = _context()
        out["reason"] = (
            f"实测读数 {m} 份与方案试管数 {n} 支不等: 数量不一致时无法建立"
            "一一对应的完整匹配。请核对是否漏录、多录或装管数量本身有误, "
            "修正后重新复核。"
        )
        return out

    # ---- 匹配边: 仅十进制精确质量差不超过统一允许偏差的试管-读数对 ----
    cost: list[list[Optional[Decimal]]] = [[None] * n for _ in range(n)]
    for k, t in enumerate(tubes):
        for j, r in enumerate(meas):
            d = abs(t["mass"] - r["mass"])
            if d <= tol:
                cost[k][j] = d

    size = 1 << n
    full = size - 1

    # 子集大小(位 1 的个数): 候选边判定只在大小恰为 k 的子集上进行
    masks_by_pop: list[list[int]] = [[0]]
    for _ in range(n):
        masks_by_pop.append([])
    for mask in range(1, size):
        masks_by_pop[mask.bit_count()].append(mask)

    # F[i][mask]: 前 i 支试管(管号升序)匹配读数集 mask 的最小总绝对质量差;
    # C[i][mask]: 达到 F[i][mask] 的匹配条数(任意精度整数)。
    # 最小代价与最优计数在同一遍子集 DP 内统一递推, 不分两遍、不枚举排列。
    F: list[list[Optional[Decimal]]] = [[None] * size for _ in range(n + 1)]
    C = [[0] * size for _ in range(n + 1)]
    F[0][0] = Decimal(0)
    C[0][0] = 1
    for i in range(1, n + 1):
        fi, fp, ci = F[i], F[i - 1], cost[i - 1]
        cnti, cntp = C[i], C[i - 1]
        for mask in range(size):
            best: Optional[Decimal] = None
            ways = 0
            sub = mask
            while sub:
                lsb = sub & (-sub)
                j = lsb.bit_length() - 1
                prev = fp[mask ^ lsb]
                c = ci[j]
                if prev is not None and c is not None:
                    v = prev + c
                    if best is None or v < best:
                        best = v
                        ways = cntp[mask ^ lsb]
                    elif v == best:  # 十进制精确相等, 无浮点容差
                        ways += cntp[mask ^ lsb]
                sub ^= lsb
            fi[mask] = best
            cnti[mask] = ways

    # G[i][mask]: 第 i 支起的试管匹配读数集 mask 的最小总绝对质量差
    G: list[list[Optional[Decimal]]] = [[None] * size for _ in range(n + 1)]
    G[n][0] = Decimal(0)
    for i in range(n - 1, -1, -1):
        gi, gn, ci = G[i], G[i + 1], cost[i]
        for mask in range(size):
            best = None
            sub = mask
            while sub:
                lsb = sub & (-sub)
                j = lsb.bit_length() - 1
                nxt = gn[mask ^ lsb]
                c = ci[j]
                if nxt is not None and c is not None:
                    v = nxt + c
                    if best is None or v < best:
                        best = v
                sub ^= lsb
            gi[mask] = best

    # ---- 不存在完整匹配: 说明原因 ----
    total = F[n][full]
    if total is None:
        # 最大可匹配对数仅在不可行时需要(可行时恒为 n), 惰性计算:
        # H[i][mask] = 前 i 支试管与读数集 mask 的子集最多可匹配对数
        H = [[0] * size for _ in range(n + 1)]
        for i in range(1, n + 1):
            hi, hp, ci = H[i], H[i - 1], cost[i - 1]
            for mask in range(size):
                best = hp[mask]  # 试管 i-1 不参与匹配
                sub = mask
                while sub:
                    lsb = sub & (-sub)
                    j = lsb.bit_length() - 1
                    if ci[j] is not None:
                        v = hp[mask ^ lsb] + 1
                        if v > best:
                            best = v
                    sub ^= lsb
                hi[mask] = best
        max_pairs = H[n][full]

        out = _context()
        out["max_matched_pairs"] = max_pairs
        no_cand_tubes = [
            t for k, t in enumerate(tubes)
            if all(cost[k][j] is None for j in range(n))
        ]
        no_cand_readings = [
            r for j, r in enumerate(meas)
            if all(cost[k][j] is None for k in range(n))
        ]
        out["tubes_without_candidate"] = [
            {"tube_id": t["tube_id"], "nominal_mass_g": t["mass_raw"],
             "plan_slot": t["plan_slot"]}
            for t in no_cand_tubes
        ]
        out["readings_without_candidate"] = [
            {"slot": r["slot"], "measured_mass_g": r["mass_raw"]}
            for r in no_cand_readings
        ]
        parts = [
            f"在统一允许偏差 ±{_dec_str(tol)} g 内不存在完整的一对一匹配: "
            f"最多只能匹配 {max_pairs}/{n} 对。"
        ]
        if no_cand_tubes:
            txt = "、".join(
                f"试管 {t['tube_id']}(标称 {str(t['mass'])} g)"
                for t in no_cand_tubes
            )
            parts.append(f"{txt} 在偏差内没有任何可对应的实测读数。")
        if no_cand_readings:
            txt = "、".join(
                f"槽 #{r['slot']}({_dec_str(r['mass'])} g)"
                for r in no_cand_readings
            )
            parts.append(f"实测读数 {txt} 在偏差内没有任何可对应的方案试管。")
        parts.append(
            "可能原因: 实际装管数量或位置与方案不符、实测质量超出允许偏差, "
            "或统一允许偏差设置过小; 请核对现场装管与读数后重新复核。"
        )
        out["reason"] = "".join(parts)
        return out

    # ---- 最优匹配条数直接取自统一 DP(任意精度整数) ----
    optimal_count = C[n][full]

    # ---- 逐边判定(双向候选): (k,j) 是否出现在某个最优匹配中 ----
    # 同一组 F/G 同时汇总: 读数 j 的候选试管 cand_of_reading, 与试管 k 的
    # 候选实测槽 cand_of_tube。等式按十进制精确比较, 无任何容差。
    cand_of_reading: list[list[int]] = [[] for _ in range(n)]  # j -> [k]
    cand_of_tube: list[list[int]] = [[] for _ in range(n)]     # k -> [j]
    for k in range(n):
        fk, gk1, ck = F[k], G[k + 1], cost[k]
        for j in range(n):
            c = ck[j]
            if c is None:
                continue
            bit = 1 << j
            hit = False
            # 前 k 支试管占用大小恰为 k、不含 j 的读数集 S; 试管 k 取 j,
            # 其后的试管取 全集\S\{j}(按 popcount 预分组, 直接枚举)
            for mask in masks_by_pop[k]:
                if mask & bit:
                    continue
                a = fk[mask]
                if a is None:
                    continue
                b = gk1[full ^ mask ^ bit]
                if b is None:
                    continue
                if a + c + b == total:
                    hit = True
                    break
            if hit:
                cand_of_reading[j].append(k)
                cand_of_tube[k].append(j)

    # ---- 逐实测槽汇总与判定 ----
    reading_rows: list[dict[str, Any]] = []
    for j, r in enumerate(meas):
        cands = [{
            "tube_id": tubes[k]["tube_id"],
            "nominal_mass_g": tubes[k]["mass_raw"],
            "plan_slot": tubes[k]["plan_slot"],
            "abs_deviation_g": float(cost[k][j]),
        } for k in cand_of_reading[j]]
        if len(cands) == 1:
            k = cand_of_reading[j][0]
            t = tubes[k]
            matched: Optional[int] = t["tube_id"]
            if t["plan_slot"] == r["slot"]:
                status = "correct"
                detail = (
                    f"全部最优匹配中该槽读数唯一对应试管 {t['tube_id']}"
                    f"(标称 {str(t['mass'])} g), 与其方案槽位一致。"
                )
            else:
                status = "wrong_slot"
                detail = (
                    f"全部最优匹配中该槽读数唯一对应试管 {t['tube_id']}"
                    f"(标称 {str(t['mass'])} g), 但试管 {t['tube_id']} 的方案槽位"
                    f"为 #{t['plan_slot']}: 该管未按方案装管(错槽)。"
                )
        else:
            status = "uncertain"
            matched = None
            ids = "、".join(str(c["tube_id"]) for c in cands)
            detail = (
                f"最优匹配共 {optimal_count} 种, 该槽读数可能对应试管 {ids}: "
                "身份不确定, 不能任取一解断言错放。"
            )
        reading_rows.append({
            "slot": r["slot"],
            "angle_deg": r["angle"],
            "measured_mass_g": r["mass_raw"],
            "candidates": cands,
            "candidate_count": len(cands),
            "status": status,
            "status_label": _STATUS_LABELS[status],
            "matched_tube_id": matched,
            "detail": detail,
        })

    # ---- 逐方案试管汇总(对称信息, 供交叉核对) ----
    tube_rows: list[dict[str, Any]] = []
    for k, t in enumerate(tubes):
        slots_j = sorted(meas[j]["slot"] for j in cand_of_tube[k])
        if slots_j == [t["plan_slot"]]:
            status = "correct"
            detail = "全部最优匹配中该管都位于其方案槽位。"
        elif t["plan_slot"] not in slots_j:
            status = "wrong_slot"
            txt = "、#".join(str(s) for s in slots_j)
            detail = (
                f"全部最优匹配中该管均不在方案槽位 #{t['plan_slot']}"
                f"(可能位于实测槽 #{txt}): 未按方案装管。"
            )
        else:
            status = "uncertain"
            txt = "、#".join(str(s) for s in slots_j)
            detail = (
                f"最优匹配不唯一, 该管可能位于实测槽 #{txt}"
                "(含其方案槽位): 不能断言错放。"
            )
        tube_rows.append({
            "tube_id": t["tube_id"],
            "nominal_mass_g": t["mass_raw"],
            "plan_slot": t["plan_slot"],
            "candidate_slots": slots_j,
            "status": status,
            "status_label": _STATUS_LABELS[status],
            "detail": detail,
        })

    n_correct = sum(1 for r in reading_rows if r["status"] == "correct")
    n_wrong = sum(1 for r in reading_rows if r["status"] == "wrong_slot")
    n_uncertain = n - n_correct - n_wrong

    out = _context()
    out.update({
        "complete": True,
        "total_abs_deviation_g": float(total),
        "optimal_matching_count": optimal_count,
        "max_matched_pairs": n,
        "readings": reading_rows,
        "tubes": tube_rows,
        "summary": {
            "correct": n_correct,
            "wrong_slot": n_wrong,
            "uncertain": n_uncertain,
        },
    })
    match_txt = (
        "最优匹配唯一" if optimal_count == 1
        else f"最优匹配共 {optimal_count} 种"
    )
    total_txt = _dec_str(total)
    if n_wrong == 0 and n_uncertain == 0:
        out["reason"] = (
            f"{n} 个实测槽全部唯一对应且与方案槽位一致: 装管全部正确"
            f"(最小总绝对质量差 {total_txt} g, {match_txt})。"
        )
    elif n_wrong > 0:
        slots = "、#".join(
            str(r["slot"]) for r in reading_rows if r["status"] == "wrong_slot"
        )
        unc_txt = (
            f"; 另有 {n_uncertain} 个实测槽身份不确定, 不能断言错放"
            if n_uncertain else ""
        )
        out["reason"] = (
            f"发现 {n_wrong} 处错槽: 实测槽 #{slots} 唯一对应的试管与其方案"
            f"槽位不符{unc_txt}(最小总绝对质量差 {total_txt} g, {match_txt})。"
        )
    else:
        slots = "、#".join(
            str(r["slot"]) for r in reading_rows if r["status"] == "uncertain"
        )
        out["reason"] = (
            f"{n_uncertain} 个实测槽(#{slots})身份不确定: 最优匹配不唯一, "
            "这些槽的读数在偏差内可对应多支试管, 不能任取一解断言错放"
            f"(最小总绝对质量差 {total_txt} g, {match_txt})。"
        )
    return out
