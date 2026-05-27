"""
百合生长模型数据监测智能体平台 (Streamlit 版) - V2 布局优化
功能：数据导入 | 阈值评估 | 分析交互 | 测试数据 | 数据总库
排版：上下流式布局，卡片分隔，减少左右拥挤
"""

import streamlit as st

# ============================================================
# Streamlit 页面配置（必须是第一个 Streamlit 命令）
# ============================================================
st.set_page_config(
    page_title="百合生长模型数据监测智能体平台",
    page_icon="🌷",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
import zipfile
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import stats

# ============================================================
# 全局路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
QUALIFIED_DIR = os.path.join(DATA_DIR, "qualified")
UNQUALIFIED_DIR = os.path.join(DATA_DIR, "unqualified")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
TEST_DIR = os.path.join(DATA_DIR, "test")

for d in [RAW_DIR, QUALIFIED_DIR, UNQUALIFIED_DIR, METADATA_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# 加载物理极限表
PHYSICAL_LIMITS_PATH = os.path.join(CONFIG_DIR, "physical_limits.json")
with open(PHYSICAL_LIMITS_PATH, "r", encoding="utf-8") as f:
    PHYSICAL_LIMITS = json.load(f)

# ============================================================
# CSS 样式 - 卡片分隔 + 上下布局优化
# ============================================================
st.markdown("""
<style>
    /* 全局字体 */
    * { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
    
    /* 模块卡片容器 */
    .module-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 24px 28px;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    /* 主标题 */
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: #1b5e20;
        margin: 0;
        padding: 8px 0;
    }
    
    /* 模块子标题 */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #2e7d32;
        margin: 0 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #c8e6c9;
    }
    
    /* 信息提示框 */
    .hint-box {
        background: #f1f8e9;
        border-left: 4px solid #689f38;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0 16px 0;
        color: #33691e;
        font-size: 14px;
    }
    
    /* 操作按钮区域 */
    .action-area {
        background: #fafafa;
        border-radius: 10px;
        padding: 20px;
        margin: 12px 0;
    }
    
    /* 数据预览区域 */
    .preview-area {
        background: #fafafa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 16px;
    }
    
    /* 指标卡片横向排列 */
    .metric-row {
        display: flex;
        gap: 16px;
        margin: 16px 0;
    }
    
    .metric-item {
        flex: 1;
        background: #e8f5e9;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    
    .metric-num {
        font-size: 28px;
        font-weight: 700;
        color: #2e7d32;
    }
    
    .metric-text {
        font-size: 13px;
        color: #558b2f;
        margin-top: 4px;
    }
    
    /* Streamlit 原生组件覆盖 */
    div[data-testid="stHorizontalBlock"] { gap: 12px !important; }
    div[data-testid="stVerticalBlock"] { gap: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 工具函数
# ============================================================

def get_next_batch_id():
    meta_path = os.path.join(METADATA_DIR, "batch_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata["batches"]:
            last_id = max([int(b["batch_id"]) for b in metadata["batches"]])
            return f"{last_id + 1:06d}"
    return "000001"

def load_metadata():
    meta_path = os.path.join(METADATA_DIR, "batch_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"batches": [], "version": "1.0"}

def save_metadata(metadata):
    meta_path = os.path.join(METADATA_DIR, "batch_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# ============================================================
# 列名匹配 + 值验证
# ============================================================

COLUMN_MAPPING = {
    "时期": {"required": True, "keywords": ["时期", "生长时期", "阶段", "生长期", "period", "stage", "phase"]},
    "维度": {"required": True, "keywords": ["维度", "类型", "category", "dimension", "type", "类别"]},
    "指标": {"required": True, "keywords": ["指标", "参数", "indicator", "index", "parameter", "metric", "item", "变量"]},
    "数值": {"required": True, "keywords": ["数值", "值", "value", "data", "测量值", "读数"]},
    "单位": {"required": False, "keywords": ["单位", "unit", "度量单位"]},
    "时间": {"required": False, "keywords": ["时间", "日期", "time", "date", "timestamp"]},
}

PERIOD_KEYWORDS = {
    "萌芽期": ["萌芽", "发芽", "萌发"], "展叶期": ["展叶", "叶片展开"],
    "孕蕾期": ["孕蕾", "蕾期", "花蕾期", "现蕾"], "开花期": ["开花", "花期", "盛花"],
}

DIMENSION_KEYWORDS = {
    "环境": ["环境", "气象", "天气"], "土壤": ["土壤", "土质", "泥土"],
    "生长状况": ["生长", "植株", "植物", "发育"],
}

def smart_column_match(df_columns):
    mapping = {}
    matched = set()
    for standard_name, config in COLUMN_MAPPING.items():
        best_match, best_score = None, 0
        for col in df_columns:
            if col in matched: continue
            col_l = col.lower().strip()
            for kw in config["keywords"]:
                kw_l = kw.lower().strip()
                if col_l == kw_l: best_match, best_score = col, 100; break
                elif kw_l in col_l or col_l in kw_l:
                    score = 80 if len(kw_l) >= 2 else 50
                    if score > best_score: best_match, best_score = col, score
                elif col_l.startswith(kw_l[:2]) and len(col_l) <= len(kw_l) + 2:
                    score = 60
                    if score > best_score: best_match, best_score = col, score
            if best_score == 100: break
        mapping[standard_name] = best_match
        if best_match: matched.add(best_match)
    return mapping

def validate_period_value(value):
    if pd.isna(value): return False, "空值"
    v = str(value).strip()
    if v in PERIOD_KEYWORDS: return True, v
    for sp, kws in PERIOD_KEYWORDS.items():
        for kw in kws:
            if kw in v or v in kw: return True, sp
    return False, f"无法识别的时期: {v}"

def validate_dimension_value(value):
    if pd.isna(value): return False, "空值"
    v = str(value).strip()
    if v in DIMENSION_KEYWORDS: return True, v
    for sd, kws in DIMENSION_KEYWORDS.items():
        for kw in kws:
            if kw in v or v in kw: return True, sd
    return False, f"无法识别的维度: {v}"

# ============================================================
# 物理极值 + 统计异常
# ============================================================

def check_physical_limit(period, dimension, indicator, value):
    period_data = PHYSICAL_LIMITS.get("limits", {}).get(period, {})
    dim_data = period_data.get(dimension, {})
    indicator_info = dim_data.get(indicator)
    if indicator_info is None:
        for dn, items in period_data.items():
            if indicator in items: indicator_info = items[indicator]; break
    if indicator_info is None:
        return True, "无物理极限定义", {"min": None, "max": None, "unit": ""}
    min_v, max_v, unit = indicator_info.get("min"), indicator_info.get("max"), indicator_info.get("unit", "")
    limit_info = {"min": min_v, "max": max_v, "unit": unit}
    if min_v is not None and value < min_v: return False, f"物理越限: {value} < 下限({min_v})", limit_info
    if max_v is not None and value > max_v: return False, f"物理越限: {value} > 上限({max_v})", limit_info
    return True, "物理合格", limit_info

def detect_statistical_anomaly(values):
    n = len(values)
    if n < 5: return [False] * n, "样本量不足(<5)", {"n": n, "note": "跳过统计检测"}
    arr = np.array(values, dtype=float)
    skewness = stats.skew(arr)
    if abs(skewness) < 1.0:
        mu, sigma = np.mean(arr), np.std(arr, ddof=1)
        if sigma == 0: return [False] * n, "z-score(σ=0)", {"mu": mu, "sigma": 0, "note": "σ=0"}
        z_scores = np.abs((arr - mu) / sigma)
        anomalies = (z_scores > 3.0).tolist()
        return anomalies, "统计异常(z-score)", {"method": "z-score", "mu": round(float(mu), 4),
            "sigma": round(float(sigma), 4), "threshold_z": 3.0,
            "threshold_range": [round(float(mu - 3 * sigma), 4), round(float(mu + 3 * sigma), 4)],
            "n": n, "anomaly_count": sum(anomalies)}
    else:
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        anomalies = ((arr < lb) | (arr > ub)).tolist()
        return anomalies, "统计异常(IQR)", {"method": "IQR", "Q1": round(float(q1), 4),
            "Q3": round(float(q3), 4), "IQR": round(float(iqr), 4),
            "lower_bound": round(float(lb), 4), "upper_bound": round(float(ub), 4),
            "n": n, "anomaly_count": sum(anomalies)}

# ============================================================
# 侧边栏导航
# ============================================================

def sidebar_nav():
    st.sidebar.markdown("""
    <div style="text-align:center; padding:10px 0;">
        <p style="color:#e8f5e9; font-size:24px; font-weight:700; margin:0;">🌷 百合监测平台</p>
        <p style="color:rgba(232,245,233,0.6); font-size:12px; margin:4px 0 0 0;">低代码智能体系统</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    pages = {
        "import": "📥 数据导入",
        "evaluate": "🔍 阈值评估",
        "analyze": "📊 分析交互",
        "test": "🧪 测试数据",
        "database": "🗄️ 数据总库",
    }
    selected = st.sidebar.radio("导航", list(pages.keys()), format_func=lambda x: pages[x], label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("🌱 百合生长模型数据监测智能体平台 v2.0")
    return selected

# ============================================================
# 模块1：数据导入 - 上下布局
# ============================================================

def page_import():
    st.markdown('<div class="main-title">📥 数据导入智能体</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint-box">支持 .xlsx / .csv 格式。系统自动匹配列名（时期/维度/指标/数值/单位/时间），校验空值，生成批次号。单批次上限 <b>5000</b> 行。</div>', unsafe_allow_html=True)

    st.divider()

    # 上传区域
    uploaded = st.file_uploader("📎 选择数据文件", type=["xlsx", "csv"])

    if uploaded:
        st.info(f"📁 已选择: **{uploaded.name}** | 大小: {uploaded.size / 1024:.1f} KB")

        if st.button("📤 开始导入", type="primary", use_container_width=False):
            with st.spinner("⏳ 正在导入数据，请稍候..."):
                try:
                    if uploaded.name.endswith('.csv'):
                        df = pd.read_csv(uploaded, encoding='utf-8')
                    else:
                        df = pd.read_excel(uploaded)

                    total = len(df)
                    if total == 0: st.error("❌ 文件为空"); return
                    if total > 5000: st.error(f"❌ 数据行数 {total} 超过上限 5000 行"); return

                    col_map = smart_column_match(list(df.columns))
                    missing = [k for k, v in col_map.items() if COLUMN_MAPPING[k]["required"] and v is None]
                    if missing:
                        matched = "\n".join([f"{'✅' if v else '❌'} {k}: {v or '未匹配'}" for k, v in col_map.items()])
                        st.error(f"❌ 缺少必填列: {', '.join(missing)}\n\n匹配结果:\n{matched}"); return

                    rev_map = {v: k for k, v in col_map.items() if v}
                    df_renamed = df.rename(columns=rev_map)

                    valid_rows, error_details = [], []
                    for idx, row in df_renamed.iterrows():
                        errors = []
                        for f in ["时期", "维度", "指标", "数值"]:
                            if f in row.index and pd.isna(row[f]): errors.append(f"{f}为空")
                        if "时期" in row.index and not pd.isna(row["时期"]):
                            ok, msg = validate_period_value(row["时期"])
                            if not ok: errors.append(msg)
                            else: row = row.copy(); row["时期"] = msg
                        if "维度" in row.index and not pd.isna(row["维度"]):
                            ok, msg = validate_dimension_value(row["维度"])
                            if not ok: errors.append(msg)
                            else: row = row.copy(); row["维度"] = msg
                        if "数值" in row.index and not pd.isna(row["数值"]):
                            try: float(row["数值"])
                            except: errors.append(f"数值格式无效: {row['数值']}")
                        if errors:
                            error_details.append({"行号": int(idx) + 2, "时期": str(row.get("时期", "")),
                                "维度": str(row.get("维度", "")), "指标": str(row.get("指标", "")),
                                "数值": str(row.get("数值", "")), "错误": "; ".join(errors)})
                        else:
                            valid_rows.append(row.to_dict())

                    if not valid_rows: st.error(f"❌ 全部 {total} 行均无效"); return

                    batch_id = get_next_batch_id()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    raw_data = {"batch_id": batch_id, "import_time": timestamp,
                        "original_file": uploaded.name, "total_rows": total,
                        "valid_rows": len(valid_rows), "error_rows_count": len(error_details),
                        "column_mapping": col_map, "data": valid_rows}
                    save_json(raw_data, os.path.join(RAW_DIR, f"raw_batch_{batch_id}.json"))

                    metadata = load_metadata()
                    metadata["batches"].append({"batch_id": batch_id, "batch_name": f"批次_{batch_id}",
                        "import_time": timestamp, "original_file": uploaded.name,
                        "total_rows": total, "valid_rows": len(valid_rows),
                        "status": "imported", "is_test": False})
                    save_metadata(metadata)

                    # 导入报告
                    st.divider()
                    st.markdown('<div class="section-title">📋 导入报告</div>', unsafe_allow_html=True)

                    pass_rate = len(valid_rows) / total * 100
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("📦 批次号", batch_id)
                    c2.metric("📊 总行数", total)
                    c3.metric("✅ 有效行", len(valid_rows))
                    c4.metric("📎 合格率", f"{pass_rate:.1f}%")

                    st.markdown("##### 🔍 列名匹配结果")
                    match_df = pd.DataFrame([{"标准列名": k, "匹配列名": (v or "❌ 未匹配"), "状态": ("✅" if v else "❌")} for k, v in col_map.items()])
                    st.dataframe(match_df, use_container_width=True, hide_index=True)

                    if error_details:
                        with st.expander(f"⚠️ 错误行详情 ({len(error_details)} 条)"):
                            st.dataframe(pd.DataFrame(error_details), use_container_width=True, hide_index=True)

                    st.markdown('<div class="section-title">👁️ 数据预览（前10行）</div>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(valid_rows[:10]), use_container_width=True)

                    export_df = pd.DataFrame(valid_rows)
                    export_path = os.path.join(RAW_DIR, f"raw_batch_{batch_id}_export.xlsx")
                    export_df.to_excel(export_path, index=False)
                    with open(export_path, "rb") as f:
                        st.download_button("⬇️ 下载原始数据 (Excel)", f, file_name=f"raw_batch_{batch_id}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    st.success(f"✅ 导入完成！批次 {batch_id} 已就绪，可前往「阈值评估」进行下一步。")

                except Exception as e:
                    st.error(f"❌ 导入异常: {str(e)}")

# ============================================================
# 模块2：阈值评估 - 上下布局
# ============================================================

def page_evaluate():
    st.markdown('<div class="main-title">🔍 阈值评估与数据分流智能体</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint-box">物理极值判断 → 自适应统计异常检测（z-score / IQR 自动选择）→ 分流合格/不合格数据</div>', unsafe_allow_html=True)

    st.divider()

    metadata = load_metadata()
    imported_batches = [b for b in metadata.get("batches", []) if b.get("status") in ("imported", "evaluated")]
    imported_batches.sort(key=lambda x: x.get("import_time", ""), reverse=True)

    batch_options = {f"[{b['batch_id']}] {b.get('batch_name', '')} ({b['total_rows']}条, 状态:{b.get('status', '')})": b["batch_id"]
        for b in imported_batches[:10]}

    if not batch_options:
        st.warning("⚠️ 暂无已导入的批次，请先到「数据导入」模块上传数据"); return

    selected_label = st.selectbox("📂 选择批次", list(batch_options.keys()))
    batch_id = batch_options[selected_label]

    st.divider()

    if st.button("▶️ 开始评估", type="primary", use_container_width=True):
        with st.spinner("⏳ 正在进行阈值评估，请稍候..."):
            try:
                raw_file = os.path.join(RAW_DIR, f"raw_batch_{batch_id}.json")
                if not os.path.exists(raw_file): st.error("原始数据不存在"); return

                raw_data = load_json(raw_file)
                records = raw_data.get("data", [])
                if not records: st.error("无数据"); return

                groups = defaultdict(list)
                for idx, rec in enumerate(records):
                    groups[(rec.get("时期", "未知"), rec.get("指标", "未知"))].append({"idx": idx, "record": rec})

                qualified_records, unqualified_records = [], []
                total_phy_fail = total_stat_fail = total_pass = 0
                group_results = []

                for (period, indicator), items in sorted(groups.items()):
                    phy_passed, phy_failed = [], []
                    for it in items:
                        rec, val = it["record"], it["record"].get("数值")
                        if not isinstance(val, (int, float)):
                            try: val = float(val)
                            except:
                                unqualified_records.append({**rec, "_fail_reason": "数值格式无效", "_fail_type": "格式错误", "_batch_id": batch_id})
                                continue
                        ok, reason, limit_info = check_physical_limit(rec.get("时期", ""), rec.get("维度", ""), rec.get("指标", ""), val)
                        if ok: phy_passed.append({"record": rec, "value": val})
                        else: phy_failed.append({"record": rec, "reason": reason, "limit": limit_info}); total_phy_fail += 1

                    stat_info = {"method": "跳过", "note": "样本量不足"}
                    if len(phy_passed) >= 5:
                        passed_vals = [p["value"] for p in phy_passed]
                        anomalies, method_name, stat_info = detect_statistical_anomaly(passed_vals)
                        for i, p in enumerate(phy_passed):
                            if anomalies[i]:
                                unqualified_records.append({**p["record"], "_fail_reason": f"{method_name}: 值={p['value']}",
                                    "_fail_type": method_name, "_batch_id": batch_id}); total_stat_fail += 1
                            else: qualified_records.append(p["record"]); total_pass += 1
                    else:
                        for p in phy_passed: qualified_records.append(p["record"]); total_pass += 1

                    for f in phy_failed: unqualified_records.append({**f["record"], "_fail_reason": f["reason"],
                        "_fail_type": "物理越限", "_batch_id": batch_id})

                    group_results.append({"period": period, "indicator": indicator,
                        "dimension": items[0]["record"].get("维度", ""),
                        "total": len(items), "phy_pass": len(phy_passed), "phy_fail": len(phy_failed),
                        "method": stat_info.get("method", "跳过"), "stat_info": stat_info,
                        "limit": limit_info if phy_failed else (phy_passed[0]["record"] if phy_passed else {})})

                save_json({"batch_id": batch_id, "count": len(qualified_records), "data": qualified_records},
                    os.path.join(QUALIFIED_DIR, f"qualified_{batch_id}.json"))
                save_json({"batch_id": batch_id, "count": len(unqualified_records), "data": unqualified_records},
                    os.path.join(UNQUALIFIED_DIR, f"unqualified_{batch_id}.json"))

                for b in metadata["batches"]:
                    if b["batch_id"] == batch_id:
                        b["status"] = "evaluated"
                        b["qualified_count"] = len(qualified_records)
                        b["unqualified_count"] = len(unqualified_records)
                        b["pass_rate"] = round(total_pass / len(records) * 100, 1) if records else 0
                save_metadata(metadata)

                total = len(records)
                rate = round(total_pass / total * 100, 1) if total else 0

                st.markdown('<div class="section-title">📊 评估结果概览</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📊 总数据", f"{total} 条")
                c2.metric("✅ 合格", f"{total_pass} 条", f"{rate}%")
                c3.metric("❌ 物理越限", f"{total_phy_fail} 条")
                c4.metric("⚠️ 统计异常", f"{total_stat_fail} 条")

                st.markdown('<div class="section-title">📋 分组评估详情</div>', unsafe_allow_html=True)
                for g in group_results:
                    with st.expander(f"📌 {g['period']} — {g['indicator']}（维度:{g['dimension']} | 总{g['total']}条 | 物理越限{g['phy_fail']}）"):
                        limit = g.get("limit", {})
                        st.write(f"**物理范围:** [{limit.get('min', 'N/A')}, {limit.get('max', 'N/A')}] {limit.get('unit', '')}")
                        st.write(f"**检测方法:** `{g['method']}`")
                        si = g["stat_info"]
                        if g["method"] == "z-score":
                            st.write(f"**统计量:** μ = {si.get('mu')}, σ = {si.get('sigma')}, 3σ范围 = [{si.get('threshold_range', ['N/A','N/A'])[0]}, {si.get('threshold_range', ['N/A','N/A'])[1]}]")
                        elif g["method"] == "IQR":
                            st.write(f"**统计量:** Q1 = {si.get('Q1')}, Q3 = {si.get('Q3')}, IQR = {si.get('IQR')}, 阈值 = [{si.get('lower_bound')}, {si.get('upper_bound')}]")
                        else:
                            st.write(f"**说明:** {si.get('note', 'N/A')}")

                st.success(f"✅ 评估完成！合格 {total_pass}/{total} 条 ({rate}%)。可前往「分析交互」查看图表。")

            except Exception as e:
                st.error(f"❌ 评估异常: {str(e)}")

# ============================================================
# 模块3：分析交互 - 上下布局
# ============================================================

@st.cache_data(show_spinner=False)
def cached_dim_rate(batch_id):
    q = load_json(os.path.join(QUALIFIED_DIR, f"qualified_{batch_id}.json"))
    u = load_json(os.path.join(UNQUALIFIED_DIR, f"unqualified_{batch_id}.json"))
    if not q and not u: return None
    qualified = q.get("data", []) if q else []
    unqualified = u.get("data", []) if u else []
    dim_total, dim_q = defaultdict(int), defaultdict(int)
    for r in qualified: dim_total[r.get("维度", "未知")] += 1; dim_q[r.get("维度", "未知")] += 1
    for r in unqualified: dim_total[r.get("维度", "未知")] += 1
    dims = sorted(dim_total.keys())
    rates = [round(dim_q.get(d, 0) / dim_total[d] * 100, 1) if dim_total[d] > 0 else 0 for d in dims]
    return {"dims": dims, "rates": rates, "totals": [dim_total[d] for d in dims]}

@st.cache_data(show_spinner=False)
def cached_period_rate(batch_id):
    q = load_json(os.path.join(QUALIFIED_DIR, f"qualified_{batch_id}.json"))
    u = load_json(os.path.join(UNQUALIFIED_DIR, f"unqualified_{batch_id}.json"))
    if not q and not u: return None
    qualified = q.get("data", []) if q else []
    unqualified = u.get("data", []) if u else []
    p_total, p_q = defaultdict(int), defaultdict(int)
    for r in qualified: p_total[r.get("时期", "未知")] += 1; p_q[r.get("时期", "未知")] += 1
    for r in unqualified: p_total[r.get("时期", "未知")] += 1
    order = ["萌芽期", "展叶期", "孕蕾期", "开花期"]
    periods = [p for p in order if p in p_total] + sorted([p for p in p_total if p not in order])
    rates = [round(p_q.get(p, 0) / p_total[p] * 100, 1) if p_total[p] > 0 else 0 for p in periods]
    return {"periods": periods, "rates": rates, "totals": [p_total[p] for p in periods]}

@st.cache_data(show_spinner=False)
def cached_wave_data(batch_id, period, indicator):
    q_file = os.path.join(QUALIFIED_DIR, f"qualified_{batch_id}.json")
    if not os.path.exists(q_file): return None
    with open(q_file, "r") as f: records = json.load(f).get("data", [])
    filtered = [r for r in records if r.get("时期") == period and r.get("指标") == indicator]
    if not filtered: return None
    filtered.sort(key=lambda x: x.get("时间", ""))
    period_data = PHYSICAL_LIMITS.get("limits", {}).get(period, {})
    limit_info = None
    for dn, items in period_data.items():
        if indicator in items: limit_info = items[indicator]; limit_info["dimension"] = dn; break
    values = [r["数值"] for r in filtered if isinstance(r.get("数值"), (int, float))]
    stats_d = {}
    if values: stats_d = {"count": len(values), "mean": round(np.mean(values), 2), "std": round(np.std(values), 2),
        "min": round(min(values), 2), "max": round(max(values), 2)}
    return {"records": filtered, "limit": limit_info, "stats": stats_d}

def page_analyze():
    st.markdown('<div class="main-title">📊 分析交互智能体</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint-box">合格率仪表 | 波动图表（按钮触发）| 异常数据集合（筛选 + 分页 + 导出）</div>', unsafe_allow_html=True)

    st.divider()

    metadata = load_metadata()
    eval_batches = [b for b in metadata.get("batches", []) if b.get("status") == "evaluated"]
    if not eval_batches: st.warning("⚠️ 暂无已评估的批次，请先到「阈值评估」模块进行评估"); return
    eval_batches.sort(key=lambda x: x.get("import_time", ""), reverse=True)

    batch_options = {f"[{b['batch_id']}] {b.get('batch_name', '')} (合格{b.get('qualified_count', 0)}条 / 总{b['total_rows']}条)": b["batch_id"] for b in eval_batches}
    selected_label = st.selectbox("📂 选择批次", list(batch_options.keys()), key="anal_batch")
    batch_id = batch_options[selected_label]

    # ========== 合格率仪表 ==========
    st.divider()
    st.markdown('<div class="section-title">📈 合格率仪表</div>', unsafe_allow_html=True)

    dim_data = cached_dim_rate(batch_id)
    if dim_data:
        st.markdown("##### 维度合格率")
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(dim_data["dims"]))
        ax.bar(x, dim_data["rates"], color="#4caf50", alpha=0.7, width=0.5, label="合格率(%)")
        ax.plot(x, dim_data["rates"], color="#ff5722", marker="o", linewidth=2.5, markersize=8, label="趋势")
        ax.set_xticks(x); ax.set_xticklabels(dim_data["dims"], fontsize=11)
        ax.set_ylabel("合格率 (%)", fontsize=12); ax.set_ylim(0, 105)
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
        for i, r in enumerate(dim_data["rates"]): ax.text(i, r + 2, f"{r}%", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    period_data = cached_period_rate(batch_id)
    if period_data:
        st.markdown("##### 时期合格率")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        x2 = np.arange(len(period_data["periods"]))
        ax2.bar(x2, period_data["rates"], color="#2196f3", alpha=0.7, width=0.5, label="合格率(%)")
        ax2.plot(x2, period_data["rates"], color="#ff9800", marker="s", linewidth=2.5, markersize=8, label="趋势")
        ax2.set_xticks(x2); ax2.set_xticklabels(period_data["periods"], fontsize=11)
        ax2.set_ylabel("合格率 (%)", fontsize=12); ax2.set_ylim(0, 105)
        ax2.legend(fontsize=10); ax2.grid(axis="y", alpha=0.3)
        for i, r in enumerate(period_data["rates"]): ax2.text(i, r + 2, f"{r}%", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig2, use_container_width=True)

    # ========== 波动图表（按钮触发） ==========
    st.divider()
    st.markdown('<div class="section-title">📉 波动图表（按钮触发，延迟加载）</div>', unsafe_allow_html=True)

    q_file = os.path.join(QUALIFIED_DIR, f"qualified_{batch_id}.json")
    if os.path.exists(q_file):
        with open(q_file, "r") as f:
            all_records = json.load(f).get("data", [])
        periods_avail = sorted(set(r.get("时期", "") for r in all_records if r.get("时期")))
        indicators_avail = sorted(set(r.get("指标", "") for r in all_records if r.get("指标")))

        sel_period = st.selectbox("🌱 选择时期", periods_avail, key="wave_period")
        sel_indicator = st.selectbox("📏 选择指标", indicators_avail, key="wave_index")
        gen_wave = st.button("📈 生成波动图", type="primary", use_container_width=True)

        if gen_wave:
            wave = cached_wave_data(batch_id, sel_period, sel_indicator)
            if wave and wave["records"]:
                fig3, ax3 = plt.subplots(figsize=(12, 5))
                records = wave["records"]
                times = [r.get("时间", "") for r in records]
                values = [r["数值"] for r in records if isinstance(r.get("数值"), (int, float))]
                limit_info = wave.get("limit")

                ax3.plot(range(len(values)), values, color="#4caf50", marker="o", markersize=4, linewidth=1.5, label=sel_indicator)
                if limit_info:
                    if limit_info.get("min") is not None:
                        ax3.axhline(y=limit_info["min"], color="#f44336", linestyle="--", linewidth=1.2, label=f'下限 {limit_info["min"]}{limit_info.get("unit", "")}')
                    if limit_info.get("max") is not None:
                        ax3.axhline(y=limit_info["max"], color="#f44336", linestyle="--", linewidth=1.2, label=f'上限 {limit_info["max"]}{limit_info.get("unit", "")}')

                step = max(1, len(times) // 12)
                display_times = [t if i % step == 0 else "" for i, t in enumerate(times)]
                ax3.set_xticks(range(len(display_times))); ax3.set_xticklabels(display_times, rotation=45, ha="right", fontsize=8)
                ax3.set_xlabel("时间", fontsize=11)
                unit = limit_info.get("unit", "") if limit_info else ""
                ax3.set_ylabel(f"{sel_indicator} ({unit})", fontsize=11)
                ax3.set_title(f"{sel_period} — {sel_indicator} 数据波动", fontsize=14, fontweight="bold")
                ax3.legend(fontsize=10); ax3.grid(alpha=0.3)
                plt.tight_layout(); st.pyplot(fig3, use_container_width=True)

                stats = wave.get("stats", {})
                if stats:
                    st.info(f"📊 样本数: **{stats.get('count', 0)}** | 均值: **{stats.get('mean')}** | 标准差: **{stats.get('std')}** | 范围: **[{stats.get('min')}, {stats.get('max')}]**")
            else:
                st.warning(f"{sel_period} — {sel_indicator} 无合格数据")

    # ========== 异常数据集合 ==========
    st.divider()
    st.markdown('<div class="section-title">⚠️ 异常数据集合</div>', unsafe_allow_html=True)

    uq_file = os.path.join(UNQUALIFIED_DIR, f"unqualified_{batch_id}.json")
    if os.path.exists(uq_file):
        with open(uq_file, "r") as f: uq_data = json.load(f).get("data", [])

        filter_reason = st.selectbox("🔍 按异常原因筛选", ["全部", "物理越限", "统计异常(z-score)", "统计异常(IQR)"], key="uq_filter")

        filtered = uq_data
        if filter_reason != "全部":
            if filter_reason == "物理越限": filtered = [r for r in uq_data if "物理越限" in str(r.get("_fail_type", ""))]
            elif filter_reason == "统计异常(z-score)": filtered = [r for r in uq_data if "z-score" in str(r.get("_fail_type", ""))]
            elif filter_reason == "统计异常(IQR)": filtered = [r for r in uq_data if "IQR" in str(r.get("_fail_type", ""))]

        st.write(f"📋 共 **{len(filtered)}** 条异常数据")

        page_size = 30
        total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        page_num = st.number_input("📄 页码", min_value=1, max_value=total_pages, value=1, step=1, key="uq_page")
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered))
        page_data = filtered[start_idx:end_idx]

        df_uq = pd.DataFrame([{
            "时期": r.get("时期", ""), "维度": r.get("维度", ""), "指标": r.get("指标", ""),
            "数值": r.get("数值", ""), "单位": r.get("单位", ""), "时间": r.get("时间", ""),
            "异常原因": r.get("_fail_reason", ""), "异常类型": r.get("_fail_type", ""),
        } for r in page_data])
        st.dataframe(df_uq, use_container_width=True)
        st.caption(f"第 {page_num}/{total_pages} 页 | 显示 {start_idx+1}-{end_idx} 条 / 共 {len(filtered)} 条")

        export_df = pd.DataFrame([{
            "时期": r.get("时期", ""), "维度": r.get("维度", ""), "指标": r.get("指标", ""),
            "数值": r.get("数值", ""), "单位": r.get("单位", ""), "时间": r.get("时间", ""),
            "异常原因": r.get("_fail_reason", ""), "异常类型": r.get("_fail_type", ""),
        } for r in filtered])
        csv = export_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("⬇️ 导出 CSV", csv, file_name=f"unqualified_{batch_id}_{filter_reason}.csv", mime="text/csv")
    else:
        st.info("该批次无异常数据")

# ============================================================
# 模块4：测试数据 - 上下布局 + 正式导入功能
# ============================================================

def generate_test_data(name, total, normal_ratio, phy_ratio, stat_ratio, period_cfg, start_date, fmt):
    try: period_dist = json.loads(period_cfg)
    except: period_dist = {"萌芽期": 0.25, "展叶期": 0.25, "孕蕾期": 0.25, "开花期": 0.25}

    configs = []
    for period, dim_data in PHYSICAL_LIMITS["limits"].items():
        for dim, indicators in dim_data.items():
            for ind, info in indicators.items():
                configs.append((period, dim, ind, info["unit"], info["min"], info["max"]))

    rows, base_date = [], datetime.strptime(start_date, "%Y-%m-%d")
    np.random.seed(42)

    for i in range(total):
        period = np.random.choice(list(period_dist.keys()), p=list(period_dist.values()))
        cfg_list = [c for c in configs if c[0] == period]
        if not cfg_list: continue
        period, dim, ind, unit, pmin, pmax = cfg_list[i % len(cfg_list)]

        r = np.random.random()
        if r < normal_ratio:
            val = np.random.uniform(pmin + (pmax - pmin) * 0.2, pmax - (pmax - pmin) * 0.2)
        elif r < normal_ratio + phy_ratio:
            val = np.random.choice([pmin - np.random.uniform(5, 20), pmax + np.random.uniform(5, 20)])
        else:
            bg = [np.random.uniform(pmin + (pmax - pmin) * 0.2, pmax - (pmax - pmin) * 0.2) for _ in range(10)]
            if len(bg) >= 5 and abs(stats.skew(bg)) < 1:
                mu_bg, s_bg = np.mean(bg), np.std(bg)
                val = mu_bg + np.random.choice([-1, 1]) * np.random.uniform(3.5, 5) * s_bg if s_bg > 0 else mu_bg
            else:
                q1_b, q3_b = np.percentile(bg, 25), np.percentile(bg, 75)
                iqr_b = q3_b - q1_b
                val = q3_b + np.random.uniform(1.5, 3) * iqr_b if np.random.random() > 0.5 else q1_b - np.random.uniform(1.5, 3) * iqr_b

        rows.append({"时期": period, "维度": dim, "指标": ind, "数值": round(float(val), 2),
            "单位": unit, "时间": (base_date + timedelta(days=i)).strftime("%Y-%m-%d")})

    df = pd.DataFrame(rows)
    return df

def page_test():
    st.markdown('<div class="main-title">🧪 测试数据智能体</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint-box">自定义生成模拟数据。可选择「正式导入系统」使其进入阈值评估和分析交互流程，或仅做临时评估。</div>', unsafe_allow_html=True)

    st.divider()

    # 参数设置区域
    st.markdown('<div class="section-title">⚙️ 参数设置</div>', unsafe_allow_html=True)

    if "test_cfg" not in st.session_state:
        st.session_state.test_cfg = {"name": "test_001", "total": 100, "normal": 0.7, "phy": 0.15, "stat": 0.15,
            "period": '{"萌芽期":0.25,"展叶期":0.25,"孕蕾期":0.25,"开花期":0.25}', "start": datetime.now().strftime("%Y-%m-%d")}

    cfg = st.session_state.test_cfg

    name = st.text_input("🏷️ 批次名称", value=cfg["name"], key="test_name")
    total = st.number_input("📊 总记录数", min_value=10, max_value=5000, value=cfg["total"], step=10, key="test_total")

    st.markdown("##### 数据类型比例（三者之和必须 = 1.0）")
    normal_r = st.slider("✅ 正常数据比例", 0.0, 1.0, cfg["normal"], 0.05, key="test_normal")
    phy_r = st.slider("❌ 物理超限比例", 0.0, 1.0 - normal_r, cfg["phy"], 0.05, key="test_phy")
    stat_r = st.slider("⚠️ 统计异常比例", 0.0, 1.0 - normal_r - phy_r, min(cfg["stat"], 1.0 - normal_r - phy_r), 0.05, key="test_stat")

    st.markdown("##### 其他参数")
    period_cfg = st.text_area("🌱 时期占比 (JSON格式)", value=cfg["period"], key="test_period")
    start_date = st.text_input("📅 起始日期 (YYYY-MM-DD)", value=cfg["start"], key="test_start")
    fmt = st.radio("📁 输出格式", ["xlsx", "csv"], horizontal=True, key="test_fmt")

    st.divider()

    # 生成按钮
    if st.button("⚡ 生成测试数据", type="primary", use_container_width=True):
        with st.spinner("⏳ 正在生成测试数据..."):
            try:
                if abs(normal_r + phy_r + stat_r - 1.0) > 0.01:
                    st.error("❌ 比例之和必须等于 1.0"); return

                df = generate_test_data(name, total, normal_r, phy_r, stat_r, period_cfg, start_date, fmt)
                st.session_state.test_cfg.update({"name": name, "total": total, "normal": normal_r, "phy": phy_r,
                    "stat": stat_r, "period": period_cfg, "start": start_date})
                st.session_state.last_test_df = df

                st.markdown('<div class="section-title">✅ 生成结果</div>', unsafe_allow_html=True)
                st.success(f"✅ 成功生成 **{len(df)}** 条测试数据！")

                # 统计概览
                st.markdown("##### 数据统计概览")
                c1, c2, c3, c4 = st.columns(4)
                for p in df["时期"].unique():
                    c1.metric("🌱 萌芽期", len(df[df["时期"].str.contains("萌芽", na=False)]))
                    c2.metric("🍃 展叶期", len(df[df["时期"].str.contains("展叶", na=False)]))
                    c3.metric("🌷 孕蕾期", len(df[df["时期"].str.contains("孕蕾", na=False)]))
                    c4.metric("🌸 开花期", len(df[df["时期"].str.contains("开花", na=False)]))
                    break

                st.markdown("##### 数据预览（前10行）")
                st.dataframe(df.head(10), use_container_width=True)

                # 保存到 TEST_DIR 供下载
                test_path = os.path.join(TEST_DIR, f"test_{name}.{fmt}")
                if fmt == "csv": df.to_csv(test_path, index=False, encoding="utf-8-sig")
                else: df.to_excel(test_path, index=False)

                with open(test_path, "rb") as f:
                    st.download_button("⬇️ 下载测试数据", f, file_name=f"test_{name}.{fmt}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if fmt == "xlsx" else "text/csv")

            except Exception as e:
                st.error(f"❌ 生成失败: {str(e)}")

    # ========== 操作区域 ==========
    if "last_test_df" in st.session_state:
        st.divider()
        st.markdown('<div class="section-title">🔧 后续操作</div>', unsafe_allow_html=True)

        df = st.session_state.last_test_df

        # 操作按钮（上下排列，不左右并列）
        st.markdown("##### 选项一：正式导入系统（推荐）")
        st.write("将测试数据作为正式批次导入，可进入「阈值评估」和「分析交互」流程。")
        if st.button("📥 正式导入系统", type="primary", use_container_width=True, key="btn_formal_import"):
            with st.spinner("⏳ 正在导入系统..."):
                try:
                    batch_id = get_next_batch_id()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    records = df.to_dict("records")

                    # 保存 raw_batch
                    raw_data = {"batch_id": batch_id, "import_time": timestamp,
                        "original_file": f"test_{name}.xlsx", "total_rows": len(records),
                        "valid_rows": len(records), "error_rows_count": 0,
                        "column_mapping": {k: k for k in ["时期", "维度", "指标", "数值", "单位", "时间"]},
                        "data": records}
                    save_json(raw_data, os.path.join(RAW_DIR, f"raw_batch_{batch_id}.json"))

                    # 更新 metadata
                    metadata = load_metadata()
                    metadata["batches"].append({"batch_id": batch_id, "batch_name": f"测试批次_{name}",
                        "import_time": timestamp, "original_file": f"test_{name}.xlsx",
                        "total_rows": len(records), "valid_rows": len(records),
                        "status": "imported", "is_test": False})
                    save_metadata(metadata)

                    st.success(f"✅ 测试数据已正式导入系统！批次号: **{batch_id}**")
                    st.info("🎯 现在可以前往「阈值评估」模块对该批次进行评估，然后在「分析交互」中查看图表。")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ 导入失败: {str(e)}")

        st.divider()

        st.markdown("##### 选项二：临时评估（不保存到系统）")
        st.write("仅在当前页面展示评估结果，不保存到数据总库。")
        if st.button("🔍 触发临时评估", type="secondary", use_container_width=True, key="btn_temp_eval"):
            with st.spinner("⏳ 正在进行临时评估..."):
                try:
                    records = df.to_dict("records")
                    groups = defaultdict(list)
                    for rec in records:
                        groups[(rec.get("时期", "未知"), rec.get("指标", "未知"))].append({"record": rec, "value": rec.get("数值", 0)})

                    t_qualified, t_unqualified, t_phy, t_stat = [], [], 0, 0
                    for (period, indicator), items in sorted(groups.items()):
                        phy_passed, phy_failed = [], []
                        for it in items:
                            rec, val = it["record"], it["value"]
                            if not isinstance(val, (int, float)):
                                try: val = float(val)
                                except: continue
                            ok, reason, _ = check_physical_limit(rec.get("时期", ""), rec.get("维度", ""), rec.get("指标", ""), val)
                            if ok: phy_passed.append({"record": rec, "value": val})
                            else: phy_failed.append({"record": rec}); t_phy += 1

                        if len(phy_passed) >= 5:
                            vals = [p["value"] for p in phy_passed]
                            anomalies, _, _ = detect_statistical_anomaly(vals)
                            for i, p in enumerate(phy_passed):
                                if anomalies[i]: t_unqualified.append({**p["record"], "_reason": "统计异常"}); t_stat += 1
                                else: t_qualified.append(p["record"])
                        else:
                            for p in phy_passed: t_qualified.append(p["record"])
                        for f in phy_failed: t_unqualified.append({**f["record"], "_reason": "物理越限"})

                    total_t = len(records)
                    rate = round(len(t_qualified)/total_t*100, 1) if total_t else 0

                    st.markdown("##### 📊 临时评估结果")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("📊 总数据", f"{total_t} 条")
                    c2.metric("✅ 合格", f"{len(t_qualified)} 条", f"{rate}%")
                    c3.metric("❌ 物理越限", f"{t_phy} 条")
                    c4.metric("⚠️ 统计异常", f"{t_stat} 条")

                    st.info("💡 此为临时结果，未保存。如需图表分析，请使用「选项一：正式导入系统」。")
                except Exception as e:
                    st.error(f"❌ 临时评估失败: {str(e)}")

# ============================================================
# 模块5：数据总库 - 上下布局
# ============================================================

def page_database():
    st.markdown('<div class="main-title">🗄️ 数据总库</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint-box">管理所有正式批次，支持汇总统计图表和批量 ZIP 导出。</div>', unsafe_allow_html=True)

    st.divider()

    tab_list, tab_charts, tab_export = st.tabs(["📋 批次列表", "📈 汇总图表", "📦 数据导出"])

    # ========== 批次列表 ==========
    with tab_list:
        metadata = load_metadata()
        formal_batches = [b for b in metadata.get("batches", []) if not b.get("is_test", False)]
        if not formal_batches: st.info("暂无正式批次"); return

        st.write(f"共 **{len(formal_batches)}** 个正式批次")
        st.divider()

        for b in formal_batches:
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([1.5, 2.5, 1, 1, 1])
                with c1: st.code(b["batch_id"])
                with c2: st.write(b.get("original_file", "-"))
                with c3: st.write(f"{b.get('total_rows', 0)} 条")
                rate = b.get('pass_rate', '-')
                with c4: st.write(f"{rate}%" if isinstance(rate, (int, float)) else "-")
                with c5:
                    if st.button("🗑️ 删除", key=f"del_{b['batch_id']}"):
                        for d, prefix in [(RAW_DIR, "raw_batch_"), (QUALIFIED_DIR, "qualified_"), (UNQUALIFIED_DIR, "unqualified_")]:
                            fpath = os.path.join(d, f"{prefix}{b['batch_id']}.json")
                            if os.path.exists(fpath): os.remove(fpath)
                        metadata["batches"] = [x for x in metadata["batches"] if x["batch_id"] != b["batch_id"]]
                        save_metadata(metadata)
                        st.success(f"批次 {b['batch_id']} 已删除"); st.rerun()
            st.divider()

    # ========== 汇总图表 ==========
    with tab_charts:
        all_qualified, all_unqualified = [], []
        for b in formal_batches:
            q = load_json(os.path.join(QUALIFIED_DIR, f"qualified_{b['batch_id']}.json"))
            u = load_json(os.path.join(UNQUALIFIED_DIR, f"unqualified_{b['batch_id']}.json"))
            if q: all_qualified.extend(q.get("data", []))
            if u: all_unqualified.extend(u.get("data", []))

        if not all_qualified and not all_unqualified: st.info("无汇总数据"); return

        st.markdown('<div class="section-title">时期 + 维度 合格率柱状图</div>', unsafe_allow_html=True)
        pd_combo_total, pd_combo_q = defaultdict(int), defaultdict(int)
        for r in all_qualified: key = f"{r.get('时期', '未知')}-{r.get('维度', '未知')}"; pd_combo_total[key] += 1; pd_combo_q[key] += 1
        for r in all_unqualified: key = f"{r.get('时期', '未知')}-{r.get('维度', '未知')}"; pd_combo_total[key] += 1
        combos = sorted(pd_combo_total.keys())
        combo_rates = [round(pd_combo_q.get(c, 0) / pd_combo_total[c] * 100, 1) if pd_combo_total[c] > 0 else 0 for c in combos]

        fig1, ax1 = plt.subplots(figsize=(12, 5))
        colors = {"环境": "#4caf50", "土壤": "#2196f3", "生长状况": "#ff9800"}
        bar_colors = [colors.get(c.split("-")[1], "#999") for c in combos]
        ax1.bar(range(len(combos)), combo_rates, color=bar_colors, alpha=0.75, width=0.6)
        ax1.set_xticks(range(len(combos))); ax1.set_xticklabels(combos, rotation=45, ha="right", fontsize=9)
        ax1.set_ylabel("合格率 (%)", fontsize=11); ax1.set_title("时期 + 维度 合格率柱状图", fontsize=14, fontweight="bold")
        ax1.set_ylim(0, 105); ax1.grid(axis="y", alpha=0.3)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[k], label=k) for k in colors]
        ax1.legend(handles=legend_elements, fontsize=10)
        plt.tight_layout(); st.pyplot(fig1, use_container_width=True)

        st.divider()
        st.markdown('<div class="section-title">时期 + 维度 数据量堆积柱状图</div>', unsafe_allow_html=True)
        pd_vol = defaultdict(lambda: defaultdict(int))
        for r in all_qualified + all_unqualified:
            pd_vol[r.get("时期", "未知")][r.get("维度", "未知")] += 1
        periods_v = sorted(set(r.get("时期", "") for r in all_qualified + all_unqualified if r.get("时期")))
        dims_v = ["环境", "土壤", "生长状况"]
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bottom = [0] * len(periods_v)
        for dim in dims_v:
            vals = [pd_vol[p].get(dim, 0) for p in periods_v]
            ax2.bar(periods_v, vals, bottom=bottom, label=dim, color=colors.get(dim, "#999"), alpha=0.75, width=0.6)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax2.set_ylabel("数据条数", fontsize=11); ax2.set_title("时期 + 维度 数据量堆积柱状图", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10); ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig2, use_container_width=True)

        st.divider()
        st.markdown('<div class="section-title">总体合格率趋势（按批次导入时间）</div>', unsafe_allow_html=True)
        formal_batches_sorted = sorted(formal_batches, key=lambda x: x.get("import_time", ""))
        batch_labels = [f"{b['batch_id']}" for b in formal_batches_sorted]
        batch_rates = [b.get("pass_rate", 0) for b in formal_batches_sorted]
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(range(len(batch_labels)), batch_rates, color="#4caf50", marker="o", linewidth=2.5, markersize=8)
        ax3.set_xticks(range(len(batch_labels))); ax3.set_xticklabels(batch_labels, fontsize=10)
        ax3.set_ylabel("合格率 (%)", fontsize=11); ax3.set_title("总体合格率趋势（按批次导入时间）", fontsize=14, fontweight="bold")
        ax3.set_ylim(0, 105); ax3.grid(alpha=0.3)
        for i, r in enumerate(batch_rates): ax3.text(i, r + 2, f"{r}%", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig3, use_container_width=True)

    # ========== 数据导出 ==========
    with tab_export:
        if st.button("📦 导出全部批次数据（ZIP）", type="primary", use_container_width=True):
            with st.spinner("⏳ 正在打包数据..."):
                zip_path = os.path.join(DATA_DIR, "all_batches_export.zip")
                with zipfile.ZipFile(zip_path, "w") as zf:
                    for b in formal_batches:
                        bid = b["batch_id"]
                        for d, prefix in [(RAW_DIR, "raw_batch_"), (QUALIFIED_DIR, "qualified_"), (UNQUALIFIED_DIR, "unqualified_")]:
                            fpath = os.path.join(d, f"{prefix}{bid}.json")
                            if os.path.exists(fpath):
                                zf.write(fpath, f"{bid}/{prefix}{bid}.json")
                with open(zip_path, "rb") as f:
                    st.download_button("⬇️ 下载 ZIP", f, file_name="all_batches_export.zip", mime="application/zip")

# ============================================================
# 主入口
# ============================================================

def main():
    st.markdown('<div class="main-title" style="text-align:center; padding:12px 0;">🌷 百合生长模型数据监测智能体平台</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#689f38; margin:0 0 16px 0;">低代码 · 多智能体协同 · 自适应统计异常检测</p>', unsafe_allow_html=True)
    st.divider()

    page = sidebar_nav()

    if page == "import": page_import()
    elif page == "evaluate": page_evaluate()
    elif page == "analyze": page_analyze()
    elif page == "test": page_test()
    elif page == "database": page_database()

main()
