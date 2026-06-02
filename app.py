"""
百合生长模型数据监测智能体平台 V3.0
重构：时期仅作标签 / 全指标强制物理极限 / 统一IQR / 组合模式孤立森林告警
"""

import streamlit as st
st.set_page_config(page_title="百合生长模型数据监测智能体平台", page_icon="🌷", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import zipfile
from scipy import stats
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# 全局路径
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

with open(os.path.join(CONFIG_DIR, "physical_limits.json"), "r", encoding="utf-8") as f:
    PHYSICAL_LIMITS = json.load(f)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .main-title { font-size: 24px; font-weight: 700; color: #1b5e20; text-align: center; padding: 8px 0; }
    .subtitle { text-align: center; color: #689f38; font-size: 13px; margin-bottom: 16px; }
    .section-title { font-size: 17px; font-weight: 600; color: #2e7d32; padding: 12px 0 8px 0; border-bottom: 2px solid #c8e6c9; margin: 8px 0 12px 0; }
    .card { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin: 8px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
    .card-title { font-size: 15px; font-weight: 600; color: #333; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    .tag { background: #e8eaf6; color: #3f51b5; font-size: 11px; padding: 2px 8px; border-radius: 12px; margin-left: 6px; }
    .tag-ok { background: #e8f5e9; color: #2e7d32; }
    .tag-warn { background: #fff3e0; color: #e65100; }
    .tag-danger { background: #ffebee; color: #c62828; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 工具函数
# ============================================================
def get_next_batch_id():
    meta_path = os.path.join(METADATA_DIR, "batch_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if m["batches"]: return f"{max([int(b['batch_id']) for b in m['batches']]) + 1:06d}"
    return "000001"

def load_metadata():
    p = os.path.join(METADATA_DIR, "batch_metadata.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    return {"batches": [], "version": "1.0"}

def save_metadata(m):
    with open(os.path.join(METADATA_DIR, "batch_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

def save_json(d, fp):
    with open(fp, "w", encoding="utf-8") as f: json.dump(d, f, ensure_ascii=False, indent=2)

def load_json(fp):
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f: return json.load(f)
    return None

# ============================================================
# 列名匹配
# ============================================================
COLUMN_MAPPING = {
    "时期": {"required": False, "keywords": ["时期", "生长时期", "阶段", "生长期", "period", "stage", "phase"]},
    "维度": {"required": False, "keywords": ["维度", "类型", "category", "dimension", "type", "类别"]},
    "指标": {"required": True, "keywords": ["指标", "参数", "indicator", "index", "parameter", "metric", "item", "变量"]},
    "数值": {"required": True, "keywords": ["数值", "值", "value", "data", "测量值", "读数"]},
    "单位": {"required": False, "keywords": ["单位", "unit", "度量单位"]},
    "时间": {"required": False, "keywords": ["时间", "日期", "time", "date", "timestamp"]},
}
PERIOD_KEYWORDS = {"萌芽期": ["萌芽", "发芽", "萌发"], "展叶期": ["展叶", "叶片展开"], "孕蕾期": ["孕蕾", "蕾期", "花蕾期", "现蕾"], "开花期": ["开花", "花期", "盛花"]}
DIMENSION_KEYWORDS = {"环境": ["环境", "气象", "天气"], "土壤": ["土壤", "土质", "泥土"], "生长状况": ["生长", "植株", "植物", "发育"]}

def smart_column_match(df_columns):
    mapping, matched = {}, set()
    for sn, cfg in COLUMN_MAPPING.items():
        bm, bs = None, 0
        for col in df_columns:
            if col in matched: continue
            cl = col.lower().strip()
            for kw in cfg["keywords"]:
                kl = kw.lower().strip()
                if cl == kl: bm, bs = col, 100; break
                elif kl in cl or cl in kl:
                    sc = 80 if len(kl) >= 2 else 50
                    if sc > bs: bm, bs = col, sc
                elif cl.startswith(kl[:2]) and len(cl) <= len(kl) + 2:
                    if 60 > bs: bm, bs = col, 60
            if bs == 100: break
        mapping[sn] = bm
        if bm: matched.add(bm)
    return mapping

def validate_period_value(v):
    if pd.isna(v): return False, "空值"
    s = str(v).strip()
    if s in PERIOD_KEYWORDS: return True, s
    for sp, kws in PERIOD_KEYWORDS.items():
        for kw in kws:
            if kw in s or s in kw: return True, sp
    return False, f"无法识别: {s}"

def validate_dimension_value(v):
    if pd.isna(v): return False, "空值"
    s = str(v).strip()
    if s in DIMENSION_KEYWORDS: return True, s
    for sd, kws in DIMENSION_KEYWORDS.items():
        for kw in kws:
            if kw in s or s in kw: return True, sd
    return False, f"无法识别: {s}"

# ============================================================
# 物理极值检查（严格模式：无配置则报错）
# ============================================================
def check_physical_limit(period, dimension, indicator, value):
    pd_ = PHYSICAL_LIMITS.get("limits", {}).get(period, {})
    dd = pd_.get(dimension, {})
    ii = dd.get(indicator)
    if ii is None:
        for dn, items in pd_.items():
            if indicator in items: ii = items[indicator]; break
    # 严格模式：未配置则抛出异常
    if ii is None:
        raise ValueError(f"指标 [{indicator}] 在时期 [{period}] 下未配置物理极限，请在 physical_limits.json 中配置")
    mn, mx, un = ii.get("min"), ii.get("max"), ii.get("unit", "")
    li = {"min": mn, "max": mx, "unit": un}
    if mn is not None and value < mn: return False, f"物理越限: {value} < {mn}", li
    if mx is not None and value > mx: return False, f"物理越限: {value} > {mx}", li
    return True, "合格", li

# ============================================================
# IQR 统计异常检测（统一方法）
# ============================================================
def detect_iqr(values, k=1.5):
    arr = np.array(values, dtype=float)
    arr_sorted = np.sort(arr)
    q1, q3 = np.percentile(arr_sorted, [25, 75])
    iqr = q3 - q1
    lb = q1 - k * iqr
    ub = q3 + k * iqr
    outlier_mask = (arr < lb) | (arr > ub)
    return outlier_mask.tolist(), {
        "method": "IQR", "k": k,
        "Q1": round(float(q1), 4), "Q3": round(float(q3), 4),
        "IQR": round(float(iqr), 4),
        "lb": round(float(lb), 4), "ub": round(float(ub), 4),
        "n": len(values), "anomaly": int(outlier_mask.sum())
    }

# ============================================================
# 组合模式检测（孤立森林）：灰霉病 + 徒长风险
# 仅告警，不参与分流
# ============================================================
def detect_combination_patterns(records):
    """
    组合模式检测：
    - 灰霉病风险：昼温 + 湿度 + 光照强度（3维）
    - 徒长风险：株高 + 茎粗（2维）
    返回告警列表，不参与合格/不合格分流
    """
    # 按时间分组，将长格式转为宽格式
    time_groups = defaultdict(dict)
    for rec in records:
        t = rec.get("时间", rec.get("时期", "unknown"))
        indicator = rec.get("指标", "")
        val = rec.get("数值")
        try:
            val = float(val)
        except:
            continue
        time_groups[t][indicator] = val
    
    alerts = []
    
    # 1. 灰霉病风险检测
    gray_mold_data = []
    gray_mold_times = []
    for t, indicators in time_groups.items():
        if all(k in indicators for k in ["昼温", "湿度", "光照强度"]):
            gray_mold_data.append([indicators["昼温"], indicators["湿度"], indicators["光照强度"]])
            gray_mold_times.append(t)
    
    if len(gray_mold_data) >= 5:
        clf = IsolationForest(contamination=0.05, random_state=42)
        preds = clf.fit_predict(np.array(gray_mold_data))
        for i, pred in enumerate(preds):
            if pred == -1:
                alerts.append({
                    "时间": gray_mold_times[i],
                    "模式": "灰霉病风险",
                    "昼温": gray_mold_data[i][0],
                    "湿度": gray_mold_data[i][1],
                    "光照强度": gray_mold_data[i][2]
                })
    
    # 2. 徒长风险检测
    etiolation_data = []
    etiolation_times = []
    for t, indicators in time_groups.items():
        if all(k in indicators for k in ["株高", "茎粗"]):
            etiolation_data.append([indicators["株高"], indicators["茎粗"]])
            etiolation_times.append(t)
    
    if len(etiolation_data) >= 5:
        clf = IsolationForest(contamination=0.05, random_state=42)
        preds = clf.fit_predict(np.array(etiolation_data))
        for i, pred in enumerate(preds):
            if pred == -1:
                alerts.append({
                    "时间": etiolation_times[i],
                    "模式": "徒长风险",
                    "株高": etiolation_data[i][0],
                    "茎粗": etiolation_data[i][1]
                })
    
    return alerts

# ============================================================
# 核心评估逻辑（V3.0 重构）
# ============================================================
def run_evaluation(records, batch_id="temp"):
    """
    评估流程：
    1. 所有数据强制物理极值检查（无配置则报错）
    2. 物理合格数据按【指标】分组（时期仅作标签，不参与分组）
    3. 组内排序后 IQR(k=1.5) 检测
    4. 组合模式孤立森林检测（仅告警，不参与分流）
    """
    if not records:
        return {"total": 0, "pass": 0, "rate": 0, "phy_fail": 0, "stat_fail": 0,
                "qualified": [], "unqualified": [], "groups": [], "combination_alerts": []}
    
    # 第一轮：物理极值检查（全量）
    idx_phy_ok = {}
    idx_phy_reason = {}
    idx_limit = {}
    total_phy_fail = 0
    
    # 第一轮：物理极值检查（全量）
    for idx, rec in enumerate(records):
        val = rec.get("数值")
        if not isinstance(val, (int, float)):
            try: val = float(val)
            except: continue
        
        # 修改：捕获未配置阈值，标记为异常而不中断
        try:
            ok, reason, limit_info = check_physical_limit(
                rec.get("时期", ""), rec.get("维度", ""), rec.get("指标", ""), val
            )
            idx_limit[idx] = limit_info
            if ok:
                idx_phy_ok[idx] = True
            else:
                idx_phy_ok[idx] = False
                idx_phy_reason[idx] = reason
                total_phy_fail += 1
        except ValueError as e:
            # 未配置物理极限：直接标记为异常，不抛错终止
            idx_phy_ok[idx] = False
            idx_phy_reason[idx] = f"未配置阈值: {e}"
            idx_limit[idx] = {"min": None, "max": None, "unit": rec.get("单位", "")}
            total_phy_fail += 1
    
    # 第二轮：按【指标】分组 IQR 检测（时期仅作标签）
    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        if idx_phy_ok.get(idx, False):
            groups[rec.get("指标", "未知")].append({"idx": idx, "record": rec, "value": float(rec.get("数值", 0))})
    
    idx_stat_anomaly = {}
    group_results = []
    
    for indicator, items in sorted(groups.items()):
        values = [it["value"] for it in items]
        n = len(values)
        
        if n < 5:
            # 样本不足，跳过统计检测
            for it in items:
                idx_stat_anomaly[it["idx"]] = False
            stat_info = {"method": "样本不足", "note": f"n={n}<5，跳过IQR检测"}
            anomalies = [False] * n
        else:
            anomalies, stat_info = detect_iqr(values, k=1.5)
            for i, it in enumerate(items):
                idx_stat_anomaly[it["idx"]] = anomalies[i]
        
        # 组装分组记录
        record_status = []
        for i, it in enumerate(items):
            status = "IQR异常" if anomalies[i] else "合格"
            record_status.append({**it["record"], "_status": status})
        
        phy_fail_count = sum(1 for idx, rec in enumerate(records) 
                            if rec.get("指标") == indicator and not idx_phy_ok.get(idx, False))
        stat_fail_count = sum(1 for r in record_status if r.get("_status") == "IQR异常")
        
        # 获取该指标任一记录的极限信息用于展示
        sample_limit = {}
        for it in items:
            if it["idx"] in idx_limit:
                sample_limit = idx_limit[it["idx"]]
                break
        
        group_results.append({
            "indicator": indicator,
            "total": n + phy_fail_count,
            "phy_fail": phy_fail_count,
            "stat_fail": stat_fail_count,
            "method": stat_info.get("method", "IQR"),
            "stat_info": stat_info,
            "limit": sample_limit,
            "records": record_status[:50]
        })
    
    # 第三轮：组合模式检测（孤立森林，仅告警，不参与分流）
    combination_alerts = detect_combination_patterns(records)
    
    # 最终分流
    qualified_records, unqualified_records = [], []
    total_stat_fail = 0
    total_pass = 0
    
    for idx, rec in enumerate(records):
        if not idx_phy_ok.get(idx, False):
            unqualified_records.append({
                **rec, "_fail_reason": idx_phy_reason.get(idx, "物理越限"),
                "_fail_type": "物理越限", "_batch_id": batch_id
            })
            continue
        
        is_stat = idx_stat_anomaly.get(idx, False)
        if is_stat:
            unqualified_records.append({
                **rec, "_fail_reason": "IQR统计异常",
                "_fail_type": "IQR异常", "_batch_id": batch_id
            })
            total_stat_fail += 1
        else:
            qualified_records.append(rec)
            total_pass += 1
    
    return {
        "qualified": qualified_records,
        "unqualified": unqualified_records,
        "groups": group_results,
        "total": len(records),
        "pass": total_pass,
        "phy_fail": total_phy_fail,
        "stat_fail": total_stat_fail,
        "rate": round(total_pass / len(records) * 100, 1) if records else 0,
        "combination_alerts": combination_alerts
    }

# ============================================================
# Plotly 图表函数
# ============================================================
def plot_bar_line(x_labels, values, title, cbar="#4caf50", cline="#ff5722", ylabel="合格率 (%)"):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_labels, y=values, name="合格率",
        marker_color=cbar, opacity=0.7, text=[f"{v}%" for v in values], textposition="outside"))
    fig.add_trace(go.Scatter(x=x_labels, y=values, name="趋势",
        mode="lines+markers", line=dict(color=cline, width=2.5), marker=dict(size=10)))
    fig.update_layout(title=title, yaxis=dict(title=ylabel, range=[0, 105], gridcolor="rgba(0,0,0,0.1)"),
        xaxis=dict(title=""), bargap=0.4, height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40))
    return fig

def plot_time_series(records, indicator, limit_info):
    """时间序列折线图：以服务生长模型为目标"""
    df = pd.DataFrame([r for r in records if r.get("指标") == indicator and isinstance(r.get("数值"), (int, float))])
    if df.empty: return None
    df = df.sort_values(by="时间" if "时间" in df.columns else "时期")
    
    x_vals = df["时间"].tolist() if "时间" in df.columns else df["时期"].tolist()
    y_vals = df["数值"].tolist()
    unit = limit_info.get("单位", "") if limit_info else ""
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_vals))), y=y_vals, mode="lines+markers",
        name=indicator, line=dict(color="#4caf50", width=2), marker=dict(size=6)))
    
    if limit_info:
        if limit_info.get("min") is not None:
            fig.add_hline(y=limit_info["min"], line_dash="dash", line_color="#f44336",
                annotation_text=f"下限 {limit_info['min']}{unit}", annotation_position="right")
        if limit_info.get("max") is not None:
            fig.add_hline(y=limit_info["max"], line_dash="dash", line_color="#f44336",
                annotation_text=f"上限 {limit_info['max']}{unit}", annotation_position="right")
    
    n = len(x_vals)
    tick_vals = list(range(0, n, max(1, n // 10)))
    tick_text = [str(x_vals[i])[:10] for i in tick_vals]
    
    fig.update_layout(title=f"{indicator} 时间序列（服务生长模型）",
        xaxis=dict(title="时间序列", tickmode="array", tickvals=tick_vals, ticktext=tick_text, tickangle=30),
        yaxis=dict(title=f"{indicator} ({unit})", gridcolor="rgba(0,0,0,0.1)"),
        height=450, margin=dict(t=80, b=60))
    return fig

def plot_box_by_period(records, indicator):
    """时期箱线图：展示同一指标在不同时期的分布"""
    df = pd.DataFrame([r for r in records if r.get("指标") == indicator and isinstance(r.get("数值"), (int, float))])
    if df.empty or "时期" not in df.columns: return None
    
    periods = sorted(df["时期"].unique())
    fig = go.Figure()
    for p in periods:
        vals = df[df["时期"] == p]["数值"].tolist()
        fig.add_trace(go.Box(y=vals, name=p, boxmean=True))
    
    fig.update_layout(title=f"{indicator} 时期分布箱线图",
        yaxis=dict(title=indicator), height=400,
        xaxis=dict(title="时期"), margin=dict(t=60, b=40))
    return fig

def plot_combination_alerts(alerts):
    """组合模式告警图"""
    if not alerts: return None
    df = pd.DataFrame(alerts)
    fig = go.Figure()
    
    for mode in df["模式"].unique():
        sub = df[df["模式"] == mode]
        fig.add_trace(go.Scatter(
            x=list(range(len(sub))), y=[1]*len(sub),
            mode="markers+text", name=mode,
            marker=dict(size=20, symbol="x", color="#c62828" if "灰霉" in mode else "#f57c00"),
            text=sub["时间"].astype(str).str[:10], textposition="top center"
        ))
    
    fig.update_layout(title="组合模式异常告警分布（孤立森林）",
        yaxis=dict(showticklabels=False, title=""),
        xaxis=dict(title="告警序号"), height=300,
        margin=dict(t=60, b=40))
    return fig

# ============================================================
# 侧边栏导航
# ============================================================
def sidebar_nav():
    st.sidebar.markdown("<div style='text-align:center; padding:6px 0;'><p style='color:#e8f5e9; font-size:22px; font-weight:700; margin:0;'>🌷 百合监测</p><p style='color:rgba(232,245,233,0.5); font-size:11px; margin:2px 0 0 0;'>低代码智能体平台 V3.0</p></div>", unsafe_allow_html=True)
    st.sidebar.divider()
    pages = {"import": "📥 数据导入", "evaluate": "🔍 阈值评估", "analyze": "📊 分析交互", "test": "🧪 测试数据", "database": "🗄️ 数据总库"}
    sel = st.sidebar.radio("导航", list(pages.keys()), format_func=lambda x: pages[x], label_visibility="collapsed")
    st.sidebar.divider()
    return sel

# ============================================================
# 模块1：数据导入
# ============================================================
# ============================================================
# 模块1：数据导入（增强版：导入时物理极值预检）
# ============================================================
def page_import():
    st.markdown('<div class="main-title">📥 数据导入智能体</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">支持 .xlsx / .csv | 智能列名匹配 | 空值校验 | 导入时物理极值预检</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("📎 选择数据文件", type=["xlsx", "csv"])
    if uploaded: st.info(f"📁 {uploaded.name} | {uploaded.size/1024:.1f} KB")

    if uploaded and st.button("📤 开始导入", type="primary"):
        with st.spinner("⏳ 导入中..."):
            try:
                df = pd.read_csv(uploaded, encoding='utf-8') if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
                total = len(df)
                if total == 0: st.error("文件为空"); return
                if total > 5000: st.error(f"超过5000行({total})"); return

                col_map = smart_column_match(list(df.columns))
                missing = [k for k, v in col_map.items() if COLUMN_MAPPING[k]["required"] and v is None]
                if missing: st.error(f"缺少必填列: {', '.join(missing)}"); return

                rev_map = {v: k for k, v in col_map.items() if v}
                df_renamed = df.rename(columns=rev_map)
                
                # ==================== 新增：导入时物理极值预检 ====================
                valid_rows, error_details, phy_warn_rows = [], [], []
                
                for idx, row in df_renamed.iterrows():
                    errs = []
                    # 基础校验
                    for f in ["指标", "数值"]:
                        if f in row.index and pd.isna(row[f]): errs.append(f"{f}空")
                    if "时期" in row.index and not pd.isna(row["时期"]):
                        ok, msg = validate_period_value(row["时期"])
                        if not ok: errs.append(msg)
                        else: row = row.copy(); row["时期"] = msg
                    if "维度" in row.index and not pd.isna(row["维度"]):
                        ok, msg = validate_dimension_value(row["维度"])
                        if not ok: errs.append(msg)
                        else: row = row.copy(); row["维度"] = msg
                    if "数值" in row.index and not pd.isna(row["数值"]):
                        try: float(row["数值"])
                        except: errs.append(f"数值无效")
                    
                    # 物理极值预检（新增核心逻辑）
                    if not errs and "数值" in row.index and not pd.isna(row["数值"]):
                        try:
                            val = float(row["数值"])
                            ok, reason, limit_info = check_physical_limit(
                                row.get("时期", ""), row.get("维度", ""), row.get("指标", ""), val
                            )
                            if not ok:
                                # 物理越限：标记为异常，但记录保留
                                phy_warn_rows.append({
                                    "行": int(idx)+2,
                                    "时期": str(row.get("时期","")),
                                    "维度": str(row.get("维度","")),
                                    "指标": str(row.get("指标","")),
                                    "数值": str(row.get("数值","")),
                                    "异常类型": "物理越限",
                                    "异常原因": reason,
                                    "物理下限": limit_info.get("min"),
                                    "物理上限": limit_info.get("max"),
                                    "单位": limit_info.get("unit","")
                                })
                                # 仍然导入，但标记状态
                                row = row.copy()
                                row["_import_status"] = "物理越限"
                                row["_import_reason"] = reason
                        except ValueError as e:
                            # 物理极限未配置：标记为异常
                            phy_warn_rows.append({
                                "行": int(idx)+2,
                                "时期": str(row.get("时期","")),
                                "维度": str(row.get("维度","")),
                                "指标": str(row.get("指标","")),
                                "数值": str(row.get("数值","")),
                                "异常类型": "未配置阈值",
                                "异常原因": str(e),
                                "物理下限": "N/A",
                                "物理上限": "N/A",
                                "单位": ""
                            })
                            row = row.copy()
                            row["_import_status"] = "未配置阈值"
                            row["_import_reason"] = str(e)
                    
                    if errs:
                        error_details.append({
                            "行": int(idx)+2,
                            "时期": str(row.get("时期","")),
                            "维度": str(row.get("维度","")),
                            "指标": str(row.get("指标","")),
                            "数值": str(row.get("数值","")),
                            "错误": ";".join(errs)
                        })
                    else:
                        valid_rows.append(row.to_dict())
                
                # 统计
                phy_fail_count = len([r for r in phy_warn_rows if r["异常类型"] == "物理越限"])
                no_limit_count = len([r for r in phy_warn_rows if r["异常类型"] == "未配置阈值"])
                # ============================================================

                if not valid_rows and not phy_warn_rows: 
                    st.error(f"全部无效"); 
                    return

                batch_id = get_next_batch_id()
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 保存所有数据（包括标记为异常的）
                save_json({
                    "batch_id": batch_id, 
                    "import_time": ts, 
                    "original_file": uploaded.name,
                    "total_rows": total, 
                    "valid_rows": len(valid_rows), 
                    "error_rows_count": len(error_details),
                    "phy_warn_rows_count": len(phy_warn_rows),
                    "phy_fail_count": phy_fail_count,
                    "no_limit_count": no_limit_count,
                    "column_mapping": col_map, 
                    "data": valid_rows
                }, os.path.join(RAW_DIR, f"raw_batch_{batch_id}.json"))
                
                # 单独保存异常数据供查看
                if phy_warn_rows:
                    save_json({
                        "batch_id": batch_id,
                        "import_time": ts,
                        "phy_warn_rows": phy_warn_rows
                    }, os.path.join(RAW_DIR, f"raw_batch_{batch_id}_warnings.json"))
                
                m = load_metadata()
                m["batches"].append({
                    "batch_id": batch_id, 
                    "batch_name": f"批次_{batch_id}", 
                    "import_time": ts,
                    "original_file": uploaded.name, 
                    "total_rows": total, 
                    "valid_rows": len(valid_rows),
                    "phy_warn_count": len(phy_warn_rows),
                    "status": "imported", 
                    "is_test": False
                })
                save_metadata(m)

                # 展示结果
                pr = len(valid_rows)/total*100 if total > 0 else 0
                st.success(f"✅ 导入完成！批次 {batch_id} | 总{total} | 有效{len(valid_rows)} | 合格率{pr:.1f}%")
                
                # 异常数据看板（新增）
                if phy_warn_rows:
                    st.markdown("<div class='section-title'>⚠️ 导入时物理极值预检异常</div>", unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("物理越限", phy_fail_count, delta_color="inverse")
                    c2.metric("未配置阈值", no_limit_count, delta_color="inverse")
                    c3.metric("异常占比", f"{len(phy_warn_rows)/total*100:.1f}%")
                    
                    with st.expander(f"📋 异常明细（{len(phy_warn_rows)} 条）"):
                        st.dataframe(pd.DataFrame(phy_warn_rows), hide_index=True)
                        csv_warn = pd.DataFrame(phy_warn_rows).to_csv(index=False, encoding="utf-8-sig")
                        st.download_button("⬇️ 导出异常数据CSV", csv_warn, file_name=f"import_warnings_{batch_id}.csv")
                    
                    if no_limit_count > 0:
                        st.error(f"❌ 有 {no_limit_count} 条数据因指标未配置物理极限被标记为异常，请检查 config/physical_limits.json")
                
                # 基础错误行
                if error_details:
                    with st.expander(f"⚠️ 格式错误行 ({len(error_details)} 条)"): 
                        st.dataframe(pd.DataFrame(error_details), hide_index=True)
                
                # 预览
                st.markdown("<div class='section-title'>预览（前10行）</div>", unsafe_allow_html=True)
                preview_df = pd.DataFrame(valid_rows[:10])
                if "_import_status" in preview_df.columns:
                    st.dataframe(preview_df[["时期","维度","指标","数值","单位","_import_status"]], hide_index=True)
                else:
                    st.dataframe(preview_df, hide_index=True)
                
                # 导出原始数据
                ep = os.path.join(RAW_DIR, f"raw_batch_{batch_id}_export.xlsx")
                pd.DataFrame(valid_rows).to_excel(ep, index=False)
                with open(ep, "rb") as f: 
                    st.download_button("⬇️ 下载原始数据", f, file_name=f"raw_{batch_id}.xlsx")
                    
            except Exception as e: 
                st.error(f"异常: {e}")

# ============================================================
# 模块2：阈值评估
# ============================================================
def page_evaluate():
    st.markdown('<div class="main-title">🔍 阈值评估与数据分流</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">物理极值硬阈值 → IQR统计异常 → 组合模式孤立森林告警</div>', unsafe_allow_html=True)

    m = load_metadata()
    batches = [b for b in m.get("batches", []) if b.get("status") in ("imported", "evaluated")]
    batches.sort(key=lambda x: x.get("import_time", ""), reverse=True)
    opts = {f"[{b['batch_id']}] {b.get('batch_name','')} ({b['total_rows']}条)": b["batch_id"] for b in batches[:10]}
    if not opts: st.warning("暂无批次，请先导入数据"); return
    bid = opts[st.selectbox("选择批次", list(opts.keys()))]

    if st.button("▶️ 开始评估", type="primary"):
        with st.spinner("⏳ 评估中..."):
            try:
                rf = os.path.join(RAW_DIR, f"raw_batch_{bid}.json")
                if not os.path.exists(rf): st.error("原始数据不存在"); return
                rd = load_json(rf)
                records = rd.get("data", [])
                if not records: st.error("无数据"); return

                try:
                    result = run_evaluation(records, bid)
                except ValueError as e:
                    st.error(f"❌ 物理极限配置错误: {e}")
                    st.info("请检查 config/physical_limits.json 中是否配置了所有指标的物理极限")
                    return

                save_json({"batch_id": bid, "count": len(result["qualified"]), "data": result["qualified"]},
                    os.path.join(QUALIFIED_DIR, f"qualified_{bid}.json"))
                save_json({"batch_id": bid, "count": len(result["unqualified"]), "data": result["unqualified"]},
                    os.path.join(UNQUALIFIED_DIR, f"unqualified_{bid}.json"))
                for b in m["batches"]:
                    if b["batch_id"] == bid:
                        b["status"] = "evaluated"
                        b["qualified_count"] = len(result["qualified"])
                        b["unqualified_count"] = len(result["unqualified"])
                        b["pass_rate"] = result["rate"]
                save_metadata(m)

                # 评估概览
                st.markdown("<div class='section-title'>评估结果概览</div>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📊 总数据", result["total"])
                c2.metric("✅ 合格", result["pass"], f"{result['rate']}%")
                c3.metric("❌ 物理越限", result["phy_fail"])
                c4.metric("⚠️ IQR异常", result["stat_fail"])

                # 组合模式告警看板（独立，不参与分流）
                if result.get("combination_alerts"):
                    alerts = result["combination_alerts"]
                    gray_mold = [a for a in alerts if a["模式"] == "灰霉病风险"]
                    etiolation = [a for a in alerts if a["模式"] == "徒长风险"]
                    
                    st.markdown("<div class='section-title'>🚨 组合模式告警（孤立森林检测，仅看板提示）</div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.error(f"🌡️💧 灰霉病风险: {len(gray_mold)} 条（昼温+湿度+光照强度异常组合）")
                        if gray_mold: st.dataframe(pd.DataFrame(gray_mold), hide_index=True)
                    with c2:
                        st.warning(f"🌱 徒长风险: {len(etiolation)} 条（株高+茎粗异常组合）")
                        if etiolation: st.dataframe(pd.DataFrame(etiolation), hide_index=True)

                # 分组卡片（按指标，不再按时期）
                st.markdown(f"<div class='section-title'>分组评估详情（共 {len(result['groups'])} 个指标）</div>", unsafe_allow_html=True)
                groups = result["groups"]
                for i in range(0, len(groups), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i+j >= len(groups): break
                        g = groups[i+j]
                        with cols[j]:
                            with st.container():
                                if g['phy_fail'] > 0:
                                    status_tag = f"{g['phy_fail']}条物理越限"
                                    tag_class = "tag-warn"
                                elif g['stat_fail'] > 0:
                                    status_tag = f"{g['stat_fail']}条IQR异常"
                                    tag_class = "tag-warn"
                                else:
                                    status_tag = "全部数据合格"
                                    tag_class = "tag-ok"
                                
                                method_display = g['method'] if g['method'] != "样本不足" else "n<<5未检"
                                
                                st.markdown(f"""
                                <div class='card'>
                                    <div class='card-title'>
                                        <span>📋 {g['indicator']}</span>
                                        <span><span class='tag'>n={g['total']}</span><span class='tag{tag_class}'>{status_tag}</span><span class='tag' style='background:#e3f2fd;color:#1976d2;'>{method_display}</span></span>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                li = g.get("limit", {})
                                st.write(f"**物理范围:** [{li.get('min')}, {li.get('max')}] {li.get('unit','')}")
                                
                                si = g["stat_info"]
                                if g["method"] == "IQR":
                                    st.write(f"Q1 = {si.get('Q1')} | Q3 = {si.get('Q3')} | IQR = {si.get('IQR')} | 阈值 = [{si.get('lb')}, {si.get('ub')}]")
                                elif g["method"] == "样本不足":
                                    st.write(f"💡 样本量 {g['total']} < 5，跳过IQR检测")
                                
                                if g["records"]:
                                    with st.expander(f"📑 数据明细（{len(g['records'])} 条）"):
                                        st.dataframe(pd.DataFrame([
                                            {"时期": r.get("时期",""), "维度": r.get("维度",""), "指标": r.get("指标",""),
                                             "数值": r.get("数值",""), "状态": r.get("_status","")}
                                            for r in g["records"]
                                        ]), hide_index=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                st.success("✅ 评估完成！可前往「分析交互」查看图表。")
            except Exception as e: st.error(f"异常: {e}")

# ============================================================
# 模块3：分析交互（服务生长模型）
# ============================================================
@st.cache_data(show_spinner=False)
def cached_dim_rate(bid):
    q, u = load_json(os.path.join(QUALIFIED_DIR, f"qualified_{bid}.json")), load_json(os.path.join(UNQUALIFIED_DIR, f"unqualified_{bid}.json"))
    if not q and not u: return None
    qt, qq = defaultdict(int), defaultdict(int)
    for r in (q.get("data",[]) if q else []): qt[r.get("维度","未知")]+=1; qq[r.get("维度","未知")]+=1
    for r in (u.get("data",[]) if u else []): qt[r.get("维度","未知")]+=1
    ds = sorted(qt.keys()); return {"dims": ds, "rates": [round(qq.get(d,0)/qt[d]*100,1) for d in ds]}

def page_analyze():
    st.markdown('<div class="main-title">📊 分析交互智能体</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">服务生长模型：时间序列 + 时期分布 + 异常分析</div>', unsafe_allow_html=True)

    m = load_metadata()
    eb = [b for b in m.get("batches",[]) if b.get("status")=="evaluated"]
    if not eb: st.warning("暂无已评估批次"); return
    eb.sort(key=lambda x: x.get("import_time",""), reverse=True)
    opts = {f"[{b['batch_id']}] {b.get('batch_name','')} (合格{b.get('qualified_count',0)}条)": b["batch_id"] for b in eb}
    bid = opts[st.selectbox("选择批次", list(opts.keys()), key="anal_batch")]

    st.sidebar.divider()
    st.sidebar.markdown("<p style='color:#e8f5e9; font-size:14px; font-weight:600;'>📊 分析子栏</p>", unsafe_allow_html=True)
    sub = st.sidebar.radio("", ["📈 合格率仪表", "📉 时间序列", "📦 时期分布", "⚠️ 异常数据"], label_visibility="collapsed")

    if sub == "📈 合格率仪表":
        st.markdown("<div class='section-title'>📈 合格率仪表</div>", unsafe_allow_html=True)
        dr = cached_dim_rate(bid)
        if dr:
            st.plotly_chart(plot_bar_line(dr["dims"], dr["rates"], "维度合格率", "#4caf50", "#ff5722"), use_container_width=True)
        
        # 时期合格率
        q, u = load_json(os.path.join(QUALIFIED_DIR, f"qualified_{bid}.json")), load_json(os.path.join(UNQUALIFIED_DIR, f"unqualified_{bid}.json"))
        pt, pq = defaultdict(int), defaultdict(int)
        for r in (q.get("data",[]) if q else []): pt[r.get("时期","未知")]+=1; pq[r.get("时期","未知")]+=1
        for r in (u.get("data",[]) if u else []): pt[r.get("时期","未知")]+=1
        if pt:
            ordr = ["萌芽期","展叶期","孕蕾期","开花期"]; ps = [p for p in ordr if p in pt]+sorted([p for p in pt if p not in ordr])
            rs = [round(pq.get(p,0)/pt[p]*100,1) for p in ps]
            st.plotly_chart(plot_bar_line(ps, rs, "时期合格率", "#2196f3", "#ff9800"), use_container_width=True)

    elif sub == "📉 时间序列":
        st.markdown("<div class='section-title'>📉 时间序列分析（服务生长模型）</div>", unsafe_allow_html=True)
        qf = os.path.join(QUALIFIED_DIR, f"qualified_{bid}.json")
        if os.path.exists(qf):
            with open(qf) as f: ar = json.load(f).get("data",[])
            iav = sorted(set(r.get("指标","") for r in ar if r.get("指标")))
            si = st.selectbox("📏 选择指标", iav, key="ts_ind")
            
            # 获取物理极限
            pd_ = PHYSICAL_LIMITS.get("limits",{}).get("萌芽期",{})  # 默认取萌芽期，实际应按记录时期
            li = None
            for dn, items in pd_.items():
                if si in items: li = items[si]; break
            
            if st.button("📈 生成时间序列图", type="primary"):
                fig = plot_time_series(ar, si, li or {})
                if fig: st.plotly_chart(fig, use_container_width=True)
                else: st.warning("该指标无时间序列数据")

    elif sub == "📦 时期分布":
        st.markdown("<div class='section-title'>📦 时期分布箱线图</div>", unsafe_allow_html=True)
        qf = os.path.join(QUALIFIED_DIR, f"qualified_{bid}.json")
        if os.path.exists(qf):
            with open(qf) as f: ar = json.load(f).get("data",[])
            iav = sorted(set(r.get("指标","") for r in ar if r.get("指标")))
            si = st.selectbox("📏 选择指标", iav, key="box_ind")
            if st.button("📈 生成箱线图", type="primary"):
                fig = plot_box_by_period(ar, si)
                if fig: st.plotly_chart(fig, use_container_width=True)
                else: st.warning("该指标无时期分布数据")

    elif sub == "⚠️ 异常数据":
        st.markdown("<div class='section-title'>⚠️ 异常数据集合</div>", unsafe_allow_html=True)
        uf = os.path.join(UNQUALIFIED_DIR, f"unqualified_{bid}.json")
        if os.path.exists(uf):
            with open(uf) as f: uq = json.load(f).get("data",[])
            fr = st.selectbox("筛选", ["全部","物理越限","IQR异常"])
            fl = uq
            if fr == "物理越限": fl = [r for r in uq if "物理越限" in str(r.get("_fail_type",""))]
            elif fr == "IQR异常": fl = [r for r in uq if "IQR" in str(r.get("_fail_type",""))]
            st.write(f"共 **{len(fl)}** 条")
            ps, tp = 30, max(1, (len(fl)+29)//30)
            pn = st.number_input("页码", 1, tp, 1)
            si, ei = (pn-1)*ps, min(pn*ps, len(fl))
            df_u = pd.DataFrame([{"时期":r.get("时期",""),"维度":r.get("维度",""),"指标":r.get("指标",""),"数值":r.get("数值",""),"单位":r.get("单位",""),"时间":r.get("时间",""),"异常原因":r.get("_fail_reason",""),"异常类型":r.get("_fail_type","")} for r in fl[si:ei]])
            st.dataframe(df_u, hide_index=True)
            st.caption(f"第 {pn}/{tp} 页 | {si+1}-{ei} / {len(fl)}")
            csv = pd.DataFrame([{"时期":r.get("时期",""),"维度":r.get("维度",""),"指标":r.get("指标",""),"数值":r.get("数值",""),"单位":r.get("单位",""),"时间":r.get("时间",""),"异常原因":r.get("_fail_reason",""),"异常类型":r.get("_fail_type","")} for r in fl]).to_csv(index=False, encoding="utf-8-sig")
            st.download_button("⬇️ 导出CSV", csv, file_name=f"uq_{bid}_{fr}.csv")
        else: st.info("无异常数据")

# ============================================================
# 模块4：测试数据
# ============================================================
def generate_test_data(name, total, normal_ratio, phy_ratio, stat_ratio, period_cfg, start_date):
    try: pd_ = json.loads(period_cfg)
    except: pd_ = {"萌芽期":0.25,"展叶期":0.25,"孕蕾期":0.25,"开花期":0.25}
    cfgs = []
    for period, dd in PHYSICAL_LIMITS["limits"].items():
        for dim, inds in dd.items():
            for ind, info in inds.items(): cfgs.append((period,dim,ind,info["unit"],info["min"],info["max"]))
    rows, bd = [], datetime.strptime(start_date, "%Y-%m-%d")
    np.random.seed(42)
    for i in range(total):
        prd = np.random.choice(list(pd_.keys()), p=list(pd_.values()))
        cl = [c for c in cfgs if c[0]==prd]
        if not cl: continue
        period,dim,ind,unit,pmin,pmax = cl[i%len(cl)]
        r = np.random.random()
        if r < normal_ratio: val = np.random.uniform(pmin+(pmax-pmin)*0.2, pmax-(pmax-pmin)*0.2)
        elif r < normal_ratio+phy_ratio: val = np.random.choice([pmin-np.random.uniform(5,20), pmax+np.random.uniform(5,20)])
        else:
            bg = [np.random.uniform(pmin+(pmax-pmin)*0.2, pmax-(pmax-pmin)*0.2) for _ in range(10)]
            if len(bg)>=5 and abs(stats.skew(bg)) < 1:
                mb, sb = np.mean(bg), np.std(bg)
                val = mb+np.random.choice([-1,1])*np.random.uniform(3.5,5)*sb if sb>0 else mb
            else:
                q1b,q3b = np.percentile(bg,25), np.percentile(bg,75)
                val = q3b+np.random.uniform(1.5,3)*(q3b-q1b) if np.random.random()>0.5 else q1b-np.random.uniform(1.5,3)*(q3b-q1b)
        rows.append({"时期":period,"维度":dim,"指标":ind,"数值":round(float(val),2),"单位":unit,"时间":(bd+timedelta(days=i)).strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)

def page_test():
    st.markdown('<div class="main-title">🧪 测试数据智能体</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">生成模拟数据 → 自动评估 → 查看图表（退出后自动清理）</div>', unsafe_allow_html=True)

    if "test_result" not in st.session_state: st.session_state.test_result = None

    st.markdown("<div class='section-title'>⚙️ 参数设置</div>", unsafe_allow_html=True)
    if "test_cfg" not in st.session_state:
        st.session_state.test_cfg = {"name":"test_001","total":100,"normal":0.7,"phy":0.15,"stat":0.15,
            "period":'{"萌芽期":0.25,"展叶期":0.25,"孕蕾期":0.25,"开花期":0.25}',"start":datetime.now().strftime("%Y-%m-%d")}
    cg = st.session_state.test_cfg

    name = st.text_input("🏷️ 批次名称", cg["name"], key="tname")
    total = st.number_input("📊 总记录数", 10, 5000, cg["total"], 10, key="ttotal")
    st.markdown("##### 数据类型比例（和=1.0）")
    nr = st.slider("✅ 正常", 0.0, 1.0, cg["normal"], 0.05, key="tnr")
    pr = st.slider("❌ 物理超限", 0.0, 1.0-nr, cg["phy"], 0.05, key="tpr")
    sr = st.slider("⚠️ 统计异常", 0.0, 1.0-nr-pr, min(cg["stat"],1.0-nr-pr), 0.05, key="tsr")
    st.markdown("##### 其他参数")
    pc = st.text_area("🌱 时期占比(JSON)", cg["period"], key="tpc")
    sd = st.text_input("📅 起始日期", cg["start"], key="tsd")

    if st.button("⚡ 生成并评估", type="primary"):
        with st.spinner("⏳ 生成数据并评估中..."):
            try:
                if abs(nr+pr+sr-1.0)>0.01: st.error("比例和≠1.0"); return
                df = generate_test_data(name, total, nr, pr, sr, pc, sd)
                st.session_state.test_cfg.update({"name":name,"total":total,"normal":nr,"phy":pr,"stat":sr,"period":pc,"start":sd})
                st.session_state.last_test_df = df

                try:
                    result = run_evaluation(df.to_dict("records"), "temp")
                except ValueError as e:
                    st.error(f"❌ 物理极限配置错误: {e}")
                    return

                st.session_state.test_result = result
                st.success(f"✅ 生成{len(df)}条 | 合格{result['pass']}({result['rate']}%) | 物理越限{result['phy_fail']} | IQR异常{result['stat_fail']}")
            except Exception as e: st.error(f"失败: {e}")

    if st.session_state.test_result:
        res = st.session_state.test_result
        st.markdown("<div class='section-title'>📊 自动评估结果</div>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("总数据", res["total"]); c2.metric("合格", res["pass"], f"{res['rate']}%")
        c3.metric("物理越限", res["phy_fail"]); c4.metric("IQR异常", res["stat_fail"])

        # 组合模式告警
        if res.get("combination_alerts"):
            alerts = res["combination_alerts"]
            gray_mold = [a for a in alerts if a["模式"] == "灰霉病风险"]
            etiolation = [a for a in alerts if a["模式"] == "徒长风险"]
            st.markdown("<div class='section-title'>🚨 组合模式告警</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.error(f"🌡️💧 灰霉病风险: {len(gray_mold)} 条")
                if gray_mold: st.dataframe(pd.DataFrame(gray_mold), hide_index=True)
            with c2:
                st.warning(f"🌱 徒长风险: {len(etiolation)} 条")
                if etiolation: st.dataframe(pd.DataFrame(etiolation), hide_index=True)

        # 分组卡片
        st.markdown(f"<div class='section-title'>分组详情（{len(res['groups'])} 组）</div>", unsafe_allow_html=True)
        for i in range(0, len(res["groups"]), 2):
            cols = st.columns(2)
            for j in range(2):
                if i+j >= len(res["groups"]): break
                g = res["groups"][i+j]
                with cols[j]:
                    with st.container():
                        if g['phy_fail'] > 0:
                            status_tag = f"{g['phy_fail']}条物理越限"
                            tag_class = "tag-warn"
                        elif g['stat_fail'] > 0:
                            status_tag = f"{g['stat_fail']}条IQR异常"
                            tag_class = "tag-warn"
                        else:
                            status_tag = "全部数据合格"
                            tag_class = "tag-ok"
                        method_display = g['method'] if g['method'] != "样本不足" else "n<<5未检"
                        st.markdown(f"""
                        <div class='card'>
                            <div class='card-title'>
                                <span>📋 {g['indicator']}</span>
                                <span><span class='tag'>n={g['total']}</span><span class='tag{tag_class}'>{status_tag}</span><span class='tag' style='background:#e3f2fd;color:#1976d2;'>{method_display}</span></span>
                            </div>
                        """, unsafe_allow_html=True)
                        li = g.get("limit",{})
                        st.write(f"范围: [{li.get('min')}, {li.get('max')}] {li.get('unit','')}")
                        si = g["stat_info"]
                        if g["method"]=="IQR": st.write(f"Q1={si.get('Q1')} Q3={si.get('Q3')} IQR={si.get('IQR')}")
                        elif g["method"]=="样本不足": st.write(f"💡 样本量不足，跳过IQR")
                        if g["records"]:
                            with st.expander(f"明细({len(g['records'])}条)"):
                                st.dataframe(pd.DataFrame([{"时期":r.get("时期",""),"维度":r.get("维度",""),"指标":r.get("指标",""),"数值":r.get("数值",""),"状态":r.get("_status","")} for r in g["records"]]), hide_index=True)
                        st.markdown("</div>", unsafe_allow_html=True)

        # 图表
        st.markdown("<div class='section-title'>📈 合格率图表</div>", unsafe_allow_html=True)
        dim_total, dim_q = defaultdict(int), defaultdict(int)
        for r in res["qualified"]: dim_total[r.get("维度","未知")]+=1; dim_q[r.get("维度","未知")]+=1
        for r in res["unqualified"]: dim_total[r.get("维度","未知")]+=1
        if dim_total:
            ds = sorted(dim_total.keys()); rs = [round(dim_q.get(d,0)/dim_total[d]*100,1) for d in ds]
            st.plotly_chart(plot_bar_line(ds, rs, "维度合格率", "#4caf50", "#ff5722"), use_container_width=True)

        if res["unqualified"]:
            st.markdown("<div class='section-title'>⚠️ 异常数据</div>", unsafe_allow_html=True)
            df_u = pd.DataFrame([{"时期":r.get("时期",""),"维度":r.get("维度",""),"指标":r.get("指标",""),"数值":r.get("数值",""),"原因":r.get("_fail_reason",""),"类型":r.get("_fail_type","")} for r in res["unqualified"]])
            st.dataframe(df_u, hide_index=True)
            csv = df_u.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("⬇️ 导出异常数据CSV", csv, file_name=f"test_unqualified_{name}.csv")

        st.info("💡 以上为临时结果，退出测试数据页面或刷新后自动清理。")

# ============================================================
# 模块5：数据总库
# ============================================================
def page_database():
    st.markdown('<div class="main-title">🗄️ 数据总库</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">批次管理 | 汇总统计 | 数据导出</div>', unsafe_allow_html=True)

    m = load_metadata()
    fb = [b for b in m.get("batches",[]) if not b.get("is_test",False)]

    st.sidebar.divider()
    st.sidebar.markdown("<p style='color:#e8f5e9; font-size:14px; font-weight:600;'>🗄️ 总库子栏</p>", unsafe_allow_html=True)
    sub = st.sidebar.radio("", ["📋 批次列表", "📈 汇总图表", "📦 数据导出"], label_visibility="collapsed")

    if sub == "📋 批次列表":
        st.markdown("<div class='section-title'>📋 批次列表</div>", unsafe_allow_html=True)
        if not fb: st.info("暂无批次"); return
        st.write(f"共 **{len(fb)}** 个正式批次")
        for b in fb:
            with st.container():
                c1,c2,c3,c4,c5,c6 = st.columns([1.2,2.2,0.8,0.8,0.6,0.6])
                with c1: st.code(b["batch_id"])
                with c2: st.write(b.get("original_file","-"))
                with c3: st.write(f"{b.get('total_rows',0)}条")
                rt = b.get('pass_rate','-')
                with c4: st.write(f"{rt}%" if isinstance(rt,(int,float)) else "-")
                with c5:
                    ep = os.path.join(RAW_DIR, f"raw_batch_{b['batch_id']}_export.xlsx")
                    if os.path.exists(ep):
                        with open(ep,"rb") as f: st.download_button("⬇️", f, file_name=f"raw_{b['batch_id']}.xlsx", key=f"dle_{b['batch_id']}")
                    else: st.write("-")
                with c6:
                    if st.button("🗑️", key=f"del_{b['batch_id']}"):
                        for d,pf in [(RAW_DIR,"raw_batch_"),(QUALIFIED_DIR,"qualified_"),(UNQUALIFIED_DIR,"unqualified_")]:
                            fp = os.path.join(d,f"{pf}{b['batch_id']}.json")
                            if os.path.exists(fp): os.remove(fp)
                        ep2 = os.path.join(RAW_DIR, f"raw_batch_{b['batch_id']}_export.xlsx")
                        if os.path.exists(ep2): os.remove(ep2)
                        m["batches"] = [x for x in m["batches"] if x["batch_id"]!=b["batch_id"]]
                        save_metadata(m); st.rerun()
            st.divider()

    elif sub == "📈 汇总图表":
        st.markdown("<div class='section-title'>📈 汇总图表</div>", unsafe_allow_html=True)
        aq, au = [], []
        for b in fb:
            q = load_json(os.path.join(QUALIFIED_DIR,f"qualified_{b['batch_id']}.json"))
            u = load_json(os.path.join(UNQUALIFIED_DIR,f"unqualified_{b['batch_id']}.json"))
            if q: aq.extend(q.get("data",[]))
            if u: au.extend(u.get("data",[]))
        if not aq and not au: st.info("无数据"); return

        ct, cq = defaultdict(int), defaultdict(int)
        for r in aq: k=f"{r.get('时期','未知')}-{r.get('维度','未知')}"; ct[k]+=1; cq[k]+=1
        for r in au: k=f"{r.get('时期','未知')}-{r.get('维度','未知')}"; ct[k]+=1
        cs = sorted(ct.keys()); rs = [round(cq.get(c,0)/ct[c]*100,1) for c in cs]
        st.plotly_chart(plot_bar_line(cs, rs, "时期+维度 合格率", "#4caf50", "#ff5722"), use_container_width=True)

        vl = defaultdict(lambda: defaultdict(int))
        for r in aq+au: vl[r.get("时期","未知")][r.get("维度","未知")]+=1
        pv = sorted(set(r.get("时期","") for r in aq+au if r.get("时期"))); dv = ["环境","土壤","生长状况"]
        dim_data = {d: [vl[p].get(d,0) for p in pv] for d in dv}
        st.plotly_chart(plot_stacked_bar(pv, dim_data, "时期+维度 数据量堆积"), use_container_width=True)

        fbs = sorted(fb, key=lambda x:x.get("import_time",""))
        lbs = [b["batch_id"] for b in fbs]; rts = [b.get("pass_rate",0) for b in fbs]
        st.plotly_chart(plot_trend(lbs, rts, "总体合格率趋势"), use_container_width=True)

    elif sub == "📦 数据导出":
        st.markdown("<div class='section-title'>📦 数据导出</div>", unsafe_allow_html=True)
        if st.button("📦 导出全部合格数据(ZIP)", type="primary"):
            with st.spinner("打包中..."):
                import tempfile
                zp = os.path.join(DATA_DIR, "all_qualified.zip")
                with zipfile.ZipFile(zp, "w") as zf:
                    for b in fb:
                        bid = b["batch_id"]
                        fp = os.path.join(QUALIFIED_DIR, f"qualified_{bid}.json")
                        if not os.path.exists(fp): continue
                        qd = load_json(fp)
                        records = qd.get("data", []) if qd else []
                        if not records: continue
                        df_q = pd.DataFrame(records)
                        drop_cols = [c for c in df_q.columns if c.startswith("_")]
                        if drop_cols: df_q = df_q.drop(columns=drop_cols)
                        tmp_xlsx = os.path.join(tempfile.gettempdir(), f"qualified_{bid}.xlsx")
                        df_q.to_excel(tmp_xlsx, index=False, engine="openpyxl")
                        zf.write(tmp_xlsx, f"{bid}/qualified_{bid}.xlsx")
                        if os.path.exists(tmp_xlsx): os.remove(tmp_xlsx)
                with open(zp, "rb") as f: st.download_button("⬇️ 下载ZIP", f, file_name="all_qualified.zip")

# ============================================================
# 主入口
# ============================================================
st.markdown('<div class="main-title">🌷 百合生长模型数据监测智能体平台</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">低代码 · 多智能体协同 · 自适应统计异常检测 V3.0</div>', unsafe_allow_html=True)
st.divider()

page = sidebar_nav()

if page != "test" and st.session_state.get("test_result"):
    st.session_state.test_result = None

if page == "import": page_import()
elif page == "evaluate": page_evaluate()
elif page == "analyze": page_analyze()
elif page == "test": page_test()
elif page == "database": page_database()
