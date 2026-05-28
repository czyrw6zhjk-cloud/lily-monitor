"""
百合生长模型数据监测智能体平台 V2.3
增强：指标专属统计异常检测（z-score / IQR / MAD / 孤立森林）+ 联合异常告警
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
# ==================== 统计异常检测模块（新增） ====================
from scipy import stats
from sklearn.ensemble import IsolationForest
# ================================================================
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

# ==================== 加载统计方法配置（新增） ====================
try:
    with open(os.path.join(CONFIG_DIR, "statistical_methods.json"), "r", encoding="utf-8") as f:
        STAT_CONFIG = json.load(f)["百合温室统计检测配置"]
except Exception:
    STAT_CONFIG = {}
# ================================================================

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .main-title { font-size: 24px; font-weight: 700; color: #1b5e20; text-align: center; padding: 8px 0; }
    .subtitle { text-align: center; color: #689f38; font-size: 13px; margin-bottom: 16px; }
    .section-title { font-size: 17px; font-weight: 600; color: #2e7d32; padding: 12px 0 8px 0; border-bottom: 2px solid #c8e6c9; margin: 8px 0 12px 0; }
    .hint-box { background: #f1f8e9; border-left: 4px solid #689f38; padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 6px 0 12px 0; color: #33691e; font-size: 13px; }
    .card { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin: 8px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
    .card-title { font-size: 15px; font-weight: 600; color: #333; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    .tag { background: #e8eaf6; color: #3f51b5; font-size: 11px; padding: 2px 8px; border-radius: 12px; margin-left: 6px; }
    .tag-ok { background: #e8f5e9; color: #2e7d32; }
    .tag-warn { background: #fff3e0; color: #e65100; }
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
    "时期": {"required": True, "keywords": ["时期", "生长时期", "阶段", "生长期", "period", "stage", "phase"]},
    "维度": {"required": True, "keywords": ["维度", "类型", "category", "dimension", "type", "类别"]},
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

def check_physical_limit(period, dimension, indicator, value):
    pd_ = PHYSICAL_LIMITS.get("limits", {}).get(period, {})
    dd = pd_.get(dimension, {})
    ii = dd.get(indicator)
    if ii is None:
        for dn, items in pd_.items():
            if indicator in items: ii = items[indicator]; break
    if ii is None: return True, "无极限定义", {"min": None, "max": None, "unit": ""}
    mn, mx, un = ii.get("min"), ii.get("max"), ii.get("unit", "")
    li = {"min": mn, "max": mx, "unit": un}
    if mn is not None and value < mn: return False, f"物理越限: {value} < {mn}", li
    if mx is not None and value > mx: return False, f"物理越限: {value} > {mx}", li
    return True, "合格", li


# ==================== 统计异常检测（增强版） ====================
def detect_statistical_anomaly(values, indicator=""):
    """
    增强版：支持指标专属方法配置（z-score / IQR / MAD / 孤立森林）
    返回: (异常掩码列表, 方法名, 统计信息dict)
    """
    n = len(values)
    if n < 5:
        return [False]*n, "跳过", {"method": "跳过", "note": "n<<5"}  # 修复：n<<5 → n<<5
    
    arr = np.array(values, dtype=float)
    
    # 指标专属配置
    cfg = STAT_CONFIG.get(indicator, {})
    method = cfg.get("method", "")
    
    # 无配置时按偏度自适应兜底
    if not method:
        sk = stats.skew(arr)
        if abs(sk) < 1.0:
            method = "zscore"
        else:
            method = "iqr"
    
    if method == "zscore":
        mu, sigma = np.mean(arr), np.std(arr, ddof=1)
        if sigma == 0:
            return [False]*n, "z-score", {"method": "z-score", "note": "sg=0", "mu": round(float(mu), 4), "sigma": 0}
        z = np.abs((arr - mu) / sigma)
        threshold = cfg.get("threshold", 3.0)
        an = (z > threshold).tolist()
        return an, "z-score", {
            "method": "z-score", "mu": round(float(mu), 4), "sigma": round(float(sigma), 4),
            "range": [round(float(mu - threshold * sigma), 4), round(float(mu + threshold * sigma), 4)],
            "n": n, "anomaly": sum(an)
        }
    
    elif method == "mad":
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad == 0:
            return [False]*n, "MAD", {"method": "MAD", "note": "MAD=0", "median": round(float(median), 4)}
        modified_z = np.abs(0.6745 * (arr - median) / mad)
        threshold = cfg.get("threshold", 3.5)
        an = (modified_z > threshold).tolist()
        return an, "MAD", {
            "method": "MAD", "median": round(float(median), 4), "MAD": round(float(mad), 4),
            "threshold": threshold, "n": n, "anomaly": sum(an)
        }
    
    elif method == "isolation_forest":
        if n < 10:
            # 样本不足时回退到 IQR
            q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
            iq = q3 - q1
            lb, ub = q1 - 1.5 * iq, q3 + 1.5 * iq
            an = ((arr < lb) | (arr > ub)).tolist()
            return an, "IQR(回退)", {
                "method": "IQR(回退)", "note": "孤立森林需n≥10",
                "Q1": round(float(q1), 4), "Q3": round(float(q3), 4), "IQR": round(float(iq), 4),
                "lb": round(float(lb), 4), "ub": round(float(ub), 4), "n": n, "anomaly": sum(an)
            }
        contamination = cfg.get("contamination", 0.05)
        clf = IsolationForest(contamination=contamination, random_state=42)
        an = (clf.fit_predict(arr.reshape(-1, 1)) == -1).tolist()
        return an, "孤立森林", {
            "method": "孤立森林", "contamination": contamination,
            "n": n, "anomaly": sum(an)
        }
    
    else:  # iqr default
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iq = q3 - q1
        k = cfg.get("k", 1.5)
        lb, ub = q1 - k * iq, q3 + k * iq
        an = ((arr < lb) | (arr > ub)).tolist()
        return an, "IQR", {
            "method": "IQR", "Q1": round(float(q1), 4), "Q3": round(float(q3), 4),
            "IQR": round(float(iq), 4), "lb": round(float(lb), 4), "ub": round(float(ub), 4),
            "n": n, "anomaly": sum(an)
        }
# ============================================================


# ============================================================
# 核心评估逻辑（增强版：联合异常检测）
# ============================================================
def run_evaluation(records, batch_id="temp"):
    """
    阈值评估：
    1. 所有数据先经过物理极值判断
    2. 物理正常的数据进入统计异常检测
    3. 高温高湿看板提示（不参与分流）
    """
    # 按时期+指标分组
    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        groups[(rec.get("时期", "未知"), rec.get("指标", "未知"))].append({"idx": idx, "record": rec})
    
    # 记录每个idx的状态
    idx_phy_ok = {}       # idx -> bool
    idx_phy_reason = {}   # idx -> str
    idx_stat_anomaly = {} # idx -> bool
    idx_stat_method = {}  # idx -> str
    idx_limit = {}        # idx -> dict
    
    group_results = []
    total_phy_fail = 0
    
    # 第一轮：物理检查 + 统计检测
    for (period, indicator), items in sorted(groups.items()):
        phy_passed, phy_failed = [], []
        
        for it in items:
            rec, val = it["record"], it["record"].get("数值")
            idx = it["idx"]
            if not isinstance(val, (int, float)):
                try: val = float(val)
                except: continue
            ok, reason, limit_info = check_physical_limit(
                rec.get("时期", ""), rec.get("维度", ""), rec.get("指标", ""), val
            )
            idx_limit[idx] = limit_info
            if ok:
                phy_passed.append({"record": rec, "value": val, "idx": idx})
                idx_phy_ok[idx] = True
            else:
                phy_failed.append({"record": rec, "reason": reason, "idx": idx})
                idx_phy_ok[idx] = False
                idx_phy_reason[idx] = reason
                total_phy_fail += 1
        
        # 统计检测（仅对物理正常的数据）
        stat_info = {"method": "样本不足", "note": "n<<5，跳过统计检测"}
        anomalies = [False] * len(phy_passed)
        method_name = "样本不足"
        
        if len(phy_passed) >= 5:
            passed_vals = [p["value"] for p in phy_passed]
            anomalies, method_name, stat_info = detect_statistical_anomaly(passed_vals, indicator)
            for i, p in enumerate(phy_passed):
                idx = p["idx"]
                idx_stat_anomaly[idx] = anomalies[i]
                idx_stat_method[idx] = method_name
        else:
            for p in phy_passed:
                idx_stat_anomaly[p["idx"]] = False
                idx_stat_method[p["idx"]] = "样本不足"
        
        # 组装分组记录
        record_status = []
        for p in phy_passed:
            idx = p["idx"]
            stat = "统计异常" if idx_stat_anomaly.get(idx, False) else "合格"
            record_status.append({**p["record"], "_status": stat, "__idx": idx})
        for f in phy_failed:
            record_status.append({**f["record"], "_status": "物理越限", "__idx": f["idx"]})
        
        # 统计该分组的问题数量
        phy_fail_count = len(phy_failed)
        stat_fail_count = sum(1 for r in record_status if r.get("_status") == "统计异常")
        
        group_results.append({
            "period": period, 
            "indicator": indicator,
            "dimension": items[0]["record"].get("维度", ""),
            "total": len(items), 
            "phy_pass": len(phy_passed), 
            "phy_fail": phy_fail_count,
            "stat_fail": stat_fail_count,
            "method": method_name, 
            "stat_info": stat_info,
            "limit": limit_info if phy_failed else (phy_passed[0]["record"] if phy_passed else {}),
            "records": [{k: v for k, v in r.items() if not k.startswith("__")} for r in record_status[:50]]
        })
    
    # 最终分流（所有数据都经过物理判断）
    qualified_records, unqualified_records = [], []
    total_stat_fail = 0
    total_pass = 0
    
    for idx, rec in enumerate(records):
        val = rec.get("数值")
        if not isinstance(val, (int, float)):
            try: val = float(val)
            except: continue
        
        if not idx_phy_ok.get(idx, False):
            unqualified_records.append({
                **rec, 
                "_fail_reason": idx_phy_reason.get(idx, "物理越限"),
                "_fail_type": "物理越限", 
                "_batch_id": batch_id
            })
            continue
        
        is_stat = idx_stat_anomaly.get(idx, False)
        if is_stat:
            method_name = idx_stat_method.get(idx, "统计")
            unqualified_records.append({
                **rec, 
                "_fail_reason": "统计异常",
                "_fail_type": f"统计异常({method_name})", 
                "_batch_id": batch_id
            })
            total_stat_fail += 1
        else:
            qualified_records.append(rec)
            total_pass += 1
    
    # 高温高湿看板提示检测（不参与分流，仅提示）
    high_temp_high_humidity = []
    for rec in records:
        indicator = rec.get("指标", "")
        val = rec.get("数值")
        try: val = float(val)
        except: continue
        
        # 检查昼温/环境温度在5-30范围内
        if indicator in ["昼温", "环境温度"] and 5 <= val <= 30:
            time_key = rec.get("时间", rec.get("时期", ""))
            for r2 in records:
                if r2.get("时间", r2.get("时期", "")) == time_key:
                    if r2.get("指标", "") in ["湿度", "环境相对湿度"]:
                        try:
                            h_val = float(r2.get("数值", 0))
                            if 85 <= h_val <= 100:
                                high_temp_high_humidity.append({
                                    "时间": time_key,
                                    "时期": rec.get("时期", ""),
                                    "昼温": val,
                                    "湿度": h_val
                                })
                        except:
                            pass
    
    # 去重
    seen = set()
    hthh_unique = []
    for item in high_temp_high_humidity:
        key = (item["时间"], item["昼温"], item["湿度"])
        if key not in seen:
            seen.add(key)
            hthh_unique.append(item)
    
    return {
        "qualified": qualified_records,
        "unqualified": unqualified_records,
        "groups": group_results,
        "total": len(records),
        "pass": total_pass,
        "phy_fail": total_phy_fail,
        "stat_fail": total_stat_fail,
        "rate": round(total_pass / len(records) * 100, 1) if records else 0,
        "high_temp_high_humidity": hthh_unique
    }

# ============================================================
# Plotly 图表函数（前端渲染，天然支持中文）
# ============================================================
def plot_bar_line(x_labels, values, title, cbar="#4caf50", cline="#ff5722", ylabel="合格率 (%)"):
    """柱状+折线混合图"""
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

def plot_wave(records, sel_period, sel_indicator, limit_info):
    """波动图"""
    times = [r.get("时间", "") for r in records]
    values = [r["数值"] for r in records if isinstance(r.get("数值"), (int, float))]
    unit = limit_info.get("单位", "") if limit_info else ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(values))), y=values, mode="lines+markers",
        name=sel_indicator, line=dict(color="#4caf50", width=1.5), marker=dict(size=5)))
    if limit_info:
        if limit_info.get("min") is not None:
            fig.add_hline(y=limit_info["min"], line_dash="dash", line_color="#f44336",
                annotation_text=f"下限 {limit_info['min']}{unit}", annotation_position="right")
        if limit_info.get("max") is not None:
            fig.add_hline(y=limit_info["max"], line_dash="dash", line_color="#f44336",
                annotation_text=f"上限 {limit_info['max']}{unit}", annotation_position="right")
    n = len(times)
    tick_vals = list(range(0, n, max(1, n // 12)))
    tick_text = [times[i] if i < n else "" for i in tick_vals]
    fig.update_layout(title=f"{sel_period} - {sel_indicator} 数据波动",
        xaxis=dict(title="时间", tickmode="array", tickvals=tick_vals, ticktext=tick_text, tickangle=30),
        yaxis=dict(title=f"{sel_indicator} ({unit})", gridcolor="rgba(0,0,0,0.1)"),
        height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60))
    return fig

def plot_group_bar(labels, values, title):
    """时期+维度合格率柱状图"""
    cls_map = {"环境": "#4caf50", "土壤": "#2196f3", "生长状况": "#ff9800"}
    colors = [cls_map.get(l.split("-")[1] if "-" in l else l, "#999") for l in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=colors, opacity=0.75, text=[f"{v}%" for v in values], textposition="outside"))
    fig.update_layout(title=title, yaxis=dict(title="合格率 (%)", range=[0, 105], gridcolor="rgba(0,0,0,0.1)"),
        xaxis=dict(title="", tickangle=25), height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=80))
    return fig

def plot_stacked_bar(categories, dim_data, title):
    """堆积柱状图"""
    cls_map = {"环境": "#4caf50", "土壤": "#2196f3", "生长状况": "#ff9800"}
    fig = go.Figure()
    for dim, vals in dim_data.items():
        fig.add_trace(go.Bar(name=dim, x=categories, y=vals, marker_color=cls_map.get(dim, "#999"), opacity=0.75))
    fig.update_layout(barmode="stack", title=title,
        yaxis=dict(title="数据条数", gridcolor="rgba(0,0,0,0.1)"),
        xaxis=dict(title=""), height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40))
    return fig

def plot_trend(x_labels, values, title):
    """合格率趋势折线图"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(x_labels))), y=values, mode="lines+markers",
        name="合格率", line=dict(color="#4caf50", width=2.5), marker=dict(size=8, color="#4caf50")))
    fig.update_layout(title=title, yaxis=dict(title="合格率 (%)", range=[0, 105], gridcolor="rgba(0,0,0,0.1)"),
        xaxis=dict(title="批次号", tickmode="array", tickvals=list(range(len(x_labels))), ticktext=x_labels),
        height=420, margin=dict(t=80, b=60))
    for i, v in enumerate(values):
        fig.add_annotation(x=i, y=v+3, text=f"{v}%", showarrow=False, font=dict(size=10))
    return fig

# ============================================================
# 侧边栏导航
# ============================================================
def sidebar_nav():
    st.sidebar.markdown("<div style='text-align:center; padding:6px 0;'><p style='color:#e8f5e9; font-size:22px; font-weight:700; margin:0;'>🌷 百合监测</p><p style='color:rgba(232,245,233,0.5); font-size:11px; margin:2px 0 0 0;'>低代码智能体平台 V2.3</p></div>", unsafe_allow_html=True)
    st.sidebar.divider()
    pages = {"import": "📥 数据导入", "evaluate": "🔍 阈值评估", "analyze": "📊 分析交互", "test": "🧪 测试数据", "database": "🗄️ 数据总库"}
    sel = st.sidebar.radio("导航", list(pages.keys()), format_func=lambda x: pages[x], label_visibility="collapsed")
    st.sidebar.divider()
    return sel

# ============================================================
# 模块1：数据导入
# ============================================================
def page_import():
    st.markdown('<div class="main-title">📥 数据导入智能体</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">支持 .xlsx / .csv | 智能列名匹配 | 空值校验 | 单批次上限 5000 行</div>', unsafe_allow_html=True)

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
                valid_rows, error_details = [], []
                for idx, row in df_renamed.iterrows():
                    errs = []
                    for f in ["时期", "维度", "指标", "数值"]:
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
                    if errs: error_details.append({"行": int(idx)+2, "时期": str(row.get("时期","")), "维度": str(row.get("维度","")), "指标": str(row.get("指标","")), "数值": str(row.get("数值","")), "错误": ";".join(errs)})
                    else: valid_rows.append(row.to_dict())
                if not valid_rows: st.error(f"全部无效"); return

                batch_id = get_next_batch_id()
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_json({"batch_id": batch_id, "import_time": ts, "original_file": uploaded.name,
                    "total_rows": total, "valid_rows": len(valid_rows), "error_rows_count": len(error_details),
                    "column_mapping": col_map, "data": valid_rows}, os.path.join(RAW_DIR, f"raw_batch_{batch_id}.json"))
                m = load_metadata()
                m["batches"].append({"batch_id": batch_id, "batch_name": f"批次_{batch_id}", "import_time": ts,
                    "original_file": uploaded.name, "total_rows": total, "valid_rows": len(valid_rows),
                    "status": "imported", "is_test": False})
                save_metadata(m)

                pr = len(valid_rows)/total*100
                st.success(f"✅ 导入成功！批次 {batch_id} | 总{total} | 有效{len(valid_rows)} | 合格率{pr:.1f}%")
                if error_details:
                    with st.expander(f"⚠️ 错误行 ({len(error_details)} 条)"): st.dataframe(pd.DataFrame(error_details), hide_index=True)
                st.markdown("<div class='section-title'>预览（前10行）</div>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(valid_rows[:10]), hide_index=True)
                ep = os.path.join(RAW_DIR, f"raw_batch_{batch_id}_export.xlsx")
                pd.DataFrame(valid_rows).to_excel(ep, index=False)
                with open(ep, "rb") as f: st.download_button("⬇️ 下载原始数据", f, file_name=f"raw_{batch_id}.xlsx")
            except Exception as e: st.error(f"异常: {e}")

# ============================================================
# 模块2：阈值评估 - 卡片网格布局
# ============================================================
def page_evaluate():
    st.markdown('<div class="main-title">🔍 阈值评估与数据分流</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">物理极值判断 → 自适应统计异常检测（z-score / IQR / MAD / 孤立森林）</div>', unsafe_allow_html=True)

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

                result = run_evaluation(records, bid)
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

                st.markdown("<div class='section-title'>评估结果概览</div>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📊 总数据", result["total"])
                c2.metric("✅ 合格", result["pass"], f"{result['rate']}%")
                c3.metric("❌ 物理越限", result["phy_fail"])
                c4.metric("⚠️ 统计异常", result["stat_fail"])

                # ==================== 高温高湿看板提示（新增） ====================
                if result.get("high_temp_high_humidity"):
                    hthh = result["high_temp_high_humidity"]
                    st.warning(f"🌡️💧 检测到 **{len(hthh)}** 条高温高湿记录（昼温5-30℃且湿度85-100%），灰霉病风险预警")
                    with st.expander("查看高温高湿明细"):
                        st.dataframe(pd.DataFrame(hthh), hide_index=True)
                # ============================================================

                # 分组卡片网格（2列）
                st.markdown(f"<div class='section-title'>分组评估详情（共 {len(result['groups'])} 个分组）</div>", unsafe_allow_html=True)
                groups = result["groups"]
                for i in range(0, len(groups), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i+j >= len(groups): break
                        g = groups[i+j]
                        with cols[j]:
                            with st.container():
                                # ==================== 修改：问题统计标签（替换原"跳过"） ====================
                                # 判定该分组的状态标签
                                if g['phy_fail'] > 0:
                                    status_tag = f"{g['phy_fail']}条物理越限"
                                    tag_class = "tag-warn"
                                elif g['stat_fail'] > 0:
                                    status_tag = f"{g['stat_fail']}条统计异常"
                                    tag_class = "tag-warn"
                                else:
                                    status_tag = "全部数据合格"
                                    tag_class = "tag-ok"
                                
                                # 方法标签（样本不足时显示提示）
                                method_display = g['method']
                                if method_display == "样本不足":
                                    method_display = "n<<5未检"
                                # ===================================================================
                                
                                st.markdown(f"""
                                <div class='card'>
                                    <div class='card-title'>
                                        <span>📋 {g['period']} — {g['dimension']} — {g['indicator']}</span>
                                        <span><span class='tag'>n={g['total']}</span><span class='tag{tag_class}'>{status_tag}</span><span class='tag' style='background:#e3f2fd;color:#1976d2;'>{method_display}</span></span>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # ==================== 修改：物理范围N/A处理 ====================
                                li = g.get("limit", {})
                                min_v = li.get("min")
                                max_v = li.get("max")
                                unit_v = li.get("unit", "")
                                if min_v is None and max_v is None:
                                    st.write(f"**物理范围:** 未配置物理极限 {unit_v}")
                                else:
                                    min_show = min_v if min_v is not None else "无下限"
                                    max_show = max_v if max_v is not None else "无上限"
                                    st.write(f"**物理范围:** [{min_show}, {max_show}] {unit_v}")
                                # ============================================================
                                
                                si = g["stat_info"]
                                if g["method"]=="z-score": 
                                    st.write(f"μ = {si.get('mu')} | σ = {si.get('sigma')} | 3σ = [{si.get('range',[0,0])[0]}, {si.get('range',[0,0])[1]}]")
                                elif g["method"]=="IQR": 
                                    st.write(f"Q1 = {si.get('Q1')} | Q3 = {si.get('Q3')} | IQR = {si.get('IQR')} | 阈值 = [{si.get('lb')}, {si.get('ub')}]")
                                elif g["method"]=="MAD": 
                                    st.write(f"中位数 = {si.get('median')} | MAD = {si.get('MAD')} | 阈值 = ±{si.get('threshold', 3.5)}")
                                elif g["method"]=="孤立森林": 
                                    st.write(f"孤立森林 | 污染率 = {si.get('contamination')} | 异常数 = {si.get('anomaly')}")
                                elif g["method"]=="IQR(回退)": 
                                    st.write(f"IQR回退(孤立森林样本不足) | Q1={si.get('Q1')} Q3={si.get('Q3')} | 阈值 = [{si.get('lb')}, {si.get('ub')}]")
                                elif g["method"]=="样本不足" or g["method"]=="n<<5未检": 
                                    st.write(f"💡 样本量{n} < 5，仅通过物理极值判断，未进行统计异常检测")
                                else: 
                                    st.write(f"💡 {si.get('note','')}")
                                
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
# 模块3：分析交互 - 侧边栏下设三个子栏
# ============================================================
@st.cache_data(show_spinner=False)
def cached_dim_rate(bid):
    q, u = load_json(os.path.join(QUALIFIED_DIR, f"qualified_{bid}.json")), load_json(os.path.join(UNQUALIFIED_DIR, f"unqualified_{bid}.json"))
    if not q and not u: return None
    qt, qq = defaultdict(int), defaultdict(int)
    for r in (q.get("data",[]) if q else []): qt[r.get("维度","未知")]+=1; qq[r.get("维度","未知")]+=1
    for r in (u.get("data",[]) if u else []): qt[r.get("维度","未知")]+=1
    ds = sorted(qt.keys()); return {"dims": ds, "rates": [round(qq.get(d,0)/qt[d]*100,1) for d in ds]}

@st.cache_data(show_spinner=False)
def cached_period_rate(bid):
    q, u = load_json(os.path.join(QUALIFIED_DIR, f"qualified_{bid}.json")), load_json(os.path.join(UNQUALIFIED_DIR, f"unqualified_{bid}.json"))
    if not q and not u: return None
    pt, pq = defaultdict(int), defaultdict(int)
    for r in (q.get("data",[]) if q else []): pt[r.get("时期","未知")]+=1; pq[r.get("时期","未知")]+=1
    for r in (u.get("data",[]) if u else []): pt[r.get("时期","未知")]+=1
    ordr = ["萌芽期","展叶期","孕蕾期","开花期"]; ps = [p for p in ordr if p in pt]+sorted([p for p in pt if p not in ordr])
    return {"periods": ps, "rates": [round(pq.get(p,0)/pt[p]*100,1) for p in ps]}

@st.cache_data(show_spinner=False)
def cached_wave_data(bid, period, indicator):
    qf = os.path.join(QUALIFIED_DIR, f"qualified_{bid}.json")
    if not os.path.exists(qf): return None
    with open(qf) as f: rs = json.load(f).get("data",[])
    fl = [r for r in rs if r.get("时期")==period and r.get("指标")==indicator]
    if not fl: return None
    fl.sort(key=lambda x: x.get("时间",""))
    pd_ = PHYSICAL_LIMITS.get("limits",{}).get(period,{})
    li = None
    for dn, items in pd_.items():
        if indicator in items: li = items[indicator]; li["dimension"]=dn; break
    vs = [r["数值"] for r in fl if isinstance(r.get("数值"),(int,float))]
    st_ = {}
    if vs: st_ = {"count":len(vs),"mean":round(np.mean(vs),2),"std":round(np.std(vs),2),"min":round(min(vs),2),"max":round(max(vs),2)}
    return {"records":fl,"limit":li,"stats":st_}

def page_analyze():
    st.markdown('<div class="main-title">📊 分析交互智能体</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">合格率仪表 | 波动图表 | 异常数据集合</div>', unsafe_allow_html=True)

    m = load_metadata()
    eb = [b for b in m.get("batches",[]) if b.get("status")=="evaluated"]
    if not eb: st.warning("暂无已评估批次"); return
    eb.sort(key=lambda x: x.get("import_time",""), reverse=True)
    opts = {f"[{b['batch_id']}] {b.get('batch_name','')} (合格{b.get('qualified_count',0)}条)": b["batch_id"] for b in eb}
    bid = opts[st.selectbox("选择批次", list(opts.keys()), key="anal_batch")]

    # 侧边栏子栏
    st.sidebar.divider()
    st.sidebar.markdown("<p style='color:#e8f5e9; font-size:14px; font-weight:600;'>📊 分析子栏</p>", unsafe_allow_html=True)
    sub = st.sidebar.radio("", ["📈 合格率仪表", "📉 波动图表", "⚠️ 异常数据"], label_visibility="collapsed")

    if sub == "📈 合格率仪表":
        st.markdown("<div class='section-title'>📈 合格率仪表</div>", unsafe_allow_html=True)
        dr = cached_dim_rate(bid)
        if dr:
            st.plotly_chart(plot_bar_line(dr["dims"], dr["rates"], "维度合格率", "#4caf50", "#ff5722"), use_container_width=True)
        pr = cached_period_rate(bid)
        if pr:
            st.plotly_chart(plot_bar_line(pr["periods"], pr["rates"], "时期合格率", "#2196f3", "#ff9800"), use_container_width=True)

    elif sub == "📉 波动图表":
        st.markdown("<div class='section-title'>📉 波动图表</div>", unsafe_allow_html=True)
        qf = os.path.join(QUALIFIED_DIR, f"qualified_{bid}.json")
        if os.path.exists(qf):
            with open(qf) as f: ar = json.load(f).get("data",[])
            pav = sorted(set(r.get("时期","") for r in ar if r.get("时期")))
            iav = sorted(set(r.get("指标","") for r in ar if r.get("指标")))
            sp = st.selectbox("🌱 时期", pav, key="wvp")
            si = st.selectbox("📏 指标", iav, key="wvi")
            if st.button("📈 生成波动图", type="primary"):
                wd = cached_wave_data(bid, sp, si)
                if wd and wd["records"]:
                    st.plotly_chart(plot_wave(wd["records"], sp, si, wd.get("limit")), use_container_width=True)
                    s = wd.get("stats",{})
                    if s: st.info(f"样本{s.get('count')} | 均值{s.get('mean')} | 标准差{s.get('std')} | 范围[{s.get('min')}, {s.get('max')}]")
                else: st.warning("无数据")

    elif sub == "⚠️ 异常数据":
        st.markdown("<div class='section-title'>⚠️ 异常数据集合</div>", unsafe_allow_html=True)
        uf = os.path.join(UNQUALIFIED_DIR, f"unqualified_{bid}.json")
        if os.path.exists(uf):
            with open(uf) as f: uq = json.load(f).get("data",[])
            # ==================== 增强：增加 MAD / 孤立森林 / 联合异常 筛选（新增） ====================
            fr = st.selectbox("筛选", ["全部","物理越限","统计异常(z-score)","统计异常(IQR)","统计异常(MAD)","统计异常(孤立森林)","联合异常"])
            fl = uq
            if fr != "全部":
                if fr == "物理越限": fl = [r for r in uq if "物理越限" in str(r.get("_fail_type",""))]
                elif fr == "统计异常(z-score)": fl = [r for r in uq if "z-score" in str(r.get("_fail_type",""))]
                elif fr == "统计异常(IQR)": fl = [r for r in uq if "IQR" in str(r.get("_fail_type",""))]
                elif fr == "统计异常(MAD)": fl = [r for r in uq if "MAD" in str(r.get("_fail_type",""))]
                elif fr == "统计异常(孤立森林)": fl = [r for r in uq if "孤立森林" in str(r.get("_fail_type",""))]
                elif fr == "联合异常": fl = [r for r in uq if "联合异常" in str(r.get("_fail_type",""))]
            # ================================================================================
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
# 模块4：测试数据 - 自动生成+评估+图表，退出清理
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
            # ==================== 修复：<<1 改为 < 1 ====================
            if len(bg)>=5 and abs(stats.skew(bg)) < 1:
            # ==========================================================
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

                result = run_evaluation(df.to_dict("records"), "temp")
                st.session_state.test_result = result
                st.success(f"✅ 生成{len(df)}条 | 合格{result['pass']}({result['rate']}%) | 物理越限{result['phy_fail']} | 统计异常{result['stat_fail']}")
            except Exception as e: st.error(f"失败: {e}")

    # 展示评估结果 + 分析图表
    if st.session_state.test_result:
        res = st.session_state.test_result
        st.markdown("<div class='section-title'>📊 自动评估结果</div>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("总数据", res["total"]); c2.metric("合格", res["pass"], f"{res['rate']}%")
        c3.metric("物理越限", res["phy_fail"]); c4.metric("统计异常", res["stat_fail"])

# ==================== 高温高湿看板提示（新增） ====================
if res.get("high_temp_high_humidity"):
    hthh = res["high_temp_high_humidity"]
    st.warning(f"🌡️💧 检测到 **{len(hthh)}** 条高温高湿记录（昼温5-30℃且湿度85-100%），灰霉病风险预警")
    with st.expander("查看高温高湿明细"):
        st.dataframe(pd.DataFrame(hthh), hide_index=True)
# ============================================================

        # 分组卡片
        st.markdown(f"<div class='section-title'>分组详情（{len(res['groups'])} 组）</div>", unsafe_allow_html=True)
        for i in range(0, len(res["groups"]), 2):
            cols = st.columns(2)
            for j in range(2):
                if i+j >= len(res["groups"]): break
                g = res["groups"][i+j]
                with cols[j]:
                    with st.container():
                        st.markdown(f"""
                        <div class='card'>
                            <div class='card-title'>
                                <span>📋 {g['period']} — {g['dimension']} — {g['indicator']}</span>
                                <span><span class='tag'>n={g['total']}</span><span class='tag{' tag-warn' if g['phy_fail']>0 or g['stat_fail']>0 else ' tag-ok'}'>{f"{g['phy_fail']}条物理越限" if g['phy_fail']>0 else (f"{g['stat_fail']}条统计异常" if g['stat_fail']>0 else "全部数据合格")}</span><span class='tag' style='background:#e3f2fd;color:#1976d2;'>{g['method'] if g['method']!="样本不足" else "n<<5未检"}</span></span>
                            </div>
                        """, unsafe_allow_html=True)
                        li = g.get("limit",{})
                        st.write(f"范围: [{li.get('min','N/A')}, {li.get('max','N/A')}] {li.get('unit','')}")
                        si = g["stat_info"]
                        # ==================== 增强：支持 MAD / 孤立森林 显示（新增） ====================
                        if g["method"]=="z-score": 
                            st.write(f"μ={si.get('mu')} σ={si.get('sigma')}")
                        elif g["method"]=="IQR": 
                            st.write(f"Q1={si.get('Q1')} Q3={si.get('Q3')} IQR={si.get('IQR')}")
                        elif g["method"]=="MAD": 
                            st.write(f"中位数={si.get('median')} MAD={si.get('MAD')} 阈值=±{si.get('threshold',3.5)}")
                        elif g["method"]=="孤立森林": 
                            st.write(f"孤立森林 污染率={si.get('contamination')} 异常数={si.get('anomaly')}")
                        elif g["method"]=="IQR(回退)": 
                            st.write(f"IQR回退 Q1={si.get('Q1')} Q3={si.get('Q3')}")
                        else: 
                            st.write(f"💡 {si.get('note','')}")
                        # ================================================================================
                        if g["records"]:
                            with st.expander(f"明细({len(g['records'])}条)"):
                                st.dataframe(pd.DataFrame([{"时期":r.get("时期",""),"维度":r.get("维度",""),"指标":r.get("指标",""),"数值":r.get("数值",""),"状态":r.get("_status","")} for r in g["records"]]), hide_index=True)
                        st.markdown("</div>", unsafe_allow_html=True)

        # 自动展示图表
        st.markdown("<div class='section-title'>📈 合格率图表</div>", unsafe_allow_html=True)
        dim_total, dim_q = defaultdict(int), defaultdict(int)
        for r in res["qualified"]: dim_total[r.get("维度","未知")]+=1; dim_q[r.get("维度","未知")]+=1
        for r in res["unqualified"]: dim_total[r.get("维度","未知")]+=1
        if dim_total:
            ds = sorted(dim_total.keys()); rs = [round(dim_q.get(d,0)/dim_total[d]*100,1) for d in ds]
            st.plotly_chart(plot_bar_line(ds, rs, "维度合格率（测试数据）", "#4caf50", "#ff5722"), use_container_width=True)
        pt, pq = defaultdict(int), defaultdict(int)
        for r in res["qualified"]: pt[r.get("时期","未知")]+=1; pq[r.get("时期","未知")]+=1
        for r in res["unqualified"]: pt[r.get("时期","未知")]+=1
        if pt:
            ordr = ["萌芽期","展叶期","孕蕾期","开花期"]; ps = [p for p in ordr if p in pt]+sorted([p for p in pt if p not in ordr])
            rs2 = [round(pq.get(p,0)/pt[p]*100,1) for p in ps]
            st.plotly_chart(plot_bar_line(ps, rs2, "时期合格率（测试数据）", "#2196f3", "#ff9800"), use_container_width=True)

        # 异常数据表
        if res["unqualified"]:
            st.markdown("<div class='section-title'>⚠️ 异常数据</div>", unsafe_allow_html=True)
            df_u = pd.DataFrame([{"时期":r.get("时期",""),"维度":r.get("维度",""),"指标":r.get("指标",""),"数值":r.get("数值",""),"原因":r.get("_fail_reason",""),"类型":r.get("_fail_type","")} for r in res["unqualified"]])
            st.dataframe(df_u, hide_index=True)
            csv = df_u.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("⬇️ 导出异常数据CSV", csv, file_name=f"test_unqualified_{name}.csv")

        st.info("💡 以上为临时结果，退出测试数据页面或刷新后自动清理。如需保存请使用「数据导入」模块。")

# ============================================================
# 模块5：数据总库 - 下设三个子栏 + 原文件下载
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
                # 原文件下载
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

        # 图表1: 时期+维度合格率
        ct, cq = defaultdict(int), defaultdict(int)
        for r in aq: k=f"{r.get('时期','未知')}-{r.get('维度','未知')}"; ct[k]+=1; cq[k]+=1
        for r in au: k=f"{r.get('时期','未知')}-{r.get('维度','未知')}"; ct[k]+=1
        cs = sorted(ct.keys()); rs = [round(cq.get(c,0)/ct[c]*100,1) for c in cs]
        st.plotly_chart(plot_group_bar(cs, rs, "时期+维度 合格率柱状图"), use_container_width=True)

        # 图表2: 堆积柱状图
        vl = defaultdict(lambda: defaultdict(int))
        for r in aq+au: vl[r.get("时期","未知")][r.get("维度","未知")]+=1
        pv = sorted(set(r.get("时期","") for r in aq+au if r.get("时期"))); dv = ["环境","土壤","生长状况"]
        dim_data = {d: [vl[p].get(d,0) for p in pv] for d in dv}
        st.plotly_chart(plot_stacked_bar(pv, dim_data, "时期+维度 数据量堆积柱状图"), use_container_width=True)

        # 图表3: 合格率趋势
        fbs = sorted(fb, key=lambda x:x.get("import_time",""))
        lbs = [b["batch_id"] for b in fbs]; rts = [b.get("pass_rate",0) for b in fbs]
        st.plotly_chart(plot_trend(lbs, rts, "总体合格率趋势（按批次导入时间）"), use_container_width=True)

    elif sub == "📦 数据导出":
        st.markdown("<div class='section-title'>📦 数据导出</div>", unsafe_allow_html=True)
        if st.button("📦 导出全部合格数据(ZIP)", type="primary"):
            with st.spinner("打包中..."):
                import tempfile
                zp = os.path.join(DATA_DIR, "all_qualified.zip")
                
                with zipfile.ZipFile(zp, "w") as zf:
                    for b in fb:
                        bid = b["batch_id"]
                        # 读取合格JSON数据
                        fp = os.path.join(QUALIFIED_DIR, f"qualified_{bid}.json")
                        if not os.path.exists(fp):
                            continue
                        
                        qd = load_json(fp)
                        records = qd.get("data", []) if qd else []
                        if not records:
                            continue
                        
                        # 转为DataFrame并写入临时Excel
                        df_q = pd.DataFrame(records)
                        # 清理内部字段，只保留业务数据
                        drop_cols = [c for c in df_q.columns if c.startswith("_")]
                        if drop_cols:
                            df_q = df_q.drop(columns=drop_cols)
                        
                        tmp_xlsx = os.path.join(tempfile.gettempdir(), f"qualified_{bid}.xlsx")
                        df_q.to_excel(tmp_xlsx, index=False, engine="openpyxl")
                        
                        # 加入ZIP，按批次分文件夹
                        zf.write(tmp_xlsx, f"{bid}/qualified_{bid}.xlsx")
                        
                        # 删除临时文件
                        if os.path.exists(tmp_xlsx):
                            os.remove(tmp_xlsx)
                
                with open(zp, "rb") as f:
                    st.download_button("⬇️ 下载ZIP", f, file_name="all_qualified.zip")

# ============================================================
# 主入口
# ============================================================
st.markdown('<div class="main-title">🌷 百合生长模型数据监测智能体平台</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">低代码 · 多智能体协同 · 自适应统计异常检测 V2.3</div>', unsafe_allow_html=True)
st.divider()

page = sidebar_nav()

# 页面切换时清理测试数据临时状态
if page != "test" and st.session_state.get("test_result"):
    st.session_state.test_result = None

if page == "import": page_import()
elif page == "evaluate": page_evaluate()
elif page == "analyze": page_analyze()
elif page == "test": page_test()
elif page == "database": page_database()
