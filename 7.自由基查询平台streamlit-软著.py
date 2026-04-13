import os
import re
import warnings

import numpy as np
import pandas as pd
import psycopg
import streamlit as st

warnings.filterwarnings("ignore")

# ===================== RDKit imports =====================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem import MACCSkeys
except ImportError as e:
    st.error(f"RDKit 导入失败：{e}")
    st.error("请检查依赖配置，并优先使用 Python 3.12 部署。")
    st.stop()

# ===================== Page config =====================
st.set_page_config(
    page_title="RadLogk-AI | 自由基反应动力学数据库",
    page_icon="🧪",
    layout="wide",
)

# ===================== CSS =====================
st.markdown(
    r"""
<style>
html, body, [class*="css"] {font-family: "Times New Roman", "Microsoft YaHei", "SimHei", serif;}
body {background-color: #F6F7FB;}
a {text-decoration:none;}
.small{color:#6B7280; font-size: 13.5px; line-height:1.35;}
.hr{height:1px;background:#EEF0F3;margin:10px 0;}

.topbar{
  background: linear-gradient(90deg, #0B2A6F 0%, #133B9A 60%, #0B2A6F 100%);
  padding: 14px 18px;
  border-radius: 14px;
  color: white;
  box-shadow: 0 6px 18px rgba(0,0,0,0.10);
  margin: 10px 0 14px 0;
  display:flex;
  align-items:center;
  justify-content:center;
  position:relative;
}
.brand-badge{
  position:absolute;
  left:14px;
  width:38px;height:38px;
  border-radius:10px;
  background: rgba(255,255,255,0.14);
  border:1px solid rgba(255,255,255,0.22);
  display:flex; align-items:center; justify-content:center;
  font-weight: 800;
  font-size: 14px;
}
.topbar h1{font-size: 22px; margin: 0; font-weight: 800; text-align:center;}

.card{
  background: white;
  border-radius: 14px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  padding: 14px 14px;
  border: 1px solid #EEF0F3;
  margin-bottom: 12px;
}
.card h3{margin:0 0 8px 0; font-size: 18px;}
.section-title{
  font-size: 20px;
  font-weight: 900;
  margin: 0 0 10px 0;
  color:#0F172A;
}

.pill{
  display:inline-block; padding:6px 10px; border-radius: 999px;
  background:#F1F5F9; border:1px solid #E2E8F0;
  font-size: 13.5px; margin-right: 6px; margin-bottom: 8px;
}

.entry-head{
  font-weight: 900;
  font-size: 18px;
  color:#0F172A;
  margin-bottom: 10px;
}
.vbox{
  background:#F8FAFC;
  border:1px solid #E5E7EB;
  border-radius: 12px;
  padding: 10px 12px;
  margin-bottom: 8px;
}
.vtitle{
  font-size: 13.5px;
  color:#64748B;
  margin-bottom: 4px;
  font-weight: 700;
}
.vvalue{
  font-size: 17px;
  color:#0F172A;
  font-weight: 800;
  word-break: break-word;
  line-height:1.25;
}

.desc-grid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap:10px;
}
.desc-box{
  background:#F8FAFC;
  border:1px solid #E5E7EB;
  border-radius: 12px;
  padding: 10px 12px;
}
.desc-name{
  font-size: 14px;
  color:#64748B;
  font-weight: 800;
  margin-bottom: 4px;
}
.desc-val{
  font-size: 18px;
  color:#0F172A;
  font-weight: 900;
}

.mono-wrap{
  font-family: "Courier New", Courier, monospace;
  font-size: 14.5px;
  line-height: 1.35;
  background: #F8FAFC;
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 10px 12px;
  white-space: pre-wrap;
  word-break: break-all;
}

.stButton>button {border-radius: 10px; padding: 10px 12px; width: 100%; font-weight: 900;}
</style>
""",
    unsafe_allow_html=True,
)

# ===================== Files =====================
RADICAL_FILES = {
    "羟基自由基 (·OH)": "Hydroxyl_radical_ultimate.csv",
    "硫酸根自由基 (SO4•−)": "Sulfate_radical_ultimate.csv",
    "碳酸根自由基 (CO3•−)": "Carbonate_radical_ultimate.csv",
    "臭氧 (O3)": "Ozone_ultimate.csv",
    "游离氯 (HOCl/OCl−)": "FreeChlorine_ultimate.csv",
    "氯自由基 (Cl•)": "ChlorineRadical_ultimate.csv",
    "二氯自由基 (Cl2•−)": "DichlorineRadical_ultimate.csv",
    "单线态氧 (1O2)": "SingletOxygen_ultimate.csv"
}
REQUIRED_COLS = ["Chemical compound", "Cas", "Smiles", "Logk", "Chemical_class_27", "Ph", "T", "Ref"]

# ===================== Secrets helpers =====================
def get_dev_key() -> str:
    try:
        return str(st.secrets["RADLOGK_DEV_KEY"])
    except Exception:
        return os.environ.get("RADLOGK_DEV_KEY", "")

def get_postgres_conninfo() -> dict:
    if "postgres" not in st.secrets:
        raise RuntimeError("未在 Streamlit Secrets 中配置 PostgreSQL 连接信息。")
    pg = st.secrets["postgres"]
    return {
        "host": pg["host"],
        "port": int(pg.get("port", 5432)),
        "dbname": pg["dbname"],
        "user": pg["user"],
        "password": pg["password"],
        "sslmode": pg.get("sslmode", "require"),
    }

@st.cache_resource
def get_db_connection():
    conninfo = get_postgres_conninfo()
    return psycopg.connect(**conninfo)

def db_init():
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS app_metrics (
                metric_key TEXT PRIMARY KEY,
                metric_value BIGINT NOT NULL DEFAULT 0
            )
        """)
        cur.execute("""
            INSERT INTO app_metrics (metric_key, metric_value)
            VALUES
                ('visits', 0),
                ('queries', 0),
                ('downloads', 0)
            ON CONFLICT (metric_key) DO NOTHING
        """)
    conn.commit()

def db_get_all_metrics() -> dict:
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT metric_key, metric_value FROM app_metrics")
        rows = cur.fetchall()
    out = {"visits": 0, "queries": 0, "downloads": 0}
    for k, v in rows:
        out[str(k)] = int(v)
    return out

def db_inc_metric(metric_key: str, n: int = 1):
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO app_metrics (metric_key, metric_value)
            VALUES (%s, %s)
            ON CONFLICT (metric_key)
            DO UPDATE SET metric_value = app_metrics.metric_value + EXCLUDED.metric_value
        """, (metric_key, int(n)))
    conn.commit()

# ===================== Persistent metrics init =====================
try:
    db_init()
except Exception as e:
    st.error(f"数据库初始化失败：{e}")
    st.stop()

# visits: count once per session
if "visit_counted" not in st.session_state:
    st.session_state["visit_counted"] = True
    try:
        db_inc_metric("visits", 1)
    except Exception as e:
        st.warning(f"访问次数更新失败：{e}")

# calculation cache
if "calc_cache" not in st.session_state:
    st.session_state["calc_cache"] = {}

if "last_results" not in st.session_state:
    st.session_state["last_results"] = None
if "last_system" not in st.session_state:
    st.session_state["last_system"] = list(RADICAL_FILES.keys())[0]
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "dev_unlocked" not in st.session_state:
    st.session_state["dev_unlocked"] = False

# ===================== Query params helper =====================
def get_query_params():
    try:
        qp = st.query_params
        return dict(qp)
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}

def qp_has_dev_flag() -> bool:
    qp = get_query_params()
    v = None
    if "dev" in qp:
        v = qp["dev"]
        if isinstance(v, list) and len(v) > 0:
            v = v[0]
    return str(v).strip().lower() in ["1", "true", "yes", "on"]

# ===================== Utils =====================
def norm_text(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("").map(lambda x: x.strip()).replace({"nan": "", "NaN": ""})

def is_cas_like(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.match(r"^\d{2,7}-\d{2}-\d$", s))

def calc_k_from_logk(x):
    """将 Logk 转换为 k，失败时返回 NaN。"""
    try:
        if x is None:
            return np.nan
        s = str(x).strip()
        if s == "" or s.lower() in ["nan", "none"]:
            return np.nan
        v = float(s)
        if np.isnan(v):
            return np.nan
        return 10 ** v
    except Exception:
        return np.nan

def fmt_value(x, nd=6):
    if x is None:
        return "—"
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return "—"
    if s.lower() == "nan":
        return "nan"
    try:
        v = float(s)
        if np.isnan(v):
            return "nan"
        if abs(v) >= 1e6:
            return f"{v:.3e}"
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return s

def fmt_k_value(x):
    """k 用科学计数法显示更合适。"""
    try:
        if x is None:
            return "—"
        v = float(x)
        if np.isnan(v):
            return "—"
        return f"{v:.3e}"
    except Exception:
        return "—"

@st.cache_data(show_spinner=False)
def load_data():
    out = {}
    for system, path in RADICAL_FILES.items():
        df = pd.read_csv(path, encoding="utf-8-sig").copy()

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{system} 数据集缺少必要字段：{missing}")

        for col in ["Chemical compound", "Cas", "Smiles", "Ref", "Chemical_class_27"]:
            df[col] = safe_str_series(df[col])

        df["_name_norm"] = df["Chemical compound"].map(norm_text)
        df["_cas_norm"] = df["Cas"].map(norm_text).map(lambda x: x.replace(" ", ""))
        df["_rid"] = np.arange(1, len(df) + 1)

        # 新增 k 列
        df["k"] = pd.to_numeric(df["Logk"], errors="coerce").map(calc_k_from_logk)

        out[system] = df
    return out

def mol_from_smiles(smiles: str):
    s = str(smiles).strip()
    if s == "" or s.lower() in ["nan", "none", "unrecorded"]:
        return None
    return Chem.MolFromSmiles(s)

DESCRIPTOR_FUNCS = {
    "分子量 MolWt": lambda m: float(Descriptors.MolWt(m)),
    "辛醇/水分配系数 MolLogP": lambda m: float(Descriptors.MolLogP(m)),
    "拓扑极性表面积 TPSA": lambda m: float(rdMolDescriptors.CalcTPSA(m)),
    "氢键供体数 HBD": lambda m: float(Lipinski.NumHDonors(m)),
    "氢键受体数 HBA": lambda m: float(Lipinski.NumHAcceptors(m)),
    "可旋转键数 NumRotatableBonds": lambda m: float(Lipinski.NumRotatableBonds(m)),
    "环数 RingCount": lambda m: float(Lipinski.RingCount(m)),
    "芳香环数 NumAromaticRings": lambda m: float(Lipinski.NumAromaticRings(m)),
    "重原子数 HeavyAtomCount": lambda m: float(Lipinski.HeavyAtomCount(m)),
    "sp3碳比例 FractionCSP3": lambda m: float(Lipinski.FractionCSP3(m)),
    "摩尔折射率 MolMR": lambda m: float(Descriptors.MolMR(m)),
}
ALL_DESC_NAMES = list(DESCRIPTOR_FUNCS.keys())

def compute_descriptors(smiles: str, selected: list[str]) -> dict:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError("SMILES 无法解析，无法计算描述符。")
    return {k: float(DESCRIPTOR_FUNCS[k](mol)) for k in selected}

def compute_morgan_bits(smiles: str, n_bits: int = 1024, radius: int = 2) -> list[int]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError("SMILES 无法解析，无法计算 Morgan 指纹。")
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
    arr = np.zeros((int(n_bits),), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.tolist()

def compute_maccs_bits(smiles: str) -> list[int]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError("SMILES 无法解析，无法计算 MACCS 指纹。")
    bv = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((bv.GetNumBits(),), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.tolist()

def summarize_bits(bits: list[int]) -> dict:
    arr = np.array(bits, dtype=int)
    return {
        "length": int(arr.size),
        "on_bits": int(arr.sum()),
        "on_ratio": float(arr.mean()),
    }

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def inc_query():
    try:
        db_inc_metric("queries", 1)
    except Exception as e:
        st.warning(f"查询次数更新失败：{e}")

def inc_download():
    try:
        db_inc_metric("downloads", 1)
    except Exception as e:
        st.warning(f"下载次数更新失败：{e}")

def cache_key(system: str, rid: int) -> str:
    return f"{system}__{rid}"

def get_cache(system: str, rid: int) -> dict:
    return st.session_state["calc_cache"].get(cache_key(system, rid), {})

def set_cache(system: str, rid: int, field: str, value):
    k = cache_key(system, rid)
    st.session_state["calc_cache"].setdefault(k, {})
    st.session_state["calc_cache"][k][field] = value

def availability_summary(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {"n": 0, "class": "—", "ph_range": "—", "t_range": "—", "smiles_pct": "—", "ref_pct": "—"}

    n = len(df)
    cls = df["Chemical_class_27"].fillna("").astype(str).str.strip()
    cls = cls[cls != ""]
    class_top = cls.value_counts().index[0] if len(cls) else "—"

    phs = pd.to_numeric(df["Ph"], errors="coerce").dropna()
    ts = pd.to_numeric(df["T"], errors="coerce").dropna()
    ph_range = f"{phs.min():.2f} – {phs.max():.2f}" if len(phs) else "—"
    t_range = f"{ts.min():.1f} – {ts.max():.1f}" if len(ts) else "—"

    smiles_ok = df["Smiles"].fillna("").astype(str).str.strip()
    ref_ok = df["Ref"].fillna("").astype(str).str.strip()
    smiles_pct = f"{(smiles_ok != '').mean() * 100:.1f}%"
    ref_pct = f"{(ref_ok != '').mean() * 100:.1f}%"

    return {"n": n, "class": class_top, "ph_range": ph_range, "t_range": t_range, "smiles_pct": smiles_pct, "ref_pct": ref_pct}

# ===================== Load data =====================
data_map = load_data()

# ===================== Top bar =====================
st.markdown(
    """
<div class='topbar'>
  <div class='brand-badge'>RLAI</div>
  <h1>RadLogk-AI：自由基反应动力学数据库</h1>
</div>
""",
    unsafe_allow_html=True,
)

# ===================== 3-column layout =====================
col_left, col_mid, col_right = st.columns([1.05, 1.35, 1.60], gap="large")

# ===================== LEFT =====================
with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>查询</div>", unsafe_allow_html=True)

    system = st.selectbox(
        "自由基/氧化剂体系",
        list(RADICAL_FILES.keys()),
        index=list(RADICAL_FILES.keys()).index(st.session_state["last_system"])
        if st.session_state["last_system"] in RADICAL_FILES else 0
    )
    st.session_state["last_system"] = system

    q = st.text_input(
        "化学名称或 CAS 号",
        value=st.session_state.get("last_query", ""),
        placeholder="例如：acetaminophen 或 71-55-6"
    )

    b1, b2 = st.columns([1, 1])
    with b1:
        do_search = st.button("开始查询", type="primary")
    with b2:
        do_clear = st.button("清空")

    st.markdown(
        "<div class='small'>化学名称检索支持空格和连字符差异；CAS 号支持精确匹配和部分匹配。</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if do_clear:
        st.session_state["last_results"] = None
        st.session_state["last_query"] = ""
        st.rerun()

    if do_search:
        inc_query()
        st.session_state["last_query"] = q

        q0 = (q or "").strip()
        if q0 == "":
            st.session_state["last_results"] = None
        else:
            df_sys = data_map[system]
            q_norm = norm_text(q0)
            q_cas = norm_text(q0).replace(" ", "")

            idx_set = set()

            if is_cas_like(q0):
                mask = df_sys["_cas_norm"].str.contains(re.escape(q_cas), na=False)
                idx_set |= set(df_sys.loc[mask, "_rid"].tolist())
            else:
                mask1 = df_sys["_name_norm"].str.contains(re.escape(q_norm), na=False)
                idx_set |= set(df_sys.loc[mask1, "_rid"].tolist())

                q2 = re.sub(r"[\s\-]", "", q_norm)
                name2 = df_sys["_name_norm"].map(lambda x: re.sub(r"[\s\-]", "", x))
                mask1b = name2.str.contains(re.escape(q2), na=False)
                idx_set |= set(df_sys.loc[mask1b, "_rid"].tolist())

                mask2 = df_sys["_cas_norm"].str.contains(re.escape(q_cas), na=False)
                idx_set |= set(df_sys.loc[mask2, "_rid"].tolist())

            if len(idx_set) == 0:
                st.session_state["last_results"] = pd.DataFrame()
            else:
                res = df_sys[df_sys["_rid"].isin(sorted(idx_set))].copy()
                st.session_state["last_results"] = res

        st.rerun()

    dev_ui_enabled = qp_has_dev_flag()

    if dev_ui_enabled:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>开发者统计</div>", unsafe_allow_html=True)

        dev_key_input = st.text_input("开发者密钥", type="password", value="")

        if st.button("解锁开发者视图"):
            expected = get_dev_key()
            st.session_state["dev_unlocked"] = (expected != "" and dev_key_input == expected)

        if st.session_state.get("dev_unlocked", False):
            try:
                m = db_get_all_metrics()
                a, b, c = st.columns(3)
                a.metric("访问会话数", m.get("visits", 0))
                b.metric("查询次数", m.get("queries", 0))
                c.metric("下载次数", m.get("downloads", 0))
                st.markdown(
                    "<div class='small'>统计数据存储于外部 PostgreSQL 数据库中。访问会话数按每个会话计一次，并不等同于唯一访客人数。</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"统计数据加载失败：{e}")
        else:
            st.markdown("<div class='small'>当前处于锁定状态，普通用户不可查看统计信息。</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ===================== MID =====================
with col_mid:
    res = st.session_state.get("last_results", None)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>记录概览</div>", unsafe_allow_html=True)

    if res is None:
        st.markdown("<div class='small'>请输入化学名称或 CAS 号后点击“开始查询”。</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        if len(res) == 0:
            st.error("未找到匹配记录。请尝试缩短关键词，或检查化学名称 / CAS 号是否正确。")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.success(f"共匹配到 {len(res)} 条记录。")

            s = availability_summary(res)
            st.markdown(
                f"""
<span class="pill">记录数：<b>{s['n']}</b></span>
<span class="pill">主要类别：<b>{s['class']}</b></span>
<span class="pill">pH 范围：<b>{s['ph_range']}</b></span>
<span class="pill">温度范围（°C）：<b>{s['t_range']}</b></span>
<span class="pill">SMILES 完整率：<b>{s['smiles_pct']}</b></span>
<span class="pill">参考文献信息完整率：<b>{s['ref_pct']}</b></span>
""",
                unsafe_allow_html=True,
            )

            core_df = res[["_rid", "Chemical compound", "Cas", "Smiles", "Logk", "k",
                           "Chemical_class_27", "Ph", "T", "Ref"]].rename(
                columns={"_rid": "记录编号"}
            )
            st.download_button(
                "下载查询结果 CSV（核心字段）",
                data=to_csv_bytes(core_df),
                file_name=f"{st.session_state['last_system']}_查询结果.csv",
                mime="text/csv",
                on_click=inc_download,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>详细结果</div>", unsafe_allow_html=True)

            for i, (_, r) in enumerate(res.iterrows(), start=1):
                rid = int(r["_rid"])
                name = (r.get("Chemical compound", "") or "").strip()
                cas = (r.get("Cas", "") or "").strip()
                smiles = (r.get("Smiles", "") or "").strip()
                logk = fmt_value(r.get("Logk", ""), nd=9)
                kval = fmt_k_value(r.get("k", np.nan))
                cclass = (r.get("Chemical_class_27", "") or "").strip() or "—"
                ph = fmt_value(r.get("Ph", ""), nd=6)
                t = fmt_value(r.get("T", ""), nd=6)
                ref = (r.get("Ref", "") or "").strip()

                st.markdown(f"<div class='entry-head'>第 {i} 条 | 记录编号 #{rid}</div>", unsafe_allow_html=True)

                def _ref_html(x: str) -> str:
                    if x and x.startswith("http"):
                        return f"<a href='{x}' target='_blank'>{x}</a>"
                    return x if x else "—"

                st.markdown(
                    f"""
<div class="vbox"><div class="vtitle">化学物质名称</div><div class="vvalue">{name if name else "—"}</div></div>
<div class="vbox"><div class="vtitle">CAS 号</div><div class="vvalue">{cas if cas else "—"}</div></div>
<div class="vbox"><div class="vtitle">Logk</div><div class="vvalue">{logk}</div></div>
<div class="vbox"><div class="vtitle">k（未对数变换）</div><div class="vvalue">{kval}</div></div>
<div class="vbox"><div class="vtitle">化学类别</div><div class="vvalue">{cclass}</div></div>
<div class="vbox"><div class="vtitle">pH</div><div class="vvalue">{ph}</div></div>
<div class="vbox"><div class="vtitle">温度（°C）</div><div class="vvalue">{t}</div></div>
<div class="vbox"><div class="vtitle">参考文献</div><div class="vvalue">{_ref_html(ref)}</div></div>
<div class="vbox"><div class="vtitle">SMILES</div><div class="vvalue">{smiles if smiles else "—"}</div></div>
""",
                    unsafe_allow_html=True,
                )
                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ===================== RIGHT =====================
with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>在线计算</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>当前默认对第一条匹配记录进行计算，计算结果将在本次会话中缓存。</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    res = st.session_state.get("last_results", None)
    if res is None or (isinstance(res, pd.DataFrame) and len(res) == 0):
        st.markdown("<div class='card'><div class='small'>请先执行查询后再进行分子表征计算。</div></div>", unsafe_allow_html=True)
    else:
        r0 = res.iloc[0]
        rid = int(r0["_rid"])
        smiles = (r0.get("Smiles", "") or "").strip()
        name = (r0.get("Chemical compound", "") or "").strip()
        cas = (r0.get("Cas", "") or "").strip()
        logk_raw = r0.get("Logk", "")
        k_raw = r0.get("k", np.nan)
        system = st.session_state["last_system"]

        tabs = st.tabs(["二维描述符", "Morgan 指纹", "MACCS 指纹"])

        with tabs[0]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 二维分子描述符（11项）")

            desc_sel = st.multiselect(
                "选择描述符",
                options=ALL_DESC_NAMES,
                default=ALL_DESC_NAMES,
                key=f"desc_sel_{system}_{rid}",
            )

            if st.button("计算描述符", key=f"btn_desc_{system}_{rid}"):
                try:
                    d = compute_descriptors(smiles, desc_sel)
                    set_cache(system, rid, "desc", {"selected": desc_sel, "values": d})
                    st.success("描述符计算完成。")
                except Exception as e:
                    st.error(str(e))

            pack = get_cache(system, rid).get("desc", None)
            if pack is not None:
                dvals = pack["values"]

                st.markdown("<div class='desc-grid'>", unsafe_allow_html=True)
                for k in dvals.keys():
                    v = fmt_value(dvals[k], nd=6)
                    st.markdown(
                        f"<div class='desc-box'><div class='desc-name'>{k}</div><div class='desc-val'>{v}</div></div>",
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                out_df = pd.DataFrame([{
                    "记录编号": rid,
                    "体系": system,
                    "化学物质名称": name,
                    "CAS 号": cas,
                    "SMILES": smiles,
                    "Logk": logk_raw,
                    "k": k_raw,
                    **dvals
                }])

                st.download_button(
                    "下载描述符结果 CSV",
                    data=to_csv_bytes(out_df),
                    file_name=f"{system}_记录{rid}_描述符结果.csv",
                    mime="text/csv",
                    on_click=inc_download,
                    key=f"dl_desc_{system}_{rid}",
                )
            else:
                st.caption("点击“计算描述符”后将在此处显示结果。")
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Morgan 分子指纹")

            c1, c2 = st.columns(2)
            with c1:
                n_bits = st.number_input(
                    "指纹位数 nBits",
                    min_value=1,
                    max_value=8192,
                    value=32,
                    step=1,
                    key=f"m_bits_{system}_{rid}",
                )
            with c2:
                radius = st.number_input(
                    "半径 radius",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                    key=f"m_rad_{system}_{rid}",
                )

            if st.button("计算 Morgan 指纹", key=f"btn_morgan_{system}_{rid}"):
                try:
                    bits = compute_morgan_bits(smiles, int(n_bits), int(radius))
                    set_cache(system, rid, "morgan", {"nBits": int(n_bits), "radius": int(radius), "bits": bits})
                    st.success("Morgan 指纹计算完成。")
                except Exception as e:
                    st.error(str(e))

            mpack = get_cache(system, rid).get("morgan", None)
            if mpack is not None:
                bits = mpack["bits"]
                info = summarize_bits(bits)
                st.markdown(
                    f"<div class='small'><b>长度</b> = {info['length']} &nbsp;|&nbsp; <b>非零位数</b> = {info['on_bits']} &nbsp;|&nbsp; <b>非零位比例</b> = {info['on_ratio']:.4f}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(f"<div class='mono-wrap'>{','.join(map(str, bits))}</div>", unsafe_allow_html=True)

                nb = mpack["nBits"]
                rad = mpack["radius"]
                cols = [f"Morgan_{i}" for i in range(nb)]
                out_df = pd.DataFrame(
                    [[rid, system, name, cas, smiles, logk_raw, k_raw] + bits],
                    columns=["记录编号", "体系", "化学物质名称", "CAS 号", "SMILES", "Logk", "k"] + cols
                )

                st.download_button(
                    f"下载 Morgan 指纹 CSV（nBits={nb}, r={rad}）",
                    data=to_csv_bytes(out_df),
                    file_name=f"{system}_记录{rid}_Morgan_{nb}_r{rad}.csv",
                    mime="text/csv",
                    on_click=inc_download,
                    key=f"dl_morgan_{system}_{rid}",
                )
            else:
                st.caption("点击“计算 Morgan 指纹”后将在此处显示结果。")
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[2]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### MACCS 分子指纹（167位）")

            if st.button("计算 MACCS 指纹", key=f"btn_maccs_{system}_{rid}"):
                try:
                    bits = compute_maccs_bits(smiles)
                    set_cache(system, rid, "maccs", {"bits": bits})
                    st.success("MACCS 指纹计算完成。")
                except Exception as e:
                    st.error(str(e))

            kpack = get_cache(system, rid).get("maccs", None)
            if kpack is not None:
                bits = kpack["bits"]
                info = summarize_bits(bits)
                st.markdown(
                    f"<div class='small'><b>长度</b> = {info['length']} &nbsp;|&nbsp; <b>非零位数</b> = {info['on_bits']} &nbsp;|&nbsp; <b>非零位比例</b> = {info['on_ratio']:.4f}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(f"<div class='mono-wrap'>{','.join(map(str, bits))}</div>", unsafe_allow_html=True)

                cols = [f"MACCS_{i}" for i in range(len(bits))]
                out_df = pd.DataFrame(
                    [[rid, system, name, cas, smiles, logk_raw, k_raw] + bits],
                    columns=["记录编号", "体系", "化学物质名称", "CAS 号", "SMILES", "Logk", "k"] + cols
                )

                st.download_button(
                    "下载 MACCS 指纹 CSV",
                    data=to_csv_bytes(out_df),
                    file_name=f"{system}_记录{rid}_MACCS.csv",
                    mime="text/csv",
                    on_click=inc_download,
                    key=f"dl_maccs_{system}_{rid}",
                )
            else:
                st.caption("点击“计算 MACCS 指纹”后将在此处显示结果。")
            st.markdown("</div>", unsafe_allow_html=True)

st.caption("© 2026 RadLogk-AI 数据集团队")