"""
============================================================================
SIMULASI MONTE CARLO — ESTIMASI WAKTU PEMBANGUNAN GEDUNG FITE 5 LANTAI
Modul Praktikum 5 | Pemodelan dan Simulasi (MODSIM) 2026
Studi Kasus 2.1 — Ekuivalen Latihan 1.3 (Streamlit + Plotly)
============================================================================
Jalankan: streamlit run app_gedung_fite.py
============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Monte Carlo — Gedung FITE 5 Lantai",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0f2942;
    text-align: center;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}
.sub-title {
    text-align: center;
    color: #4a7ab5;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}
.info-box {
    background: linear-gradient(135deg, #e8f4fd 0%, #d1e8f5 100%);
    padding: 1rem 1.2rem;
    border-radius: 10px;
    border-left: 5px solid #2563eb;
    margin-bottom: 1rem;
    font-size: 0.92rem;
    line-height: 1.6;
}
.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    color: white;
    padding: 1.1rem;
    border-radius: 12px;
    text-align: center;
    margin: 0.3rem 0;
}
.metric-card h2 { margin: 0; font-size: 1.8rem; color: #fff; }
.metric-card p { margin: 0.2rem 0 0; font-size: 0.85rem; color: #cfe0f5; }
.warn-box {
    background: #fff8e1;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    border-left: 5px solid #f59e0b;
    margin-bottom: 0.8rem;
    font-size: 0.88rem;
}
.rec-box {
    background: #f0fdf4;
    padding: 1rem 1.2rem;
    border-radius: 10px;
    border-left: 5px solid #22c55e;
    margin-bottom: 0.8rem;
}
.section-label {
    font-size: 1.4rem;
    font-weight: 600;
    color: #0f2942;
    margin: 1.5rem 0 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #2563eb;
}
.stage-pill {
    display: inline-block;
    background: #e0ebff;
    color: #1d4ed8;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.15rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. KELAS PEMODELAN SISTEM
# ============================================================================

class TahapanKonstruksi:
    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic = base_params['optimistic']
        self.most_likely = base_params['most_likely']
        self.pessimistic = base_params['pessimistic']
        self.risk_factors = risk_factors or {}
        self.dependencies = dependencies or []

    def sample_duration(self, n_simulations, risk_multiplier=1.0):
        base_duration = np.random.triangular(
            self.optimistic, self.most_likely, self.pessimistic, n_simulations
        )
        for _, risk_params in self.risk_factors.items():
            if risk_params['type'] == 'discrete':
                risk_occurs = np.random.random(n_simulations) < risk_params['probability']
                base_duration = np.where(
                    risk_occurs, base_duration * (1 + risk_params['impact']), base_duration
                )
            elif risk_params['type'] == 'continuous':
                prod = np.random.normal(risk_params['mean'], risk_params['std'], n_simulations)
                base_duration = base_duration / np.clip(prod, 0.5, 1.5)
        return base_duration * risk_multiplier


class MonteCarloGedungSimulation:
    def __init__(self, stages_config, num_simulations=10000):
        self.stages_config = stages_config
        self.num_simulations = num_simulations
        self.stages = {}
        self.simulation_results = None
        self._init_stages()

    def _init_stages(self):
        for name, cfg in self.stages_config.items():
            self.stages[name] = TahapanKonstruksi(
                name=name,
                base_params=cfg['base_params'],
                risk_factors=cfg.get('risk_factors', {}),
                dependencies=cfg.get('dependencies', [])
            )

    def run_simulation(self):
        N = self.num_simulations
        results = pd.DataFrame(index=range(N))
        for sn, stage in self.stages.items():
            results[sn] = stage.sample_duration(N)

        start_t = pd.DataFrame(index=range(N))
        end_t = pd.DataFrame(index=range(N))
        for sn in self.stages.keys():
            deps = self.stages[sn].dependencies
            start_t[sn] = 0 if not deps else end_t[deps].max(axis=1)
            end_t[sn] = start_t[sn] + results[sn]

        results['Total_Duration'] = end_t.max(axis=1)
        for sn in self.stages.keys():
            results[f'{sn}_Finish'] = end_t[sn]
            results[f'{sn}_Start'] = start_t[sn]
        self.simulation_results = results
        return results

    def critical_path_prob(self):
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        total_dur = self.simulation_results['Total_Duration']
        cp = {}
        for sn in self.stages.keys():
            sf = self.simulation_results[f'{sn}_Finish']
            corr = self.simulation_results[sn].corr(total_dur)
            is_crit = (sf + 0.05) >= total_dur
            cp[sn] = {
                'prob_kritis': float(np.mean(is_crit)),
                'korelasi_durasi': float(corr),
                'rata_rata_bulan': float(self.simulation_results[sn].mean()),
                'std_bulan': float(self.simulation_results[sn].std()),
            }
        return pd.DataFrame(cp).T

    def risk_contribution(self):
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        total_var = self.simulation_results['Total_Duration'].var()
        rc = {}
        for sn in self.stages.keys():
            covar = self.simulation_results[sn].cov(self.simulation_results['Total_Duration'])
            rc[sn] = {
                'kontribusi_persen': float((covar / total_var) * 100),
                'varians': float(self.simulation_results[sn].var()),
                'std_bulan': float(self.simulation_results[sn].std()),
            }
        return pd.DataFrame(rc).T


# ============================================================================
# 3. KONFIGURASI DEFAULT TAHAPAN
# ============================================================================

DEFAULT_CONFIG = {
    "Persiapan_Perizinan": {
        "label": "🗂️ Persiapan & Perizinan",
        "base_params": {"optimistic": 1.0, "most_likely": 2.0, "pessimistic": 4.0},
        "risk_factors": {
            "hambatan_birokrasi": {"type": "discrete", "probability": 0.35, "impact": 0.40},
            "produktivitas_tim": {"type": "continuous", "mean": 1.0, "std": 0.15},
        }
    },
    "Pondasi_Struktur_Bawah": {
        "label": "⛏️ Pondasi & Struktur Bawah",
        "base_params": {"optimistic": 1.5, "most_likely": 2.0, "pessimistic": 3.5},
        "risk_factors": {
            "cuaca_buruk": {"type": "discrete", "probability": 0.30, "impact": 0.25},
            "kondisi_tanah_buruk": {"type": "discrete", "probability": 0.20, "impact": 0.45},
            "produktivitas_pekerja": {"type": "continuous", "mean": 1.0, "std": 0.20},
        },
        "dependencies": ["Persiapan_Perizinan"]
    },
    "Struktur_Bangunan_5_Lantai": {
        "label": "🏗️ Struktur Bangunan 5 Lantai",
        "base_params": {"optimistic": 4.0, "most_likely": 6.0, "pessimistic": 9.0},
        "risk_factors": {
            "cuaca_buruk": {"type": "discrete", "probability": 0.35, "impact": 0.20},
            "keterlambatan_material_struktural": {"type": "discrete", "probability": 0.25, "impact": 0.30},
            "produktivitas_pekerja": {"type": "continuous", "mean": 1.0, "std": 0.25},
        },
        "dependencies": ["Pondasi_Struktur_Bawah"]
    },
    "Arsitektur_Fasad": {
        "label": "🎨 Arsitektur & Fasad",
        "base_params": {"optimistic": 1.5, "most_likely": 3.0, "pessimistic": 5.0},
        "risk_factors": {
            "perubahan_desain": {"type": "discrete", "probability": 0.30, "impact": 0.35},
            "keterlambatan_material_fasad": {"type": "discrete", "probability": 0.20, "impact": 0.25},
            "produktivitas_pekerja": {"type": "continuous", "mean": 1.0, "std": 0.20},
        },
        "dependencies": ["Struktur_Bangunan_5_Lantai"]
    },
    "Instalasi_MEP": {
        "label": "⚡ Instalasi MEP",
        "base_params": {"optimistic": 1.5, "most_likely": 2.5, "pessimistic": 4.0},
        "risk_factors": {
            "keterlambatan_material_teknis": {"type": "discrete", "probability": 0.30, "impact": 0.35},
            "koordinasi_kontraktor": {"type": "discrete", "probability": 0.25, "impact": 0.20},
            "produktivitas_teknisi": {"type": "continuous", "mean": 1.0, "std": 0.20},
        },
        "dependencies": ["Struktur_Bangunan_5_Lantai"]
    },
    "Instalasi_Lab_Khusus": {
        "label": "🔬 Instalasi Laboratorium Khusus",
        "base_params": {"optimistic": 1.5, "most_likely": 2.5, "pessimistic": 5.0},
        "risk_factors": {
            "keterlambatan_peralatan_lab": {"type": "discrete", "probability": 0.40, "impact": 0.50},
            "perubahan_desain_lab": {"type": "discrete", "probability": 0.35, "impact": 0.40},
            "produktivitas_teknisi_lab": {"type": "continuous", "mean": 1.0, "std": 0.25},
        },
        "dependencies": ["Arsitektur_Fasad", "Instalasi_MEP"]
    },
    "Finishing_Serah_Terima": {
        "label": "✅ Finishing & Serah Terima",
        "base_params": {"optimistic": 0.5, "most_likely": 1.0, "pessimistic": 2.5},
        "risk_factors": {
            "punch_list": {"type": "discrete", "probability": 0.25, "impact": 0.50},
            "birokrasi_serah_terima": {"type": "discrete", "probability": 0.20, "impact": 0.30},
        },
        "dependencies": ["Instalasi_Lab_Khusus"]
    }
}

RESOURCE_CATALOG = {
    'Pekerja Khusus': {'cost_per_month': 8_000_000,  'productivity_gain': 0.20, 'emoji': '👷'},
    'Alat Berat':     {'cost_per_month': 25_000_000, 'productivity_gain': 0.30, 'emoji': '🚧'},
    'Insinyur Sipil': {'cost_per_month': 15_000_000, 'productivity_gain': 0.25, 'emoji': '👨‍💼'},
    'Insinyur MEP':   {'cost_per_month': 18_000_000, 'productivity_gain': 0.28, 'emoji': '🔌'},
    'Teknisi Lab':    {'cost_per_month': 12_000_000, 'productivity_gain': 0.35, 'emoji': '🧑‍🔬'},
}

STAGE_LABELS = {k: v['label'] for k, v in DEFAULT_CONFIG.items()}
DEADLINE_SCENARIOS = [16, 20, 24]


# ============================================================================
# 4. SIDEBAR — KONFIGURASI
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi Simulasi")

    num_simulations = st.slider(
        'Jumlah Iterasi Simulasi', 1000, 30000, 10000, 1000,
        help="Semakin banyak iterasi, semakin akurat (namun lebih lama)"
    )
    seed = st.number_input("Random Seed", 0, 9999, 42, help="Untuk hasil yang dapat direproduksi")

    st.markdown("---")
    st.markdown("### 📋 Durasi Tahapan (Bulan)")

    config = {}
    for stage_key, stage_cfg in DEFAULT_CONFIG.items():
        config[stage_key] = {
            'base_params': dict(stage_cfg['base_params']),
            'risk_factors': stage_cfg['risk_factors'],
            'label': stage_cfg['label']
        }
        if 'dependencies' in stage_cfg:
            config[stage_key]['dependencies'] = stage_cfg['dependencies']

        with st.expander(stage_cfg['label'], expanded=False):
            bp = stage_cfg['base_params']
            o = st.number_input("Optimistik", 0.5, 20.0, float(bp['optimistic']), 0.5, key=f"o_{stage_key}")
            m = st.number_input("Paling Mungkin", 0.5, 20.0, float(bp['most_likely']), 0.5, key=f"m_{stage_key}")
            p = st.number_input("Pesimistik", 0.5, 30.0, float(bp['pessimistic']), 0.5, key=f"p_{stage_key}")
            config[stage_key]['base_params'] = {'optimistic': o, 'most_likely': m, 'pessimistic': p}

    st.markdown("---")
    run_btn = st.button("🚀 Jalankan Simulasi", type="primary", use_container_width=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#666; margin-top:0.5rem;">
    <b>Tentang Satuan Waktu:</b><br>
    Semua durasi dalam <b>bulan</b>.<br><br>
    <b>Skenario Deadline:</b> 16 | 20 | 24 bulan<br><br>
    <b>Fasilitas Gedung FITE:</b><br>
    Lab Komputer, Lab Elektro, Lab Mobile,<br>
    Lab VR/AR, Lab Game, Ruang Dosen,<br>
    Toilet, Ruang Serbaguna
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# 5. HEADER APLIKASI
# ============================================================================

st.markdown('<h1 class="main-header">🏗️ Simulasi Monte Carlo</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Estimasi Waktu Pembangunan Gedung FITE 5 Lantai | MODSIM 2026</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<b>Latar Belakang:</b> Proyek pembangunan gedung FITE 5 lantai dengan fasilitas lengkap (ruang kelas,
laboratorium komputer, elektro, mobile, VR/AR, game, ruang dosen, toilet, dan ruang serbaguna).
Simulasi Monte Carlo dengan distribusi <i>triangular</i> (estimasi 3 titik) digunakan untuk memodelkan
ketidakpastian durasi tiap tahapan dan faktor risiko seperti cuaca buruk, keterlambatan material teknis khusus,
perubahan desain laboratorium, dan produktivitas pekerja.
<br><br>
<b>Pertanyaan Penelitian:</b> (1) Berapa lama total waktu yang dibutuhkan? &nbsp;|&nbsp;
(2) Berapa risiko keterlambatan? &nbsp;|&nbsp; (3) Mana critical path? &nbsp;|&nbsp;
(4) P(selesai) pada deadline 16/20/24 bulan? &nbsp;|&nbsp; (5) Dampak penambahan resource?
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 6. INISIALISASI SESSION STATE & RUN SIMULASI
# ============================================================================

if 'sim_results' not in st.session_state:
    st.session_state.sim_results = None
    st.session_state.simulator = None

if run_btn:
    np.random.seed(int(seed))
    with st.spinner('⏳ Menjalankan simulasi Monte Carlo...'):
        sim = MonteCarloGedungSimulation(
            stages_config=config,
            num_simulations=int(num_simulations)
        )
        results = sim.run_simulation()
        st.session_state.sim_results = results
        st.session_state.simulator = sim
    st.success(f"✅ Simulasi selesai! {num_simulations:,} iterasi berhasil dijalankan.")

# ============================================================================
# 7. TAMPILKAN HASIL
# ============================================================================

if st.session_state.sim_results is not None:
    results = st.session_state.sim_results
    sim = st.session_state.simulator

    total_dur = results['Total_Duration']
    mean_d = total_dur.mean()
    median_d = np.median(total_dur)
    std_d = total_dur.std()
    ci80 = np.percentile(total_dur, [10, 90])
    ci95 = np.percentile(total_dur, [2.5, 97.5])
    p16 = np.mean(total_dur <= 16)
    p20 = np.mean(total_dur <= 20)
    p24 = np.mean(total_dur <= 24)

    # ─── METRIK UTAMA ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label">📊 Statistik Utama — Q1. Total Waktu Proyek</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lab in [
        (c1, f"{mean_d:.1f} bln", "Rata-rata Durasi"),
        (c2, f"{median_d:.1f} bln", "Median Durasi"),
        (c3, f"{std_d:.1f} bln", "Standar Deviasi"),
        (c4, f"{ci80[0]:.1f}–{ci80[1]:.1f}", "80% CI (bln)"),
        (c5, f"{ci95[0]:.1f}–{ci95[1]:.1f}", "95% CI (bln)"),
    ]:
        col.markdown(f'<div class="metric-card"><h2>{val}</h2><p>{lab}</p></div>', unsafe_allow_html=True)

    st.markdown("")

    # ─── TAB VISUALISASI ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distribusi Durasi",
        "🎯 Probabilitas Deadline",
        "🔍 Critical Path",
        "⚠️ Analisis Risiko",
        "💼 Optimasi Resource"
    ])

    # ── TAB 1: Distribusi ─────────────────────────────────────────────────
    with tab1:
        st.markdown("### Q1. Distribusi Durasi Total Pembangunan Gedung FITE")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=total_dur, nbinsx=60,
            name='Distribusi Durasi',
            marker_color='steelblue', opacity=0.7,
            histnorm='probability density'
        ))
        fig.add_vline(x=mean_d, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"Mean: {mean_d:.1f} bln", annotation_position="top right")
        fig.add_vline(x=median_d, line_dash="dash", line_color="green", line_width=2,
                      annotation_text=f"Median: {median_d:.1f} bln")
        fig.add_vrect(x0=ci80[0], x1=ci80[1], fillcolor="yellow", opacity=0.2,
                      annotation_text="80% CI", line_width=0)
        fig.add_vrect(x0=ci95[0], x1=ci95[1], fillcolor="orange", opacity=0.08,
                      annotation_text="95% CI", line_width=0)
        for dl, clr, lbl in [(16, "purple", "16 bln"), (20, "darkorange", "20 bln"), (24, "brown", "24 bln")]:
            fig.add_vline(x=dl, line_dash="dot", line_color=clr, line_width=2,
                          annotation_text=lbl, annotation_position="top")
        fig.update_layout(
            xaxis_title="Durasi Total Proyek (Bulan)",
            yaxis_title="Densitas Probabilitas",
            height=480,
            showlegend=True,
            template="plotly_white",
            font=dict(family="IBM Plex Sans")
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Tabel Statistik Lengkap"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Statistik Deskriptif:**")
                stat_df = pd.DataFrame({
                    'Statistik': ['Rata-rata', 'Median', 'Standar Deviasi', 'Minimum', 'Maksimum',
                                  'Persentil 10%', 'Persentil 25%', 'Persentil 75%', 'Persentil 90%'],
                    'Nilai (Bulan)': [
                        round(mean_d, 2), round(median_d, 2), round(std_d, 2),
                        round(total_dur.min(), 2), round(total_dur.max(), 2),
                        round(np.percentile(total_dur, 10), 2), round(np.percentile(total_dur, 25), 2),
                        round(np.percentile(total_dur, 75), 2), round(np.percentile(total_dur, 90), 2),
                    ]
                })
                st.dataframe(stat_df, use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**Durasi Rata-rata per Tahapan:**")
                stage_means = {STAGE_LABELS[k]: round(results[k].mean(), 2)
                               for k in sim.stages.keys()}
                df_sm = pd.DataFrame(list(stage_means.items()), columns=['Tahapan', 'Rata-rata (Bulan)'])
                st.dataframe(df_sm, use_container_width=True, hide_index=True)

        # Boxplot per tahapan
        stage_names = list(sim.stages.keys())
        fig2 = go.Figure()
        palette = px.colors.qualitative.Set2
        for i, sn in enumerate(stage_names):
            fig2.add_trace(go.Box(
                y=results[sn], name=STAGE_LABELS[sn].replace(' ', '<br>', 1),
                boxmean='sd', marker_color=palette[i % len(palette)],
                boxpoints='outliers', jitter=0.2
            ))
        fig2.update_layout(
            title="Distribusi Durasi per Tahapan Konstruksi",
            yaxis_title="Durasi (Bulan)", height=430,
            showlegend=False, template="plotly_white",
            font=dict(family="IBM Plex Sans")
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 2: Probabilitas Deadline ──────────────────────────────────────
    with tab2:
        st.markdown("### Q4. Probabilitas Penyelesaian pada Deadline 16 / 20 / 24 Bulan")

        dl_range = np.arange(10, 40, 0.5)
        probs = [np.mean(total_dur <= d) for d in dl_range]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=dl_range, y=probs, mode='lines',
            line=dict(color='#1d4ed8', width=3),
            fill='tozeroy', fillcolor='rgba(147,197,253,0.25)',
            name='P(selesai ≤ deadline)'
        ))
        for yv, col, lbl in [(0.50, 'red', '50%'), (0.80, 'green', '80%'), (0.95, 'blue', '95%')]:
            fig3.add_hline(y=yv, line_dash="dash", line_color=col, opacity=0.7,
                           annotation_text=lbl, annotation_position="right")
        for dl, clr in [(16, 'purple'), (20, 'darkorange'), (24, 'brown')]:
            prob = np.mean(total_dur <= dl)
            fig3.add_trace(go.Scatter(
                x=[dl], y=[prob], mode='markers+text',
                marker=dict(size=14, color=clr, line=dict(width=2, color='black')),
                text=[f"<b>{dl} bln<br>{prob:.1%}</b>"],
                textposition='top center', showlegend=False,
                textfont=dict(size=11)
            ))
            fig3.add_vline(x=dl, line_dash="dot", line_color=clr, opacity=0.6)

        fig3.update_layout(
            xaxis_title="Deadline (Bulan)", yaxis_title="P(Selesai Tepat Waktu)",
            yaxis_range=[-0.03, 1.05], xaxis_range=[10, 38],
            height=490, template="plotly_white",
            font=dict(family="IBM Plex Sans"),
            legend=dict(yanchor="bottom", y=0.05, xanchor="left", x=0.05)
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**📅 Tabel Probabilitas per Deadline:**")
        dl_scenarios = list(range(14, 33, 2))
        dl_data = []
        for dl in dl_scenarios:
            p_on = np.mean(total_dur <= dl)
            risk_mos = max(0, np.percentile(total_dur, 95) - dl)
            highlight = "🔴" if dl in [16, 20, 24] else ""
            dl_data.append({
                'Deadline (Bulan)': f"{highlight} {dl}",
                'P(Selesai ≤ Deadline)': f"{p_on:.2%}",
                'P(Terlambat)': f"{1-p_on:.2%}",
                'Risiko Terlambat (bln, 95th)': f"{risk_mos:.1f}"
            })
        st.dataframe(pd.DataFrame(dl_data), use_container_width=True, hide_index=True)

        st.markdown("**Ringkasan Skenario Deadline:**")
        c1, c2, c3 = st.columns(3)
        for col, dl, clr in [(c1, 16, "🔴"), (c2, 20, "🟠"), (c3, 24, "🟡")]:
            p = np.mean(total_dur <= dl)
            risk = max(0, np.percentile(total_dur, 95) - dl)
            col.metric(
                label=f"{clr} Deadline {dl} Bulan",
                value=f"{p:.2%}",
                delta=f"Risiko terlambat: {risk:.1f} bln",
                delta_color="inverse"
            )

    # ── TAB 3: Critical Path ───────────────────────────────────────────────
    with tab3:
        st.markdown("### Q3. Tahapan Paling Kritis (Critical Path Analysis)")

        cp_df = sim.critical_path_prob()

        col_left, col_right = st.columns([1, 1])
        with col_left:
            cp_sorted = cp_df.sort_values('korelasi_durasi', ascending=True)
            colors_cp = ['#c0392b' if c > 0.7 else '#e67e22' if c > 0.4 else '#27ae60'
                         for c in cp_sorted['korelasi_durasi']]
            fig_cp = go.Figure()
            fig_cp.add_trace(go.Bar(
                y=[STAGE_LABELS.get(s, s) for s in cp_sorted.index],
                x=cp_sorted['korelasi_durasi'],
                orientation='h',
                marker_color=colors_cp,
                marker_line_color='#2c3e50', marker_line_width=1.2,
                text=[f"{v:.2f}" for v in cp_sorted['korelasi_durasi']],
                textposition='auto'
            ))
            fig_cp.add_vline(x=0.5, line_dash="dot", line_color="gray", opacity=0.7)
            fig_cp.add_vline(x=0.7, line_dash="dot", line_color="orange", opacity=0.8,
                             annotation_text="Kritis >0.70", annotation_position="top")
            fig_cp.update_layout(
                title="Korelasi Durasi Tahapan vs Total Proyek<br><sup>(Indikator Critical Path)</sup>",
                xaxis_title="Korelasi (Pearson)",
                xaxis_range=[0, 1.05], height=400,
                template="plotly_white",
                font=dict(family="IBM Plex Sans")
            )
            st.plotly_chart(fig_cp, use_container_width=True)

        with col_right:
            cp_sorted2 = cp_df.sort_values('prob_kritis', ascending=True)
            colors_cp2 = ['#c0392b' if p > 0.8 else '#e67e22' if p > 0.4 else '#27ae60'
                          for p in cp_sorted2['prob_kritis']]
            fig_cp2 = go.Figure()
            fig_cp2.add_trace(go.Bar(
                y=[STAGE_LABELS.get(s, s) for s in cp_sorted2.index],
                x=cp_sorted2['prob_kritis'],
                orientation='h',
                marker_color=colors_cp2,
                marker_line_color='#2c3e50', marker_line_width=1.2,
                text=[f"{v:.1%}" for v in cp_sorted2['prob_kritis']],
                textposition='auto'
            ))
            fig_cp2.update_layout(
                title="Probabilitas Berada di Critical Path<br><sup>(Finish Time ≥ Total Duration)</sup>",
                xaxis_title="Probabilitas",
                xaxis_range=[0, 1.15], height=400,
                template="plotly_white",
                font=dict(family="IBM Plex Sans")
            )
            st.plotly_chart(fig_cp2, use_container_width=True)

        with st.expander("📋 Tabel Detail Critical Path Analysis"):
            cp_show = cp_df.copy()
            cp_show.index = [STAGE_LABELS.get(s, s) for s in cp_show.index]
            cp_show.columns = ['P(Kritis)', 'Korelasi Durasi', 'Rata-rata (bln)', 'Std Dev (bln)']
            for col in ['P(Kritis)', 'Korelasi Durasi']:
                cp_show[col] = cp_show[col].apply(lambda x: f"{x:.3f}")
            for col in ['Rata-rata (bln)', 'Std Dev (bln)']:
                cp_show[col] = cp_show[col].apply(lambda x: f"{x:.2f}")
            st.dataframe(cp_show.sort_values('Korelasi Durasi', ascending=False),
                         use_container_width=True)

        st.markdown("""
        <div class="rec-box">
        <b>📌 Interpretasi Critical Path:</b><br>
        Tahapan dengan <b>korelasi tinggi (>0.7)</b> terhadap total durasi adalah penentu utama
        jadwal proyek. Keterlambatan pada tahapan kritis akan langsung berdampak pada keterlambatan
        keseluruhan proyek. Sebaliknya, percepatan tahapan non-kritis tidak berpengaruh signifikan.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 4: Analisis Risiko ─────────────────────────────────────────────
    with tab4:
        st.markdown("### Q2. Risiko Keterlambatan Akibat Faktor Ketidakpastian")

        rc_df = sim.risk_contribution()

        col1, col2 = st.columns(2)
        with col1:
            rc_sorted = rc_df.sort_values('kontribusi_persen', ascending=False)
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Bar(
                x=[STAGE_LABELS.get(s, s) for s in rc_sorted.index],
                y=rc_sorted['kontribusi_persen'],
                marker_color=px.colors.qualitative.Vivid[:len(rc_sorted)],
                marker_line_color='#2c3e50', marker_line_width=1.2,
                text=[f"{v:.1f}%" for v in rc_sorted['kontribusi_persen']],
                textposition='outside'
            ))
            fig_risk.update_layout(
                title="Kontribusi Setiap Tahapan terhadap<br>Variabilitas Total Durasi (%)",
                yaxis_title="Kontribusi (%)", height=400,
                xaxis_tickangle=-30,
                template="plotly_white",
                font=dict(family="IBM Plex Sans")
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        with col2:
            # Heatmap korelasi
            corr_mat = results[list(sim.stages.keys())].corr()
            labels = [STAGE_LABELS.get(s, s).split('&')[0].strip() for s in corr_mat.columns]
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_mat.values,
                x=labels, y=labels,
                colorscale='RdBu', zmid=0,
                text=np.round(corr_mat.values, 2),
                texttemplate='%{text}',
                textfont={"size": 9},
                hoverongaps=False,
                colorbar=dict(thickness=12)
            ))
            fig_corr.update_layout(
                title="Matriks Korelasi Antar Tahapan",
                height=400, template="plotly_white",
                font=dict(family="IBM Plex Sans"),
                xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9))
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("**🎲 Distribusi Risiko per Tahapan:**")
        with st.expander("Detail Faktor Risiko Tiap Tahapan"):
            risk_info = {
                "Persiapan_Perizinan": "Hambatan birokrasi (35%, +40%), Variasi produktivitas tim",
                "Pondasi_Struktur_Bawah": "Cuaca buruk (30%, +25%), Kondisi tanah buruk (20%, +45%), Produktivitas pekerja",
                "Struktur_Bangunan_5_Lantai": "Cuaca buruk (35%, +20%), Keterlambatan material struktural (25%, +30%), Produktivitas pekerja",
                "Arsitektur_Fasad": "Perubahan desain (30%, +35%), Keterlambatan material fasad (20%, +25%)",
                "Instalasi_MEP": "Keterlambatan material teknis (30%, +35%), Koordinasi kontraktor (25%, +20%)",
                "Instalasi_Lab_Khusus": "Keterlambatan peralatan lab VR/AR/Game (40%, +50%), Perubahan desain lab (35%, +40%)",
                "Finishing_Serah_Terima": "Punch list masalah (25%, +50%), Birokrasi serah terima (20%, +30%)"
            }
            for stage_key, risk_desc in risk_info.items():
                st.markdown(f"**{STAGE_LABELS[stage_key]}**")
                st.caption(f"↳ {risk_desc}")

        st.markdown("""
        <div class="warn-box">
        <b>⚠️ Tahapan dengan Risiko Tertinggi:</b><br>
        <b>Instalasi Lab Khusus</b> memiliki risiko tertinggi karena ketergantungan pada peralatan
        khusus (Lab VR/AR, Lab Game, Lab Mobile) yang sering mengalami keterlambatan pengiriman
        dan perubahan spesifikasi teknis. <b>Struktur Bangunan</b> merupakan kontributor variabilitas
        terbesar karena durasi yang panjang dan paparan cuaca.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 5: Optimasi Resource ───────────────────────────────────────────
    with tab5:
        st.markdown("### Q5. Pengaruh Penambahan Resource terhadap Percepatan Proyek")

        st.markdown("**Konfigurasikan skenario penambahan resource:**")

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            sel_stage = st.selectbox("Tahapan Target",
                                     list(sim.stages.keys()),
                                     format_func=lambda x: STAGE_LABELS.get(x, x))
        with col_r2:
            sel_resource = st.selectbox("Jenis Resource",
                                        list(RESOURCE_CATALOG.keys()))
        with col_r3:
            sel_qty = st.slider("Jumlah", 1, 10, 2)
        with col_r4:
            sel_dur = st.slider("Durasi Penambahan (bln)", 1, 12, 3)

        if st.button("⚡ Hitung Dampak Resource", type="secondary"):
            rp = RESOURCE_CATALOG[sel_resource]
            improvement_factor = 1 - (rp['productivity_gain'] * min(sel_qty / 3, 1))

            sc_res = results.copy()
            sc_res[sel_stage] = sc_res[sel_stage] * improvement_factor

            scenario_totals = []
            for idx in range(len(results)):
                stage_times = {}
                for curr in sim.stages.keys():
                    deps = sim.stages[curr].dependencies
                    st_time = 0 if not deps else max(stage_times.get(d, 0) for d in deps)
                    dur_val = sc_res.loc[idx, curr] if curr == sel_stage else results.loc[idx, curr]
                    stage_times[curr] = st_time + dur_val
                scenario_totals.append(max(stage_times.values()))

            sc_arr = np.array(scenario_totals)
            bl_mean = total_dur.mean()
            op_mean = sc_arr.mean()
            reduction = bl_mean - op_mean
            pct_imp = (reduction / bl_mean) * 100

            total_cost = rp['cost_per_month'] * sel_qty * sel_dur
            cost_saving = reduction * 500_000_000
            net_benefit = cost_saving - total_cost
            roi = (net_benefit / total_cost) * 100 if total_cost > 0 else 0

            st.markdown("---")
            st.markdown("**📊 Hasil Analisis Penambahan Resource:**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Baseline", f"{bl_mean:.1f} bln")
            m2.metric("Setelah Optimasi", f"{op_mean:.1f} bln", delta=f"-{reduction:.1f} bln")
            m3.metric("Peningkatan", f"{pct_imp:.1f}%")
            m4.metric("ROI", f"{roi:.0f}%",
                       delta="positif" if roi > 0 else "negatif",
                       delta_color="normal" if roi > 0 else "inverse")

            st.markdown(f"""
            <div class="rec-box">
            💰 <b>Biaya Resource Tambahan:</b> Rp {total_cost:,.0f}<br>
            💵 <b>Estimasi Penghematan Biaya Proyek:</b> Rp {cost_saving:,.0f}<br>
            📈 <b>Net Benefit:</b> Rp {net_benefit:,.0f}<br>
            🎯 <b>ROI:</b> {roi:.1f}%
            </div>
            """, unsafe_allow_html=True)

            # Perbandingan distribusi
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Histogram(
                x=total_dur, nbinsx=50, name='Baseline', opacity=0.65,
                marker_color='steelblue', histnorm='probability density'
            ))
            fig_cmp.add_trace(go.Histogram(
                x=sc_arr, nbinsx=50, name='Setelah Optimasi', opacity=0.65,
                marker_color='#22c55e', histnorm='probability density'
            ))
            for dl, clr in [(16, 'purple'), (20, 'darkorange'), (24, 'brown')]:
                bp = np.mean(total_dur <= dl)
                op = np.mean(sc_arr <= dl)
                fig_cmp.add_vline(x=dl, line_dash="dot", line_color=clr, line_width=1.5,
                                  annotation_text=f"{dl}bln: {bp:.0%}→{op:.0%}")
            fig_cmp.update_layout(
                title=f"Distribusi Durasi: Baseline vs Setelah Tambah {sel_qty} {sel_resource}",
                barmode='overlay',
                xaxis_title="Durasi Total (Bulan)", yaxis_title="Densitas",
                height=400, template="plotly_white",
                font=dict(family="IBM Plex Sans")
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            st.markdown("**Dampak terhadap Deadline 16 / 20 / 24 Bulan:**")
            for dl in [16, 20, 24]:
                bp2 = np.mean(total_dur <= dl)
                op2 = np.mean(sc_arr <= dl)
                c1_, c2_, c3_ = st.columns([1, 2, 1])
                c1_.write(f"**Deadline {dl} bln**")
                c2_.progress(min(op2, 1.0), text=f"Baseline: {bp2:.1%} → Optimasi: {op2:.1%}")
                c3_.write(f"+{(op2-bp2):.1%}")

        st.markdown("---")
        st.markdown("**📊 Perbandingan Semua Skenario Resource (Preset):**")

        preset_scenarios = [
            {'stage': 'Struktur_Bangunan_5_Lantai', 'resource': 'Alat Berat', 'qty': 2, 'dur': 5},
            {'stage': 'Struktur_Bangunan_5_Lantai', 'resource': 'Pekerja Khusus', 'qty': 10, 'dur': 6},
            {'stage': 'Instalasi_Lab_Khusus', 'resource': 'Teknisi Lab', 'qty': 3, 'dur': 3},
            {'stage': 'Instalasi_MEP', 'resource': 'Insinyur MEP', 'qty': 2, 'dur': 3},
            {'stage': 'Arsitektur_Fasad', 'resource': 'Pekerja Khusus', 'qty': 5, 'dur': 3},
            {'stage': 'Pondasi_Struktur_Bawah', 'resource': 'Insinyur Sipil', 'qty': 1, 'dur': 2},
        ]

        preset_rows = []
        for ps in preset_scenarios:
            rp = RESOURCE_CATALOG[ps['resource']]
            imp_fac = 1 - (rp['productivity_gain'] * min(ps['qty'] / 3, 1))
            sc_res2 = results.copy()
            sc_res2[ps['stage']] = sc_res2[ps['stage']] * imp_fac

            sc_totals2 = []
            for idx in range(len(results)):
                stage_times2 = {}
                for curr in sim.stages.keys():
                    deps = sim.stages[curr].dependencies
                    st2 = 0 if not deps else max(stage_times2.get(d, 0) for d in deps)
                    d2 = sc_res2.loc[idx, curr] if curr == ps['stage'] else results.loc[idx, curr]
                    stage_times2[curr] = st2 + d2
                sc_totals2.append(max(stage_times2.values()))

            sc_arr2 = np.array(sc_totals2)
            bl_m = total_dur.mean()
            op_m = sc_arr2.mean()
            reduction2 = bl_m - op_m
            cost2 = rp['cost_per_month'] * ps['qty'] * ps['dur']
            saving2 = reduction2 * 500_000_000
            roi2 = ((saving2 - cost2) / cost2) * 100 if cost2 > 0 else 0

            preset_rows.append({
                'Tahapan': STAGE_LABELS[ps['stage']],
                'Resource': f"{rp['emoji']} {ps['resource']}",
                'Qty': ps['qty'],
                'Dur (bln)': ps['dur'],
                'Pengurangan (bln)': round(reduction2, 2),
                'Peningkatan (%)': round((reduction2 / bl_m) * 100, 1),
                'Biaya (jt Rp)': round(cost2 / 1_000_000, 0),
                'ROI (%)': round(roi2, 1),
            })

        preset_df = pd.DataFrame(preset_rows).sort_values('Pengurangan (bln)', ascending=False)
        st.dataframe(preset_df, use_container_width=True, hide_index=True)

        # Chart perbandingan
        fig_preset = go.Figure()
        fig_preset.add_trace(go.Bar(
            y=preset_df['Tahapan'] + '<br>' + preset_df['Resource'],
            x=preset_df['Pengurangan (bln)'],
            orientation='h', name='Pengurangan Durasi',
            marker_color='#22c55e', marker_line_color='#166534', marker_line_width=1,
            text=[f"{v:.2f} bln" for v in preset_df['Pengurangan (bln)']],
            textposition='auto'
        ))
        fig_preset.update_layout(
            title="Perbandingan Dampak Skenario Resource terhadap Pengurangan Durasi",
            xaxis_title="Pengurangan Durasi (Bulan)",
            height=380, template="plotly_white",
            font=dict(family="IBM Plex Sans")
        )
        st.plotly_chart(fig_preset, use_container_width=True)

    # ─── REKOMENDASI AKHIR ──────────────────────────────────────────────────
    st.markdown('<p class="section-label">🏁 Kesimpulan & Rekomendasi</p>', unsafe_allow_html=True)

    sb = np.percentile(total_dur, 80) - mean_d
    cr = np.percentile(total_dur, 95) - mean_d

    col_sum1, col_sum2 = st.columns(2)
    with col_sum1:
        st.markdown(f"""
        <div class="rec-box">
        <b>🎯 Estimasi Jadwal Pembangunan Gedung FITE:</b><br><br>
        • <b>Durasi rata-rata:</b> {mean_d:.1f} bulan<br>
        • <b>Range realistis (80% CI):</b> {ci80[0]:.1f} – {ci80[1]:.1f} bulan<br>
        • <b>Safety buffer yang direkomendasikan:</b> +{sb:.1f} bulan<br>
        • <b>Estimasi jadwal dengan buffer:</b> <span style="font-size:1.1rem; font-weight:700;">
          {mean_d + sb:.1f} bulan (~{(mean_d+sb)/12:.1f} tahun)</span>
        </div>
        """, unsafe_allow_html=True)

    with col_sum2:
        st.markdown(f"""
        <div class="warn-box">
        <b>📊 Probabilitas per Skenario Deadline:</b><br><br>
        • Deadline <b>16 bulan:</b> P = <b>{p16:.2%}</b> — Sangat tidak realistis ⚠️<br>
        • Deadline <b>20 bulan:</b> P = <b>{p20:.2%}</b> — Berisiko tinggi 🟠<br>
        • Deadline <b>24 bulan:</b> P = <b>{p24:.2%}</b> — Perlu manajemen risiko 🟡<br><br>
        Contingency reserve (95% confidence): +{cr:.1f} bulan dari rata-rata
        </div>
        """, unsafe_allow_html=True)

else:
    # Belum ada simulasi
    st.markdown("""
    <div style="text-align:center; padding:3rem 2rem; background:#f8fafc;
                border-radius:16px; border: 2px dashed #cbd5e1; margin-top:1rem;">
        <h3 style="color:#1e3a5f;">🏗️ Siap untuk Memulai Simulasi?</h3>
        <p style="color:#475569;">Atur parameter di sidebar kiri, lalu klik tombol <b>🚀 Jalankan Simulasi</b>.</p>
        <p style="color:#64748b; font-size:0.9rem;">Hasil simulasi akan menjawab 5 pertanyaan penelitian
        seputar estimasi waktu pembangunan Gedung FITE 5 Lantai.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📋 Preview Konfigurasi Tahapan Konstruksi:**")
    cols_prev = st.columns(len(DEFAULT_CONFIG))
    for i, (sk, sv) in enumerate(DEFAULT_CONFIG.items()):
        bp = sv['base_params']
        cols_prev[i].markdown(f"""
        <div style="background:#f1f5f9; padding:0.7rem; border-radius:8px;
                    border-top:4px solid #2563eb; font-size:0.8rem; text-align:center;">
        <b>{sv['label']}</b><br><br>
        O: {bp['optimistic']} bln<br>
        M: {bp['most_likely']} bln<br>
        P: {bp['pessimistic']} bln
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.82rem; padding:0.5rem 0;">
<b>Monte Carlo Simulation — Estimasi Waktu Pembangunan Gedung FITE 5 Lantai</b> |
Modul Praktikum 5, MODSIM 2026 | Studi Kasus 2.1<br>
⚠️ Hasil simulasi merupakan estimasi probabilistik, bukan prediksi deterministik.
</div>
""", unsafe_allow_html=True)