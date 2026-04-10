import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Monte Carlo Gedung FITE", layout="wide")

# =========================
# MODEL
# =========================
class Stage:
    def __init__(self, o, m, p, deps=[]):
        self.o, self.m, self.p = o, m, p
        self.deps = deps

    def sample(self, n):
        return np.random.triangular(self.o, self.m, self.p, n)

class Simulation:
    def __init__(self, cfg, n):
        self.n, self.cfg = n, cfg

    def run(self):
        N = self.n
        dur = {k: Stage(**v).sample(N) for k,v in self.cfg.items()}
        df = pd.DataFrame(dur)

        start, end = {}, {}
        for k in self.cfg:
            deps = self.cfg[k]["deps"]
            start[k] = 0 if not deps else np.max([end[d] for d in deps], axis=0)
            end[k] = start[k] + df[k]

        df["Total"] = np.max(list(end.values()), axis=0)
        return df

# =========================
# CONFIG
# =========================
DEFAULT = {
    "Perizinan": {"o":1,"m":2,"p":4,"deps":[]},
    "Pondasi": {"o":1.5,"m":2,"p":3.5,"deps":["Perizinan"]},
    "Struktur": {"o":4,"m":6,"p":9,"deps":["Pondasi"]},
    "Fasad": {"o":1.5,"m":3,"p":5,"deps":["Struktur"]},
    "MEP": {"o":1.5,"m":2.5,"p":4,"deps":["Struktur"]},
    "Lab": {"o":1.5,"m":2.5,"p":5,"deps":["Fasad","MEP"]},
    "Finishing": {"o":0.5,"m":1,"p":2.5,"deps":["Lab"]},
}

# =========================
# UI SIDEBAR
# =========================
st.sidebar.title("⚙️ Pengaturan")
n = st.sidebar.slider("Simulasi",1000,20000,5000,1000)
seed = st.sidebar.number_input("Seed",0,9999,42)

cfg = {}
for k,v in DEFAULT.items():
    st.sidebar.markdown(f"**{k}**")
    o = st.sidebar.number_input(f"O {k}",0.5,20.0,float(v["o"]),key=k+"o")
    m = st.sidebar.number_input(f"M {k}",0.5,20.0,float(v["m"]),key=k+"m")
    p = st.sidebar.number_input(f"P {k}",0.5,30.0,float(v["p"]),key=k+"p")
    cfg[k] = {"o":o,"m":m,"p":p,"deps":v["deps"]}

run = st.sidebar.button("🚀 Jalankan")

# =========================
# OUTPUT
# =========================
st.title("🏗️ Simulasi Monte Carlo Gedung FITE")

if run:
    np.random.seed(seed)
    sim = Simulation(cfg,n)
    res = sim.run()

    total = res["Total"]

    mean = total.mean()
    median = np.median(total)
    std = total.std()

    c1,c2,c3 = st.columns(3)
    c1.metric("Mean",f"{mean:.1f} bln")
    c2.metric("Median",f"{median:.1f} bln")
    c3.metric("Std Dev",f"{std:.1f}")

    # Histogram
    fig = go.Figure()
    fig.add_histogram(x=total, nbinsx=50)
    fig.add_vline(x=mean,line_color="red")
    fig.add_vline(x=median,line_color="green")
    st.plotly_chart(fig,use_container_width=True)

    # Deadline
    st.subheader("Probabilitas Deadline")
    for d in [16,20,24]:
        p = np.mean(total <= d)
        st.write(f"{d} bulan → {p:.2%}")

    # Tabel
    st.subheader("Statistik")
    st.dataframe(res.describe())
