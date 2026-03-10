import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="EpiMap IKN Nusantara", layout="wide", page_icon="🌿")

# ==========================================
# DATA TOPOLOGI GRAF METRIK IKN (KIPP)
# ==========================================
# Koordinat disesuaikan ke area Kawasan Inti Pusat Pemerintahan (KIPP) IKN
nodes_data = {
    0: {"nama": "Istana & Kantor Kementerian", "lat": -0.9650, "lon": 116.7000, "pop": 10000},
    1: {"nama": "Perumahan ASN & Polri", "lat": -0.9500, "lon": 116.7150, "pop": 30000},
    2: {"nama": "Hunian Pekerja Konstruksi", "lat": -0.9800, "lon": 116.7100, "pop": 15000},
    3: {"nama": "Kawasan Komersial & Retail", "lat": -0.9600, "lon": 116.7300, "pop": 20000},
    4: {"nama": "RS Internasional IKN", "lat": -0.9750, "lon": 116.6900, "pop": 5000}
}
num_nodes = len(nodes_data)

# Sisi penghubung (Jaringan jalan arteri KIPP)
edges = [
    (0, 1), # Istana ke Perumahan ASN
    (0, 2), # Istana ke Hunian Pekerja
    (0, 3), # Istana ke Komersial
    (0, 4), # Istana ke RS Internasional
    (1, 3), # Perumahan ASN ke Komersial
    (2, 4)  # Hunian Pekerja ke RS Internasional
]
M = len(edges)
Nx = 40 

# Pre-komputasi koordinat koridor jalan (Edges)
lat_edges = np.zeros((M, Nx))
lon_edges = np.zeros((M, Nx))
for m, (u, v) in enumerate(edges):
    lat_edges[m, :] = np.linspace(nodes_data[u]["lat"], nodes_data[v]["lat"], Nx)
    lon_edges[m, :] = np.linspace(nodes_data[u]["lon"], nodes_data[v]["lon"], Nx)

# ==========================================
# ENGINE PDE KINERJA TINGGI (DENGAN CACHE)
# ==========================================
@st.cache_data(show_spinner=False)
def solve_pde_network(outbreak_idx, I_awal_pct, beta, gamma, alpha, p, v, D_mobilitas, T_hari):
    Lambda, mu = 0.0001, 0.0001
    Ds = Di = D_mobilitas 
    L, dx = 10.0, 10.0/Nx
    
    dt_safe = 0.4 * (dx**2) / max(Ds, 1e-6)
    dt = min(0.01, dt_safe) 
    Nt = int(T_hari / dt)

    node_S = np.full(num_nodes, 1.0)
    node_I = np.zeros(num_nodes)
    node_I[outbreak_idx] = float(I_awal_pct)
    node_S[outbreak_idx] -= float(I_awal_pct)
    
    edge_S, edge_I = np.zeros((M, Nx)), np.zeros((M, Nx))
    for m, (u, v_node) in enumerate(edges):
        edge_S[m, :] = np.linspace(node_S[u], node_S[v_node], Nx)
        edge_I[m, :] = np.linspace(node_I[u], node_I[v_node], Nx)
        
    daily_edge_I, daily_node_I, hist_S_agg, hist_I_agg = [], [], [], []
    hari_tercatat = 0

    for n in range(Nt + 1):
        edge_S_old, edge_I_old = edge_S.copy(), edge_I.copy()
        node_S_old, node_I_old = node_S.copy(), node_I.copy()
        
        if n * dt >= hari_tercatat:
            daily_edge_I.append(edge_I_old.copy()); daily_node_I.append(node_I_old.copy())
            hist_S_agg.append(np.mean(node_S_old)); hist_I_agg.append(np.mean(node_I_old))
            hari_tercatat += 1

        for m in range(M):
            for i in range(1, Nx - 1):
                rS = (1-p)*Lambda - beta*edge_S_old[m,i]*edge_I_old[m,i] - (mu+v)*edge_S_old[m,i]
                rI = beta*edge_S_old[m,i]*edge_I_old[m,i] - (gamma+mu+alpha)*edge_I_old[m,i]
                lap_S = (edge_S_old[m,i+1] - 2*edge_S_old[m,i] + edge_S_old[m,i-1]) / dx**2
                lap_I = (edge_I_old[m,i+1] - 2*edge_I_old[m,i] + edge_I_old[m,i-1]) / dx**2
                edge_S[m,i] = edge_S_old[m,i] + dt*(Ds*lap_S + rS)
                edge_I[m,i] = edge_I_old[m,i] + dt*(Di*lap_I + rI)
                
        for k in range(num_nodes):
            sum_S_flux, sum_I_flux, degree = 0.0, 0.0, 0
            for m, (u, v_node) in enumerate(edges):
                if u == k:   
                    sum_S_flux += edge_S_old[m, 1]; sum_I_flux += edge_I_old[m, 1]; degree += 1
                elif v_node == k: 
                    sum_S_flux += edge_S_old[m, -2]; sum_I_flux += edge_I_old[m, -2]; degree += 1
                    
            rS = (1-p)*Lambda - beta*node_S_old[k]*node_I_old[k] - (mu+v)*node_S_old[k]
            rI = beta*node_S_old[k]*node_I_old[k] - (gamma+mu+alpha)*node_I_old[k]
            node_S[k] = node_S_old[k] + dt*(Ds*(2.0/(degree*dx**2))*(sum_S_flux - degree*node_S_old[k]) + rS)
            node_I[k] = node_I_old[k] + dt*(Di*(2.0/(degree*dx**2))*(sum_I_flux - degree*node_I_old[k]) + rI)
            
        for m, (u, v_node) in enumerate(edges):
            edge_S[m, 0], edge_I[m, 0] = node_S[u], node_I[u]
            edge_S[m, -1], edge_I[m, -1] = node_S[v_node], node_I[v_node]

        edge_S = np.clip(np.nan_to_num(edge_S), 0.0, 1.0); edge_I = np.clip(np.nan_to_num(edge_I), 0.0, 1.0)
        node_S = np.clip(np.nan_to_num(node_S), 0.0, 1.0); node_I = np.clip(np.nan_to_num(node_I), 0.0, 1.0)

    return {
        "edge_history": daily_edge_I, "node_history": daily_node_I,
        "hist_S": hist_S_agg, "hist_I": hist_I_agg,
        "max_day": len(daily_edge_I) - 1,
        "max_inf": max(0.01, float(np.max(daily_node_I)))
    }

# ==========================================
# ANTARMUKA PENGGUNA (SIDEBAR)
# ==========================================
st.sidebar.title("🎛️ Panel Skenario IKN")

profil_db = {
    "Penyakit Campak (Measles)": {"beta": 1.2, "gamma": 0.1, "alpha": 0.01},
    "Flu Musiman / ISPA (Airborne)": {"beta": 0.4, "gamma": 0.15, "alpha": 0.005},
    "Demam Berdarah Dengue (Vektor)": {"beta": 0.25, "gamma": 0.08, "alpha": 0.02},
    "Mutasi Virus Baru (Disease X)": {"beta": 0.8, "gamma": 0.05, "alpha": 0.05}
}

penyakit = st.sidebar.selectbox("Pilih Karakteristik Epidemi:", list(profil_db.keys()))
default_params = profil_db[penyakit]

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Titik Awal Wabah")
nama_lokasi = [v["nama"] for k, v in nodes_data.items()]
outbreak_nama = st.sidebar.selectbox("Ground Zero (Titik Nol IKN):", nama_lokasi)
outbreak_idx = nama_lokasi.index(outbreak_nama)
I0_input = st.sidebar.slider("Letupan Infeksi Awal (%)", 0.01, 0.20, 0.05, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parameter Kinetik")
beta = st.sidebar.slider("Laju Transmisi (β)", 0.01, 2.0, default_params["beta"], step=0.05)
gamma = default_params["gamma"]
alpha = default_params["alpha"]

st.sidebar.subheader("🛡️ Intervensi Otorita IKN")
p_vaksin = st.sidebar.slider("Imunisasi Dasar / Vaksinasi Awal (p)", 0.0, 1.0, 0.6, step=0.05)
v_susulan = st.sidebar.slider("Laju Vaksinasi Susulan (v)", 0.0, 0.05, 0.001, step=0.001)

st.sidebar.markdown("---")
st.sidebar.subheader("🚗 Mobilitas Smart City")
diff_coeff = st.sidebar.slider("Koefisien Mobilitas (Autonomous Transport)", 0.0, 0.2, 0.08, step=0.01)
hari_simulasi = st.sidebar.slider("Durasi Simulasi (Hari)", 30, 365, 90)

R0_eff = (beta / (gamma + 0.0001 + alpha)) * ((1 - p_vaksin) * 0.0001 / (0.0001 + v_susulan))

st.sidebar.markdown("---")
if R0_eff > 1:
    st.sidebar.error(f"**Status Efektif (Rv): {R0_eff:.2f}**\nWabah Ekspansif / Pandemik")
else:
    st.sidebar.success(f"**Status Efektif (Rv): {R0_eff:.2f}**\nWabah Terkendali")

# ==========================================
# EKSEKUSI KOMPUTASI
# ==========================================
with st.spinner("Menghitung model PDE Reaksi-Difusi pada Graf IKN..."):
    res = solve_pde_network(outbreak_idx, I0_input, beta, gamma, alpha, p_vaksin, v_susulan, diff_coeff, hari_simulasi)

# ==========================================
# TATA LETAK UTAMA
# ==========================================
st.title("Sistem Pendukung Keputusan Epidemiologi Ibu Kota Nusantara (IKN)")

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Peta Graf Metrik KIPP", "📈 Kurva Agregat", "📋 Rekomendasi Kebijakan Otorita", "📖 Formulasi Matematis"])

with tab1:
    waktu_tinjau = st.slider("Tinjau Animasi Peta pada Hari ke-", 0, res["max_day"], res["max_day"])
    
    fig_map = go.Figure()
    
    node_lats = [nodes_data[k]["lat"] for k in range(num_nodes)]
    node_lons = [nodes_data[k]["lon"] for k in range(num_nodes)]
    node_I_now = res["node_history"][waktu_tinjau]
    edge_I_now = res["edge_history"][waktu_tinjau]
    
    # Trace 1: Rute Jalan (Edges)
    fig_map.add_trace(go.Scattermapbox(
        lat=lat_edges.flatten(), 
        lon=lon_edges.flatten(), 
        mode='markers',
        marker=dict(size=7, color=edge_I_now.flatten(), colorscale='YlOrRd', cmin=0, cmax=res["max_inf"], 
                    colorbar=dict(title="Infeksi<br>Koridor", x=1.0, y=0.5, len=0.75, bgcolor="rgba(255,255,255,0.7)")),
        text=[f"Densitas Mobilitas Infeksi: {val:.4f}" for val in edge_I_now.flatten()],
        hoverinfo="text",
        name="Koridor Otonom IKN"
    ))

    # Trace 2: Titik Fasilitas/Kawasan (Nodes) - Diatur TANPA 'line' pada marker
    fig_map.add_trace(go.Scattermapbox(
        lat=node_lats, 
        lon=node_lons, 
        mode='markers+text',
        marker=dict(size=30, color=node_I_now, colorscale='Reds', cmin=0, cmax=res["max_inf"]),
        text=nama_lokasi, 
        textposition="bottom center",
        textfont=dict(size=15, color='black', family="Arial Black"),
        hovertemplate="<b>%{text}</b><br>Densitas Infeksi Kawasan: %{marker.color:.4f}<extra></extra>",
        name="Kawasan Utama"
    ))
    
    fig_map.update_layout(
        mapbox_style="open-street-map", 
        mapbox=dict(
            center=dict(lat=-0.9650, lon=116.7100), # Pusat Koordinat Sepaku IKN
            zoom=12.5,
            uirevision="constant" 
        ),
        margin={"r":0,"t":0,"l":0,"b":0}, 
        height=650,
        showlegend=False
    )
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    st.subheader("Dinamika Populasi S-I-R Agregat IKN")
    t_arr = np.arange(res["max_day"] + 1)
    S_arr = np.array(res["hist_S"])
    I_arr = np.array(res["hist_I"])
    R_arr = 1.0 - S_arr - I_arr
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=t_arr, y=S_arr, name='Rentan (S)', line=dict(color='blue', width=3)))
    fig_ts.add_trace(go.Scatter(x=t_arr, y=I_arr, name='Terinfeksi (I)', line=dict(color='red', width=3)))
    fig_ts.add_trace(go.Scatter(x=t_arr, y=R_arr, name='Pulih/Kebal (R)', line=dict(color='green', width=3)))
    fig_ts.update_layout(xaxis_title="Waktu (Hari)", yaxis_title="Proporsi Populasi", height=450)
    st.plotly_chart(fig_ts, use_container_width=True)
    
    df_export = pd.DataFrame({'Hari': t_arr, 'Rentan_S': S_arr, 'Terinfeksi_I': I_arr, 'Kebal_R': R_arr})
    st.download_button(
        label="📥 Unduh Data Agregat (CSV)",
        data=df_export.to_csv(index=False).encode('utf-8'),
        file_name=f'simulasi_ikn_{penyakit.replace(" ", "_").lower()}.csv',
        mime='text/csv',
    )

with tab3:
    st.subheader("Rekomendasi Kebijakan Otorita IKN")
    
    node_I_final = res["node_history"][-1]
    id_max = np.argmax(node_I_final)
    kec_kritis = nodes_data[id_max]["nama"]
    
    st.markdown(f"**Prediksi Episentrum Kritis:** Berdasarkan komputasi matriks spasial, gelombang wabah tertinggi diprediksi menumpuk di kawasan **{kec_kritis}**.")

    if penyakit == "Penyakit Campak (Measles)":
        st.error(f"**Tindakan Darurat: Pengepungan Vaksinasi KIPP**")
        st.markdown(f"""
        Campak sangat menular melalui *airborne droplet*. Infrastruktur publik IKN yang terkoneksi erat bisa mempercepat penularan.
        * **Tata Ruang:** Batasi mobilitas ASN dan pekerja keluar-masuk dari **{kec_kritis}**.
        * **Medis:** Distribusi massal vaksin *Catch-up* di fasilitas kesehatan dekat Perumahan ASN dan Hunian Pekerja.
        """)
    elif penyakit == "Demam Berdarah Dengue (Vektor)":
        st.info(f"**Tindakan Darurat: Sanitasi Ekosistem & Genangan Konstruksi**")
        st.markdown(f"""
        Pada area yang sedang gencar dibangun, genangan air di lokasi konstruksi sangat berbahaya.
        * **Fokus Pembangunan:** Audit drainase di kawasan **{kec_kritis}**. Pembatasan mobilitas sistem transportasi pintar IKN **tidak perlu** dihentikan karena tidak mempengaruhi vektor nyamuk.
        * **Preventif:** *Fogging* darurat di sekitar area Hunian Pekerja Konstruksi.
        """)
    else:
        st.warning(f"**Tindakan Darurat: Mitigasi Sistemik**")
        st.markdown(f"""
        * **Pengendalian:** Batasi kapasitas *autonomous rapid transit* (ART) dan *smart transport* yang menghubungkan **{kec_kritis}** dengan wilayah lain.
        * **Fasilitas:** Aktifkan protokol *screening* suhu tubuh digital di lobi kantor kementerian dan kawasan komersial retail.
        """)

with tab4:
    st.markdown("### Pemodelan Reaksi-Difusi Spasial pada Graf Metrik IKN")
    st.markdown("""
    Sistem Pendukung Keputusan ini menggunakan metode Beda Hingga (*Finite Difference*) pada graf kontinu yang memodelkan tata ruang Ibu Kota Nusantara. Simpul (*nodes*) mewakili zona esensial di KIPP, dan sisi (*edges*) mewakili koridor mobilitas sistem transportasi cerdas IKN.
    
    **1. Persamaan Transportasi (Edges):**
    """)
    st.latex(r"\frac{\partial S}{\partial t} = D \frac{\partial^2 S}{\partial x^2} + (1-p)\Lambda - \beta S I - (\mu + v)S")
    st.latex(r"\frac{\partial I}{\partial t} = D \frac{\partial^2 I}{\partial x^2} + \beta S I - (\gamma + \mu + \alpha)I")
    st.latex(r"\frac{\partial R}{\partial t} = D \frac{\partial^2 R}{\partial x^2} + p\Lambda + vS + \gamma I - \mu R")
    
    st.markdown("""
    **2. Syarat Batas Aliran Mobilitas (Vertices):**
    Menerapkan Hukum Kirchhoff-Neumann di persimpangan zona:
    """)
    st.latex(r"\sum_{e_j \sim v_k} D \frac{\partial I_j}{\partial n}(v_k, t) = 0")
