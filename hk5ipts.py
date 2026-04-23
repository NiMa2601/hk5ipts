import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="HK5-IPTS", layout="wide")

# =========================
# REDE EDUCACIONAL
# =========================
G = nx.Graph()

nodes = ["TH", "TPs", "TA", "DU", "DUA", "5Rs", "ODS"]
G.add_nodes_from(nodes)

edges = [
    ("TH", "TPs"),
    ("TPs", "TA"),
    ("TA", "5Rs"),
    ("DU", "TPs"),
    ("DUA", "TA"),
    ("ODS", "TH"),
    ("ODS", "TPs"),
    ("ODS", "TA"),
]

G.add_edges_from(edges)

# =========================
# FUNÇÃO DE INCLUSÃO COM CAUSALIDADE
# =========================
def I(th, tps, ta, rs, ods, bullying=0.0, nutricion=0.0, violencia_domestica=0.0, exclusao_digital=0.0):
    # Causalidade com variáveis adicionais
    return (
        0.3 * th +
        0.25 * tps +
        0.2 * ta +
        0.15 * rs +
        0.1 * ods +
        bullying * 0.1 +      # Impacto do bullying
        nutricion * 0.05 +    # Impacto da nutrição
        violencia_domestica * 0.1 +  # Impacto da violência doméstica
        exclusao_digital * 0.2  # Impacto da exclusão digital
    )

# =========================
# SIDEBAR (CONTROLE)
# =========================
st.sidebar.title("HK5-IPTS Control Panel")

TH = st.sidebar.slider("Tecnologia Humana (TH)", 0.0, 1.0, 0.7)
TPs = st.sidebar.slider("Tecnologias Pedagógicas (TPs)", 0.0, 1.0, 0.6)
TA = st.sidebar.slider("Tecnologia Assistiva (TA)", 0.0, 1.0, 0.5)
RS = st.sidebar.slider("Sustentabilidade (5Rs)", 0.0, 1.0, 0.5)
ODS = st.sidebar.slider("ODS Globais", 0.0, 1.0, 0.8)
# Variáveis causais
bullying = st.sidebar.slider("Impacto do Bullying", 0.0, 1.0, 0.2)
nutricion = st.sidebar.slider("Impacto da Nutrição", 0.0, 1.0, 0.3)
violencia_domestica = st.sidebar.slider("Impacto da Violência Doméstica", 0.0, 1.0, 0.1)
exclusao_digital = st.sidebar.slider("Impacto da Exclusão Digital", 0.0, 1.0, 0.2)

# =========================
# TÍTULO PRINCIPAL
# =========================
st.title("HK5-IPTS – Sistema de Inclusão Educacional")

# =========================
# CÁLCULO DE I(t)
# =========================
index = I(TH, TPs, TA, RS, ODS, bullying, nutricion, violencia_domestica, exclusao_digital)
st.metric("Índice de Inclusão I(t)", round(index, 3))

# =========================
# GRAFO VISUAL
# =========================
pos = nx.spring_layout(G, seed=42)

edge_x, edge_y = [], []
for e in G.edges():
    x0, y0 = pos[e[0]]
    x1, y1 = pos[e[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

node_x, node_y = [], []
for n in G.nodes():
    x, y = pos[n]
    node_x.append(x)
    node_y.append(y)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode="lines",
    line=dict(width=2)
))
fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=nodes,
    textposition="top center",
    marker=dict(size=15)
))

st.plotly_chart(fig, use_container_width=True)

# =========================
# SIMULAÇÃO POLÍTICAS COM MONTE CARLO
# =========================
def monte_carlo_simulation(num_simulations=1000):
    simulation_results = []
    for _ in range(num_simulations):
        simulated_th = np.random.uniform(0, 1)
        simulated_tps = np.random.uniform(0, 1)
        simulated_ta = np.random.uniform(0, 1)
        simulated_rs = np.random.uniform(0, 1)
        simulated_ods = np.random.uniform(0, 1)
        simulated_bullying = np.random.uniform(0, 1)
        simulated_nutricion = np.random.uniform(0, 1)
        simulated_violencia_domestica = np.random.uniform(0, 1)
        simulated_exclusao_digital = np.random.uniform(0, 1)
        
        simulated_inclusion = I(simulated_th, simulated_tps, simulated_ta, simulated_rs, simulated_ods,
                                simulated_bullying, simulated_nutricion, simulated_violencia_domestica,
                                simulated_exclusao_digital)
        simulation_results.append(simulated_inclusion)
        
    return simulation_results

simulation_results = monte_carlo_simulation()

st.subheader("🔮 Simulação de Políticas Educacionais com Monte Carlo")
st.write(f"Média do índice de inclusão simulada: {np.mean(simulation_results):.2f}")
st.line_chart(simulation_results)

# =========================
# EXPORTAÇÃO PARA POWERBI
# =========================
st.subheader("📥 Exportação de Dados")

csv_data = pd.DataFrame({
    "TH": [TH],
    "TPs": [TPs],
    "TA": [TA],
    "RS": [RS],
    "ODS": [ODS],
    "Bullying": [bullying],
    "Nutrição": [nutricion],
    "Violência Doméstica": [violencia_domestica],
    "Exclusão Digital": [exclusao_digital],
    "Índice de Inclusão": [index],
})

st.download_button(
    label="Baixar CSV de Dados",
    data=csv_data.to_csv(index=False),
    file_name="dados_inclusao_educacional.csv",
    mime="text/csv"
)

# =========================
# GNN AVANÇADA + EMBEDDINGS
# =========================
class GATHK5(nn.Module):
    def __init__(self):
        super().__init__()

        self.gat1 = GATConv(3, 16, heads=2)
        self.gat2 = GATConv(32, 16, heads=2)
        self.gat3 = GATConv(32, 16, heads=1)

        self.out = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)

        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)

        x = F.elu(self.gat3(x, edge_index))

        embeddings = x
        out = torch.sigmoid(self.out(x))

        return out, embeddings
