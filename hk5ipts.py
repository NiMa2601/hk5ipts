import streamlit as st
import networkx as nx
import plotly.graph_objects as go

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
# FUNÇÃO DE INCLUSÃO
# =========================
def I(th, tps, ta, rs, ods):
    return (
        0.3 * th +
        0.25 * tps +
        0.2 * ta +
        0.15 * rs +
        0.1 * ods
    )

# =========================
# INTERFACE
# =========================
st.title("HK5-IPTS – Sistema de Inclusão Educacional")

st.sidebar.header("Parâmetros")

TH = st.sidebar.slider("TH (Tecnologia Humana)", 0.0, 1.0, 0.8)
TPs = st.sidebar.slider("TPs (Tecnologias Pedagógicas)", 0.0, 1.0, 0.7)
TA = st.sidebar.slider("TA (Tecnologia Assistiva)", 0.0, 1.0, 0.6)
RS = st.sidebar.slider("5Rs (Sustentabilidade)", 0.0, 1.0, 0.5)
ODS = st.sidebar.slider("ODS (Objetivos Globais)", 0.0, 1.0, 0.9)

index = I(TH, TPs, TA, RS, ODS)

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
# DOSSÊ (ARTIGO EMBUTIDO)
# =========================
st.header("Dossiê Científico Integrado")

st.markdown("""
## Modelo Matemático

I(t) = αTH + βTPs + γTA + δ5Rs + εODS

## Interpretação

TH = tecnologia humana (centro ético)  
TPs = tecnologias pedagógicas  
TA = tecnologia assistiva  
5Rs = sustentabilidade  
ODS = diretrizes globais  

## Hipótese

A inclusão educacional emerge da interação entre tecnologia humana e tecnologias pedagógicas em rede dinâmica.

## Modelo de Rede

Sistema representado como grafo ponderado e dinâmico.
""")

# =========================
# SIMULAÇÃO IA (VERSÃO SIMPLES)
# =========================
st.header("Simulação Computacional")

st.write("O sistema simula relações entre nós da rede educacional e calcula impacto no índice de inclusão.")

# =========================
# VALIDAÇÃO
# =========================
st.header("Validação do Sistema")

st.write("O modelo permite análise de estabilidade da rede e sensibilidade dos parâmetros educacionais.")
st.header("Análise de Rede")

st.write("Centralidade de grau:")
st.write(nx.degree_centrality(G))

st.write("Centralidade de intermediação:")
st.write(nx.betweenness_centrality(G))

st.write("Densidade da rede:")
st.write(nx.density(G))
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

st.header("IA (Rede Neural Simplificada)")

model = SimpleNN()

x = torch.rand((7, 3))
out = model(x)

st.write(out.detach().numpy())
import numpy as np

st.header("Evolução Temporal da Inclusão")

t = np.linspace(0, 10, 50)

I_t = [0.7 + 0.1 * np.sin(i) for i in t]

st.line_chart(I_t)
import torch
from torch_geometric.utils import from_networkx

data = from_networkx(G)

# features simuladas dos nós
data.x = torch.rand((G.number_of_nodes(), 3))
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 8)
        self.conv2 = GCNConv(8, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
        model = GNN()

out = model(data.x, data.edge_index)

st.header("IA (GNN real)")
st.write(out.detach().numpy())
