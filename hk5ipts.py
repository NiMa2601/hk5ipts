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
# FUNÇÃO DE INCLUSÃO
# =========================
def I(th, tps, ta, rs, ods):
    # Garantir que todos os parâmetros estão dentro do intervalo [0, 1]
    th = np.clip(th, 0.0, 1.0)
    tps = np.clip(tps, 0.0, 1.0)
    ta = np.clip(ta, 0.0, 1.0)
    rs = np.clip(rs, 0.0, 1.0)
    ods = np.clip(ods, 0.0, 1.0)
    
    return (
        0.3 * th +
        0.25 * tps +
        0.2 * ta +
        0.15 * rs +
        0.1 * ods
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

# Número de simulações para o Monte Carlo
num_simulations = st.sidebar.slider("Número de Simulações de Monte Carlo", 100, 5000, 1000)

# =========================
# TÍTULO PRINCIPAL
# =========================
st.title("HK5-IPTS – Sistema de Inclusão Educacional")

# =========================
# CÁLCULO DE I(t)
# =========================
index = I(TH, TPs, TA, RS, ODS)
st.metric("Índice de Inclusão I(t)", round(index, 3))

# =========================
# GRAFO VISUAL
# =========================
pos = nx.spring_layout(G, seed=42)

# Gerenciamento da renderização do gráfico
graph_placeholder = st.empty()

# Função para atualizar o gráfico com animação
def update_graph():
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
        line=dict(width=2),
        name="Arestas"
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=nodes,
        textposition="top center",
        marker=dict(size=15),
        name="Nós"
    ))

    # Animação simples no gráfico (redesenhando várias vezes)
    for _ in range(5):  # Simulando a animação
        graph_placeholder.plotly_chart(fig, use_container_width=True)
    
# Atualiza o gráfico
update_graph()

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
# IA (GAT CIENTÍFICO)
# =========================

st.header("🧠 IA – Sistema GAT Científico Avançado")
data = from_networkx(G)

data.x = torch.tensor([ 
    [TH, 1, 0], 
    [TPs, 1, 1], 
    [TA, 0, 1], 
    [0.4, 0, 0], 
    [0.5, 1, 0], 
    [RS, 0, 1], 
    [ODS, 1, 0] 
], dtype=torch.float)

node_names = nodes

# =========================
# MODELO GAT AVANÇADO
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

# =========================
# STATE SAFE
# =========================
if "model" not in st.session_state:
    st.session_state.model = GATHK5()

if "loss_history" not in st.session_state:
    st.session_state.loss_history = []

model = st.session_state.model
loss_history = st.session_state.loss_history

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.MSELoss()

# =========================
# TARGET ESTÁVEL
# =========================
target = torch.tensor([[index] for _ in range(len(nodes))], dtype=torch.float)

# =========================
# TREINAMENTO CONTROLADO
# =========================
train = st.button("🚀 Treinar GAT")

if train:
    model.train()
    progress_bar = st.progress(0)  # Barra de progresso
    
    for epoch in range(70):
        optimizer.zero_grad()
        out, emb = model(data.x, data.edge_index)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        
        # Atualizar a barra de progresso
        progress_bar.progress((epoch + 1) / 70)
        
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            st.write(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# =========================
# RESULTADOS FINAIS
# =========================
model.eval()
with torch.no_grad():
    out, emb = model(data.x, data.edge_index)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Previsão I(t)")
    st.write(torch.clamp(out, 0, 1).numpy())

with col2:
    st.subheader("📈 I(t) médio")
    st.metric("Índice médio", round(float(out.mean()), 3))
