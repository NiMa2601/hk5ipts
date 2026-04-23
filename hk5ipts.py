import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
import numpy as np
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

@st.cache_data
def create_graph():
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
        line=dict(width=2, color="rgba(125,125,125,0.2)"),
        name="Arestas"
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=nodes,
        textposition="top center",
        marker=dict(size=15, color="blue"),
        name="Nós"
    ))
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40)
    )
    
    return fig

st.plotly_chart(create_graph(), use_container_width=True, key="graph_main")

# =========================
# DOSSÊ (ARTIGO EMBUTIDO)
# =========================
st.header("Dossiê Científico Integrado")
st.markdown(""" 
## Modelo Matemático
I(t) = αTH + βTPs + γTA + δ5Rs + εODS

## Interpretação
- **TH** = tecnologia humana (centro ético)  
- **TPs** = tecnologias pedagógicas  
- **TA** = tecnologia assistiva  
- **5Rs** = sustentabilidade  
- **ODS** = diretrizes globais  

## Hipótese
A inclusão educacional emerge da interação entre tecnologia humana e tecnologias pedagógicas em rede dinâmica.

## Modelo de Rede
Sistema representado como grafo ponderado e dinâmico.
""")

# =========================
# SIMULAÇÃO IA
# =========================
st.header("Simulação Computacional")
st.write("O sistema simula relações entre nós da rede educacional e calcula impacto no índice de inclusão.")

# =========================
# IA (GAT CIENTÍFICO)
# =========================

st.header("🧠 IA – Sistema GAT Científico Avançado")

# Preparar dados
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

# =========================
# MODELO GAT
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
        out = torch.sigmoid(self.out(x))
        return out, x

# =========================
# INICIALIZAR ESTADO
# =========================
if "model" not in st.session_state:
    st.session_state.model = GATHK5()
    st.session_state.loss_history = []
    st.session_state.training_complete = False

model = st.session_state.model
target = torch.tensor([[index] for _ in range(len(nodes))], dtype=torch.float)

# =========================
# TREINAMENTO
# =========================
col_train, col_info = st.columns([1, 3])

with col_train:
    train = st.button("🚀 Treinar GAT", use_container_width=True)

with col_info:
    st.info(f"Status: {'✅ Treinamento ativo' if train else '⏸️ Aguardando...'}")

if train:
    st.session_state.training_complete = False
    st.session_state.loss_history = []
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart = st.empty()
    
    for epoch in range(70):
        optimizer.zero_grad()
        out, emb = model(data.x, data.edge_index)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        
        st.session_state.loss_history.append(loss.item())
        
        # Atualizar progresso
        progress = (epoch + 1) / 70
        progress_bar.progress(progress)
        
        # Atualizar status
        if epoch % 10 == 0:
            status_text.write(f"**Epoch {epoch}/70** | Loss: `{loss.item():.6f}`")
        
        # Atualizar gráfico a cada 5 epochs
        if epoch % 5 == 0 and len(st.session_state.loss_history) > 0:
            loss_df = pd.DataFrame({
                "Epoch": range(len(st.session_state.loss_history)),
                "Loss": st.session_state.loss_history
            })
            loss_chart.line_chart(loss_df.set_index("Epoch"), use_container_width=True)
    
    st.session_state.training_complete = True
    status_text.success("✅ Treinamento concluído!")

# =========================
# RESULTADOS
# =========================
st.divider()

model.eval()
with torch.no_grad():
    out, emb = model(data.x, data.edge_index)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Previsões por Nó")
    predictions = torch.clamp(out, 0, 1).detach().numpy()
    results_df = pd.DataFrame({
        "Nó": nodes,
        "Previsão": predictions.flatten().round(4)
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📈 Índice Médio")
    avg_index = float(out.mean())
    st.metric("I(t) Médio", round(avg_index, 4))
    
    # Comparação
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        st.metric("I(t) Calculado", round(index, 4), delta="baseline")
    with col_comp2:
        delta = round(avg_index - index, 4)
        st.metric("Diferença", delta, delta=f"{delta:+.4f}")

# =========================
# HISTÓRICO FINAL
# =========================
if st.session_state.loss_history:
    st.divider()
    st.subheader("📉 Histórico de Treinamento")
    
    loss_df = pd.DataFrame({
        "Epoch": range(len(st.session_state.loss_history)),
        "Loss": st.session_state.loss_history
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=loss_df["Epoch"],
        y=loss_df["Loss"],
        mode="lines",
        name="Loss",
        line=dict(color="red", width=2)
    ))
    fig.update_layout(
        title="Evolução do Loss",
        xaxis_title="Época",
        yaxis_title="Valor do Loss",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True, key="loss_chart_final")
