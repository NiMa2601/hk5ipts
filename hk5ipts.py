import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

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
# SIDEBAR (CONTROLE)
# =========================
st.sidebar.title("HK5-IPTS Control Panel")

TH = st.sidebar.slider("Tecnologia Humana (TH)", 0.0, 1.0, 0.7)
TPs = st.sidebar.slider("Tecnologias Pedagógicas (TPs)", 0.0, 1.0, 0.6)
TA = st.sidebar.slider("Tecnologia Assistiva (TA)", 0.0, 1.0, 0.5)
RS = st.sidebar.slider("Sustentabilidade (5Rs)", 0.0, 1.0, 0.5)
ODS = st.sidebar.slider("ODS Globais", 0.0, 1.0, 0.8)

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

# =========================
# IA (GNN REAL) AVANÇADA
# =========================

st.header("IA (GNN real)")

# =========================
# CONVERTER GRAFO CORRETAMENTE
# =========================
data = from_networkx(G)

# features dos nós (simuladas)
data.x = torch.rand((G.number_of_nodes(), 3))  # Exemplo de 3 características por nó

# =========================
# MODELO GNN AVANÇADO
# =========================
class AdvancedGNN(torch.nn.Module):
    def __init__(self):
        super(AdvancedGNN, self).__init__()

        self.conv1 = GCNConv(3, 16)
        self.bn1 = torch.nn.BatchNorm1d(16)

        self.conv2 = GCNConv(16, 16)
        self.bn2 = torch.nn.BatchNorm1d(16)

        self.conv3 = GCNConv(16, 8)
        self.bn3 = torch.nn.BatchNorm1d(8)

        self.conv_out = GCNConv(8, 1)

        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x, edge_index):

        # Camada 1
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        # Camada 2 (residual)
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = x2 + x1  # RESIDUAL CONNECTION

        # Camada 3
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        # Saída
        out = self.conv_out(x3, edge_index)

        return out, x3  # retorna também embeddings

# =========================
# CONFIGURANDO OTIMIZADOR E PERDA
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

loss_history = []

# =========================
# TREINAMENTO COM MONITORAMENTO
# =========================
for epoch in range(100):  # Número de épocas de treinamento
    model.train()  # Modo de treinamento
    
    optimizer.zero_grad()  # Zera os gradientes
    
    # Passa os dados pela rede
    out, embeddings = model(data.x, data.edge_index)  
    
    # Calculando a perda
    target = torch.tensor([[index] for _ in range(G.number_of_nodes())], dtype=torch.float)  # Target (I(t) por nó)
    loss = loss_fn(out, target)  # Função de perda
    
    loss.backward()  # Retropropagação
    optimizer.step()  # Atualiza os pesos
    
    loss_history.append(loss.item())  # Armazena a perda
    
    if epoch % 10 == 0:  # Exibe a perda a cada 10 iterações
        st.write(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# =========================
# RESULTADOS FINAIS
# =========================
st.subheader("Resultados da GNN")

st.write("Previsão da IA (I(t) por nó):")
st.write(out.detach().numpy())  # Saída do modelo (I(t) calculado)

# =========================
# ANÁLISE DA CONVERGÊNCIA
# =========================
st.subheader("Análise da Convergência do Modelo")

st.line_chart(loss_history)  # Exibe a curva de convergência

# =========================
# EMBEDDINGS (INTERPRETAÇÃO DA REDE)
# =========================
st.subheader("Embeddings da Rede (Representação Latente)")

st.write(embeddings.detach().numpy())  # Exibe os embeddings da rede
