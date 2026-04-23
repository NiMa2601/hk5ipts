import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

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
# MODELO GAT AVANÇADO
# =========================
class GATHK5(nn.Module):
    def __init__(self):
        super().__init__()

        self.gat1 = GATConv(3, 16, heads=2, concat=True)
        self.gat2 = GATConv(32, 16, heads=2, concat=True)
        self.gat3 = GATConv(32, 8, heads=1, concat=False)

        self.out = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, edge_index):

        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)

        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)

        x = F.elu(self.gat3(x, edge_index))

        embeddings = x
        out = torch.sigmoid(self.out(x))  # Estabilidade com sigmoid

        return out, embeddings

# =========================
# CONFIGURANDO OTIMIZADOR E PERDA
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
# TARGET CIENTÍFICO
# =========================
target = torch.tensor([[index] for _ in range(len(nodes))], dtype=torch.float)

# =========================
# SIMULAÇÃO DE POLÍTICAS EDUCAÇÃO REAIS (LBI, PNTA, BNCC, LDBs)
# =========================
st.header("🌍 Simulação de Políticas Educacionais")

# Definição de cenários de políticas
policies = {
    "Aumento de Tecnologia Humana (TH)": {"TH": 0.9, "TPs": 0.6, "TA": 0.5, "5Rs": 0.5, "ODS": 0.8},
    "Aumento de Tecnologias Pedagógicas (TPs)": {"TH": 0.7, "TPs": 1.0, "TA": 0.5, "5Rs": 0.5, "ODS": 0.8},
    "Foco em Sustentabilidade (5Rs)": {"TH": 0.7, "TPs": 0.6, "TA": 0.5, "5Rs": 1.0, "ODS": 0.8},
    "Aumento de ODS Globais": {"TH": 0.7, "TPs": 0.6, "TA": 0.5, "5Rs": 0.5, "ODS": 1.0},
    "Cenário Base (Sem alteração)": {"TH": 0.7, "TPs": 0.6, "TA": 0.5, "5Rs": 0.5, "ODS": 0.8},
}

selected_policy = st.selectbox("Escolha a Política Educacional", list(policies.keys()))

policy_values = policies[selected_policy]
st.write(f"Política Selecionada: {selected_policy}")
st.write(f"Configuração da Política: {policy_values}")

# Calcular o índice de inclusão para a política selecionada
index_policy = I(policy_values["TH"], policy_values["TPs"], policy_values["TA"], policy_values["5Rs"], policy_values["ODS"])
st.metric("Índice de Inclusão I(t) para esta Política", round(index_policy, 3))

# =========================
# VISUALIZAÇÃO POLÍTICA
# =========================
policy_df = pd.DataFrame(list(policy_values.items()), columns=["Variável", "Valor"])
fig_policy = px.bar(policy_df, x="Variável", y="Valor", title="Impacto das Variáveis na Política Educacional")
st.plotly_chart(fig_policy, use_container_width=True)

# =========================
# TREINAMENTO E MONITORAMENTO
# =========================
train = st.button("🚀 Treinar Modelo GAT")

if train:

    model.train()

    for epoch in range(80):

        optimizer.zero_grad()

        out, emb = model(data.x, data.edge_index)

        loss = loss_fn(out, target)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 10 == 0:
            st.write(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# =========================
# CONVERGÊNCIA E EMBEDDINGS
# =========================
st.subheader("📉 Convergência do Modelo")

if len(loss_history) > 5:
    fig_loss = px.line(y=loss_history, title="Loss Evolution")
    st.plotly_chart(fig_loss, use_container_width=True)

st.subheader("🧬 Embeddings (PCA - Estrutura Latente)")

emb_np = emb.detach().numpy()

pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb_np)

df = {
    "node": node_names,
    "x": emb_2d[:, 0],
    "y": emb_2d[:, 1],
}

fig = px.scatter(df, x="x", y="y", text="node",
                 title="Espaço Latente da Rede Educacional")

st.plotly_chart(fig, use_container_width=True)

# =========================
# INTERPRETAÇÃO FINAL
# =========================
st.subheader("🧪 Interpretação Científica")

st.write("""
- Nós próximos = maior similaridade educacional
- TH/TPs formam núcleo central
- ODS atua como regulador global
- TA e 5Rs funcionam como nós de suporte sistêmico
""")
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

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
# MODELO GAT AVANÇADO
# =========================
class GATHK5(nn.Module):
    def __init__(self):
        super().__init__()

        self.gat1 = GATConv(3, 16, heads=2, concat=True)
        self.gat2 = GATConv(32, 16, heads=2, concat=True)
        self.gat3 = GATConv(32, 8, heads=1, concat=False)

        self.out = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, edge_index):

        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)

        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)

        x = F.elu(self.gat3(x, edge_index))

        embeddings = x
        out = torch.sigmoid(self.out(x))  # Estabilidade com sigmoid

        return out, embeddings

# =========================
# CONFIGURANDO OTIMIZADOR E PERDA
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
# TARGET CIENTÍFICO
# =========================
target = torch.tensor([
    [TH],
    [TPs],
    [TA],
    [0.3 * TH],
    [0.5 * TPs],
    [RS],
    [ODS]
], dtype=torch.float)

# =========================
# SIMULAÇÃO DE POLÍTICAS EDUCAÇÃO REAIS (LBI, PNTA, BNCC, LDBs)
# =========================
st.header("🌍 Simulação de Políticas Educacionais")

# Definição de cenários de políticas
policies = {
    "Aumento de Tecnologia Humana (TH)": {"TH": 0.9, "TPs": 0.6, "TA": 0.5, "5Rs": 0.5, "ODS": 0.8},
    "Aumento de Tecnologias Pedagógicas (TPs)": {"TH": 0.7, "TPs": 1.0, "TA": 0.5, "5Rs": 0.5, "ODS": 0.8},
    "Foco em Sustentabilidade (5Rs)": {"TH": 0.7, "TPs": 0.6, "TA": 0.5, "5Rs": 1.0, "ODS": 0.8},
    "Aumento de ODS Globais": {"TH": 0.7, "TPs": 0.6, "TA": 0.5, "5Rs": 0.5, "ODS": 1.0},
    "Cenário Base (Sem alteração)": {"TH": 0.7, "TPs": 0.6, "TA": 0.5, "5Rs": 0.5, "ODS": 0.8},
}

selected_policy = st.selectbox("Escolha a Política Educacional", list(policies.keys()))

policy_values = policies[selected_policy]
st.write(f"Política Selecionada: {selected_policy}")
st.write(f"Configuração da Política: {policy_values}")

# Calcular o índice de inclusão para a política selecionada
index_policy = I(policy_values["TH"], policy_values["TPs"], policy_values["TA"], policy_values["5Rs"], policy_values["ODS"])
st.metric("Índice de Inclusão I(t) para esta Política", round(index_policy, 3))

# =========================
# VISUALIZAÇÃO POLÍTICA
# =========================
policy_df = pd.DataFrame(list(policy_values.items()), columns=["Variável", "Valor"])
fig_policy = px.bar(policy_df, x="Variável", y="Valor", title="Impacto das Variáveis na Política Educacional")
st.plotly_chart(fig_policy, use_container_width=True)

# =========================
# TREINAMENTO E MONITORAMENTO
# =========================
train = st.button("🚀 Treinar Modelo GAT")

if train:

    model.train()

    for epoch in range(80):

        optimizer.zero_grad()

        out, emb = model(data.x, data.edge_index)

        loss = loss_fn(out, target)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 10 == 0:
            st.write(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# =========================
# CONVERGÊNCIA E EMBEDDINGS
# =========================
st.subheader("📉 Convergência do Modelo")

if len(loss_history) > 5:
    fig_loss = px.line(y=loss_history, title="Loss Evolution")
    st.plotly_chart(fig_loss, use_container_width=True)

st.subheader("🧬 Embeddings (PCA - Estrutura Latente)")

emb_np = emb.detach().numpy()

pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb_np)

df = {
    "node": node_names,
    "x": emb_2d[:, 0],
    "y": emb_2d[:, 1],
}

fig = px.scatter(df, x="x", y="y", text="node",
                 title="Espaço Latente da Rede Educacional")

st.plotly_chart(fig, use_container_width=True)

# =========================
# INTERPRETAÇÃO FINAL
# =========================
st.subheader("🧪 Interpretação Científica")

st.write("""
- Nós próximos = maior similaridade educacional
- TH/TPs formam núcleo central
- ODS atua como regulador global
- TA e 5Rs funcionam como nós de suporte sistêmico
""")
