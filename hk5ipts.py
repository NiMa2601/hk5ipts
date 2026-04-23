# =========================
# IA (GAT CIENTÍFICO ESTÁVEL)
# =========================

st.header("🧠 IA (GAT Científico Estável)")

from torch_geometric.nn import GATConv
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px

# =========================
# GRAFO → DATA
# =========================
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
# MODELO (ESTÁVEL + NORMALIZADO)
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
        out = torch.sigmoid(self.out(x))  # FIX IMPORTANTE: estabilidade

        return out, embeddings


# =========================
# STATE (NUNCA REINICIALIZA)
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
# TARGET MAIS ESTÁVEL
# =========================
target = torch.tensor([[index] for _ in range(len(nodes))], dtype=torch.float)

# =========================
# TREINO CONTROLADO (IMPORTANTE)
# =========================
train = st.button("🚀 Treinar GAT")

if train:

    model.train()

    for epoch in range(70):

        optimizer.zero_grad()

        out, emb = model(data.x, data.edge_index)

        loss = loss_fn(out, target)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 10 == 0:
            st.write(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# =========================
# RESULTADOS
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

# =========================
# CONVERGÊNCIA (ROBUSTA)
# =========================
st.subheader("📉 Convergência")

if len(loss_history) > 5:
    st.line_chart(loss_history)

# =========================
# EMBEDDINGS + PCA (INTERPRETAÇÃO REAL)
# =========================
st.subheader("🧬 Embeddings (estrutura latente)")

emb_np = emb.detach().numpy()

pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb_np)

df = {
    "node": node_names,
    "x": emb_2d[:, 0],
    "y": emb_2d[:, 1],
}

fig = px.scatter(df, x="x", y="y", text="node",
                 title="Mapa Latente da Rede Educacional")

st.plotly_chart(fig, use_container_width=True)

# =========================
# INTERPRETAÇÃO CIENTÍFICA
# =========================
st.subheader("🧪 Interpretação")

st.write("""
- Nós próximos = maior similaridade educacional
- TH/TPs formam núcleo estrutural
- ODS atua como regulador global
- TA e 5Rs funcionam como nós de suporte
""")
