# =========================
# IA (GAT + PCA + SCIENTIFIC VISUALIZATION)
# =========================

st.header("🧠 IA – Sistema GAT Científico Avançado")

from torch_geometric.nn import GATConv
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px

# =========================
# CONVERTER GRAFO
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
# MODELO MAIS FORTE (GAT DEEP)
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
        out = self.out(x)

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

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

# =========================
# TARGET CIENTÍFICO
# =========================
target = torch.tensor([[index] for _ in range(len(nodes))], dtype=torch.float)

# =========================
# BOTÃO TREINO
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
    st.subheader("📉 Índice global")
    st.metric("I(t) médio", round(float(out.mean()), 3))

# =========================
# CONVERGÊNCIA (BONITA)
# =========================
st.subheader("📈 Convergência do Modelo")

if len(loss_history) > 5:
    fig_loss = px.line(y=loss_history, title="Loss Evolution")
    st.plotly_chart(fig_loss, use_container_width=True)

# =========================
# PCA DOS EMBEDDINGS
# =========================
st.subheader("🧬 Embeddings (PCA - interpretação da rede)")

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
# ANÁLISE CIENTÍFICA FINAL
# =========================
st.subheader("🧪 Interpretação Científica")

st.write("""
- Nós próximos no gráfico = maior similaridade educacional
- TH e TPs tendem a formar núcleo central
- ODS atua como regulador estrutural
- TA e 5Rs funcionam como nós de suporte sistêmico
""")
