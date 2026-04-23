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
st.sidebar.title("🎛️ HK5-IPTS Control Panel")

TH = st.sidebar.slider("Tecnologia Humana (TH)", 0.0, 1.0, 0.7)
TPs = st.sidebar.slider("Tecnologias Pedagógicas (TPs)", 0.0, 1.0, 0.6)
TA = st.sidebar.slider("Tecnologia Assistiva (TA)", 0.0, 1.0, 0.5)
RS = st.sidebar.slider("Sustentabilidade (5Rs)", 0.0, 1.0, 0.5)
ODS = st.sidebar.slider("ODS Globais", 0.0, 1.0, 0.8)

num_simulations = st.sidebar.slider("Número de Simulações de Monte Carlo", 100, 5000, 1000)

# =========================
# TÍTULO PRINCIPAL
# =========================
st.title("🌍 HK5-IPTS – Sistema de Inclusão Educacional")

# =========================
# CÁLCULO DE I(t)
# =========================
index = I(TH, TPs, TA, RS, ODS)

# Interpretação do índice
def interpretar_indice(valor):
    if valor < 0.2:
        return "🔴 Crítico - Sistema muito frágil", "Necessária intervenção urgente"
    elif valor < 0.4:
        return "🟠 Baixo - Muitos desafios", "Reforço em múltiplas áreas"
    elif valor < 0.6:
        return "🟡 Moderado - Parcialmente inclusivo", "Melhorias pontuais necessárias"
    elif valor < 0.8:
        return "🟢 Bom - Sistema inclusivo", "Manutenção e refinamento"
    else:
        return "🟢🟢 Excelente - Altamente inclusivo", "Sistema otimizado"

status, recomendacao = interpretar_indice(index)

col_metric1, col_metric2 = st.columns(2)
with col_metric1:
    st.metric("📊 Índice de Inclusão I(t)", round(index, 3))
with col_metric2:
    st.info(f"**Status:** {status}\n\n**Recomendação:** {recomendacao}")

# =========================
# COMPONENTES DO ÍNDICE
# =========================
st.subheader("📋 Breakdown dos Componentes")

componentes_data = {
    "Componente": ["TH (30%)", "TPs (25%)", "TA (20%)", "5Rs (15%)", "ODS (10%)"],
    "Valor": [TH, TPs, TA, RS, ODS],
    "Contribuição": [
        round(0.3 * TH, 3),
        round(0.25 * TPs, 3),
        round(0.2 * TA, 3),
        round(0.15 * RS, 3),
        round(0.1 * ODS, 3)
    ]
}

df_componentes = pd.DataFrame(componentes_data)
st.dataframe(df_componentes, use_container_width=True, hide_index=True)

# Gráfico de pizza
fig_pie = go.Figure(data=[go.Pie(
    labels=componentes_data["Componente"],
    values=componentes_data["Contribuição"],
    marker=dict(colors=["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"])
)])
fig_pie.update_layout(title="Contribuição de Cada Componente ao Índice", height=400)
st.plotly_chart(fig_pie, use_container_width=True, key="pie_componentes")

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
        name="Conexões"
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=nodes,
        textposition="top center",
        marker=dict(size=20, color="rgb(66, 135, 245)", 
                   line=dict(width=2, color="darkblue")),
        name="Nós"
    ))
    
    fig.update_layout(
        title="Rede Educacional HK5-IPTS",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        height=500
    )
    
    return fig

st.subheader("🔗 Topologia da Rede Educacional")
st.plotly_chart(create_graph(), use_container_width=True, key="graph_main")

# Legendas dos nós
st.markdown("""
| Nó | Significado |
|----|------------|
| **TH** | Tecnologia Humana (núcleo ético) |
| **TPs** | Tecnologias Pedagógicas (metodologia) |
| **TA** | Tecnologia Assistiva (acessibilidade) |
| **DU** | Docentes e Usuários (agentes) |
| **DUA** | Desenho Universal para Aprendizagem |
| **5Rs** | Sustentabilidade (Reduzir, Reutilizar, Reciclar, Recuperar, Repensar) |
| **ODS** | Objetivos de Desenvolvimento Sustentável (ONU) |
""")

# =========================
# DOSSÊ CIENTÍFICO
# =========================
st.header("📚 Dossiê Científico Integrado")

with st.expander("📖 Modelo Matemático", expanded=True):
    st.markdown("""
    ### Fórmula de Inclusão
    
    $$I(t) = 0.3 \\times TH + 0.25 \\times TPs + 0.2 \\times TA + 0.15 \\times 5Rs + 0.1 \\times ODS$$
    
    Onde cada componente é normalizado no intervalo **[0, 1]**:
    
    - **α = 0.30** → TH é o componente mais crítico (tecnologia humana)
    - **β = 0.25** → TPs sustentam a metodologia educacional
    - **γ = 0.20** → TA garante acessibilidade
    - **δ = 0.15** → 5Rs promovem sustentabilidade
    - **ε = 0.10** → ODS alinham com metas globais
    """)

with st.expander("🎯 Hipótese Científica"):
    st.markdown("""
    A inclusão educacional é um fenômeno **emergente** que surge da **interação dinâmica** entre:
    
    1. **Tecnologia centrada no humano** (não apenas máquinas, mas pessoas)
    2. **Pedagogia mediada por tecnologia** (métodos adaptados)
    3. **Sistemas de acessibilidade** (tecnologia assistiva)
    4. **Sustentabilidade dos processos** (longevidade)
    5. **Alinhamento com objetivos globais** (agenda 2030)
    
    A rede educacional funciona como um **grafo ponderado** onde cada nó influencia os demais.
    """)

with st.expander("🔬 Interpretação dos Resultados"):
    st.markdown(f"""
    ### Seu Índice Atual: **{round(index, 3)}**
    
    #### Status: {status}
    
    **Análise Detalhada:**
    
    | Métrica | Valor | Análise |
    |---------|-------|--------|
    | TH (Tecnologia Humana) | {TH} | {"✅ Excelente" if TH >= 0.8 else "⚠️ Moderado" if TH >= 0.5 else "❌ Baixo"} |
    | TPs (Pedagógicas) | {TPs} | {"✅ Excelente" if TPs >= 0.8 else "⚠️ Moderado" if TPs >= 0.5 else "❌ Baixo"} |
    | TA (Assistiva) | {TA} | {"✅ Excelente" if TA >= 0.8 else "⚠️ Moderado" if TA >= 0.5 else "❌ Baixo"} |
    | 5Rs (Sustentabilidade) | {RS} | {"✅ Excelente" if RS >= 0.8 else "⚠️ Moderado" if RS >= 0.5 else "❌ Baixo"} |
    | ODS (Objetivos Globais) | {ODS} | {"✅ Excelente" if ODS >= 0.8 else "⚠️ Moderado" if ODS >= 0.5 else "❌ Baixo"} |
    
    **Recomendações:**
    """)
    
    # Gerar recomendações automáticas
    recomendacoes = []
    if TH < 0.6:
        recomendacoes.append("🔴 **TH baixo**: Aumentar foco em aspectos humanísticos da tecnologia")
    if TPs < 0.6:
        recomendacoes.append("🔴 **TPs baixo**: Melhorar métodos pedagógicos e capacitação docente")
    if TA < 0.6:
        recomendacoes.append("🔴 **TA baixo**: Investir em tecnologia assistiva e acessibilidade")
    if RS < 0.6:
        recomendacoes.append("🔴 **5Rs baixo**: Avaliar sustentabilidade ambiental e social")
    if ODS < 0.6:
        recomendacoes.append("🔴 **ODS baixo**: Alinhar com agenda 2030 da ONU")
    
    if not recomendacoes:
        st.success("✅ Sistema bem equilibrado! Mantenha os bons níveis.")
    else:
        for rec in recomendacoes:
            st.warning(rec)

# =========================
# IA (GAT CIENTÍFICO)
# =========================

st.header("🧠 IA – Sistema GAT Científico Avançado")

st.markdown("""
**Graph Attention Network (GAT)**: Rede neural que aprende relações entre nós da rede.
Ela identifica como cada componente influencia o índice final através de **atenção ponderada**.
""")

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
    st.info(f"Status: {'✅ Treinamento ativo' if train else '⏸️ Aguardando início do treinamento...'}")

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
        
        progress = (epoch + 1) / 70
        progress_bar.progress(progress)
        
        if epoch % 10 == 0:
            status_text.write(f"**Epoch {epoch}/70** | Loss: `{loss.item():.6f}`")
        
        if epoch % 5 == 0 and len(st.session_state.loss_history) > 0:
            loss_df = pd.DataFrame({
                "Epoch": range(len(st.session_state.loss_history)),
                "Loss": st.session_state.loss_history
            })
            loss_chart.line_chart(loss_df.set_index("Epoch"), use_container_width=True)
    
    st.session_state.training_complete = True
    status_text.success("✅ Treinamento concluído! O modelo aprendeu os padrões da rede.")

# =========================
# RESULTADOS
# =========================
st.divider()

model.eval()
with torch.no_grad():
    out, emb = model(data.x, data.edge_index)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Previsões por Nó (GAT)")
    predictions = torch.clamp(out, 0, 1).detach().numpy()
    results_df = pd.DataFrame({
        "Nó": nodes,
        "Previsão GAT": predictions.flatten().round(4),
        "Interpretação": [
            "🟢 Forte" if p >= 0.7 else "🟡 Médio" if p >= 0.4 else "🔴 Fraco"
            for p in predictions.flatten()
        ]
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📈 Análise Comparativa")
    avg_index = float(out.mean())
    
    st.metric("🧠 I(t) Predito (GAT)", round(avg_index, 4))
    st.metric("📊 I(t) Calculado", round(index, 4))
    
    delta = round(avg_index - index, 4)
    delta_pct = round((delta / index * 100) if index > 0 else 0, 2)
    
    col_delta1, col_delta2 = st.columns(2)
    with col_delta1:
        st.metric("Diferença", delta, delta=f"{delta:+.4f}")
    with col_delta2:
        st.metric("Variação %", f"{delta_pct:+.2f}%", delta=f"{delta_pct:+.2f}%")
    
    # Interpretação da diferença
    if abs(delta) < 0.01:
        st.success("✅ Modelo muito bem calibrado!")
    elif delta > 0:
        st.info(f"ℹ️ GAT prediz inclusão {delta_pct:.1f}% mais alta")
    else:
        st.warning(f"⚠️ GAT prediz inclusão {abs(delta_pct):.1f}% mais baixa")

# =========================
# INSIGHTS FINAIS
# =========================
st.divider()
st.header("💡 Insights e Recomendações")

insight_data = {
    "Métrica": ["Força Dominante", "Ponto Fraco", "Recomendação Prioritária"],
    "Valor": [
        f"{['TH', 'TPs', 'TA', '5Rs', 'ODS'][[TH, TPs, TA, RS, ODS].index(max(TH, TPs, TA, RS, ODS))]} ({max(TH, TPs, TA, RS, ODS):.2f})",
        f"{['TH', 'TPs', 'TA', '5Rs', 'ODS'][[TH, TPs, TA, RS, ODS].index(min(TH, TPs, TA, RS, ODS))]} ({min(TH, TPs, TA, RS, ODS):.2f})",
        "Elevar componentes críticos (< 0.5)" if min(TH, TPs, TA, RS, ODS) < 0.5 else "Fortalecer TH (peso maior)"
    ]
}

insight_df = pd.DataFrame(insight_data)
st.dataframe(insight_df, use_container_width=True, hide_index=True)

# =========================
# HISTÓRICO FINAL
# =========================
if st.session_state.loss_history:
    st.divider()
    st.subheader("📉 Histórico de Treinamento GAT")
    
    loss_df = pd.DataFrame({
        "Epoch": range(len(st.session_state.loss_history)),
        "Loss": st.session_state.loss_history
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=loss_df["Epoch"],
        y=loss_df["Loss"],
        mode="lines+markers",
        name="MSE Loss",
        line=dict(color="rgb(255, 0, 0)", width=3),
        marker=dict(size=4)
    ))
    
    # Adicionar linha de tendência
    z = np.polyfit(loss_df["Epoch"], loss_df["Loss"], 2)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=loss_df["Epoch"],
        y=p(loss_df["Epoch"]),
        mode="lines",
        name="Tendência",
        line=dict(color="rgba(0, 0, 255, 0.5)", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title="Convergência do Modelo GAT durante Treinamento",
        xaxis_title="Época",
        yaxis_title="Erro (MSE)",
        hovermode="x unified",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, key="loss_chart_final")
    
    # Estatísticas finais
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric("Loss Inicial", f"{st.session_state.loss_history[0]:.6f}")
    with col_stats2:
        st.metric("Loss Final", f"{st.session_state.loss_history[-1]:.6f}")
    with col_stats3:
        improvement = ((st.session_state.loss_history[0] - st.session_state.loss_history[-1]) / st.session_state.loss_history[0] * 100)
        st.metric("Melhoria", f"{improvement:.1f}%", delta=f"{improvement:+.1f}%")
