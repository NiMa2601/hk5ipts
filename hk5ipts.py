import streamlit as st
# ========================
# IMPORTAR DATABASE E COLETA
# ========================
try:
    from database import get_ods_indicadores, get_brasil_dados
except:
    pass
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="HK5-IPTS Advanced", layout="wide")

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
# INTERPRETAÇÕES DETALHADAS
# =========================
def interpretar_indice_detalhado(valor):
    """Interpretação multi-nível do índice"""
    interpretacoes = {
        0: {
            "status": "🔴 CRÍTICO - Exclusão Sistêmica",
            "cor": "#FF4444",
            "nivel": 0,
            "descricao": "Sistema educacional altamente excludente",
            "problemas": [
                "Ausência quase total de tecnologia humanizada",
                "Pedagogia não inclusiva ou inexistente",
                "Barreiras severas de acessibilidade",
                "Nenhuma preocupação com sustentabilidade",
                "Desconexão com objetivos globais"
            ],
            "acoes_urgentes": [
                "Reforma estrutural imediata",
                "Capacitação emergencial de docentes",
                "Investimento em tecnologia assistiva",
                "Alinhamento com ODS urgentemente"
            ]
        },
        1: {
            "status": "🟠 BAIXO - Muitas Barreiras",
            "cor": "#FF8844",
            "nivel": 1,
            "descricao": "Múltiplos desafios de inclusão presentes",
            "problemas": [
                "Foco insuficiente em humanização da tecnologia",
                "Métodos pedagógicos parcialmente inclusivos",
                "Tecnologia assistiva limitada",
                "Sustentabilidade comprometida",
                "Fraco alinhamento com ODS"
            ],
            "acoes_urgentes": [
                "Reforço em TH (tecnologia humana)",
                "Desenvolvimento pedagógico contínuo",
                "Implementação de TA em larga escala",
                "Plano de sustentabilidade"
            ]
        },
        2: {
            "status": "🟡 MODERADO - Parcialmente Inclusivo",
            "cor": "#FFBB33",
            "nivel": 2,
            "descricao": "Sistema em transição para maior inclusão",
            "problemas": [
                "Tecnologia humana em desenvolvimento",
                "Pedagogia modernizada, mas inconsistente",
                "Acessibilidade presente mas com lacunas",
                "Sustentabilidade em fase inicial",
                "ODS reconhecidos mas não plenamente integrados"
            ],
            "acoes_urgentes": [
                "Consolidar boas práticas em TH",
                "Padronizar TPs em toda rede",
                "Expandir e aprofundar TA",
                "Institucionalizar sustentabilidade"
            ]
        },
        3: {
            "status": "🟢 BOM - Sistema Inclusivo",
            "cor": "#44BB44",
            "nivel": 3,
            "descricao": "Educação inclusiva bem estabelecida",
            "problemas": [
                "Alguns ajustes finos necessários em TH",
                "Pequenas inconsistências em TPs",
                "TA poderia ser mais robusta",
                "Sustentabilidade pode evoluir",
                "ODS bem integrados mas com oportunidades"
            ],
            "acoes_urgentes": [
                "Refinar abordagens de TH",
                "Expandir TPs para novos contextos",
                "Fortalecer TA com novas tecnologias",
                "Aprofundar compromisso com ODS"
            ]
        },
        4: {
            "status": "🟢🟢 EXCELENTE - Altamente Inclusivo",
            "cor": "#00CC44",
            "nivel": 4,
            "descricao": "Educação verdadeiramente inclusiva e transformadora",
            "problemas": [
                "Sistema próximo da otimização"
            ],
            "acoes_urgentes": [
                "Manter excelência através de inovação contínua",
                "Documentar e compartilhar melhores práticas",
                "Liderar transformação educacional",
                "Servir como modelo para outras instituições"
            ]
        }
    }
    
    if valor < 0.2:
        return interpretacoes[0]
    elif valor < 0.4:
        return interpretacoes[1]
    elif valor < 0.6:
        return interpretacoes[2]
    elif valor < 0.8:
        return interpretacoes[3]
    else:
        return interpretacoes[4]

def analisar_componentes(th, tps, ta, rs, ods):
    """Análise detalhada de cada componente"""
    componentes = {
        "TH": {
            "valor": th,
            "peso": 0.30,
            "nome": "Tecnologia Humana",
            "descricao": "Centralidade do ser humano na tecnologia educacional",
            "baixo": "Tecnologia deshumanizada, sem considerações éticas",
            "medio": "Tecnologia com foco parcial no humano",
            "alto": "Tecnologia profundamente humanizada e ética",
            "metricas": [
                "Capacidade de customização individual",
                "Considerações éticas implementadas",
                "Feedback humanizado",
                "Foco em bem-estar estudantil"
            ]
        },
        "TPs": {
            "valor": tps,
            "peso": 0.25,
            "nome": "Tecnologias Pedagógicas",
            "descricao": "Metodologias e ferramentas de ensino-aprendizagem",
            "baixo": "Pedagogia tradicional, sem integração tecnológica",
            "medio": "Pedagogia parcialmente modernizada",
            "alto": "Pedagogia inovadora, totalmente integrada",
            "metricas": [
                "Adaptabilidade a estilos de aprendizagem",
                "Suporte a aprendizagem colaborativa",
                "Feedback inteligente e personalizado",
                "Integração com metodologias ativas"
            ]
        },
        "TA": {
            "valor": ta,
            "peso": 0.20,
            "nome": "Tecnologia Assistiva",
            "descricao": "Ferramentas para acessibilidade e inclusão",
            "baixo": "Acessibilidade precária ou inexistente",
            "medio": "Acessibilidade básica implementada",
            "alto": "Acessibilidade universal e proativa",
            "metricas": [
                "Suporte a deficiências visuais",
                "Suporte a deficiências auditivas",
                "Suporte a deficiências motoras",
                "Suporte a neurodiversidade"
            ]
        },
        "5Rs": {
            "valor": rs,
            "peso": 0.15,
            "nome": "Sustentabilidade (5Rs)",
            "descricao": "Reduzir, Reutilizar, Reciclar, Recuperar, Repensar",
            "baixo": "Sem preocupação com sustentabilidade",
            "medio": "Sustentabilidade em algumas áreas",
            "alto": "Sustentabilidade integral do sistema",
            "metricas": [
                "Redução de resíduos digitais",
                "Reutilização de recursos",
                "Reciclagem de equipamentos",
                "Longevidade das soluções"
            ]
        },
        "ODS": {
            "valor": ods,
            "peso": 0.10,
            "nome": "Objetivos de Desenvolvimento Sustentável",
            "descricao": "Alinhamento com agenda 2030 da ONU",
            "baixo": "Desconexão com objetivos globais",
            "medio": "Alinhamento parcial com ODS",
            "alto": "Alinhamento integral com ODS",
            "metricas": [
                "ODS 4: Educação de qualidade",
                "ODS 5: Igualdade de gênero",
                "ODS 10: Redução de desigualdades",
                "ODS 17: Parcerias para objetivos"
            ]
        }
    }
    return componentes

# =========================
# SIDEBAR (CONTROLE)
# =========================
st.sidebar.title("🎛️ HK5-IPTS Control Panel")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Parâmetros do Sistema")

TH = st.sidebar.slider("🤝 Tecnologia Humana (TH)", 0.0, 1.0, 0.7, 
                       help="Quanto a tecnologia é centrada no ser humano?")
TPs = st.sidebar.slider("📚 Tecnologias Pedagógicas (TPs)", 0.0, 1.0, 0.6,
                        help="Quão avançada é a pedagogia?")
TA = st.sidebar.slider("♿ Tecnologia Assistiva (TA)", 0.0, 1.0, 0.5,
                       help="Qual o nível de acessibilidade?")
RS = st.sidebar.slider("🌱 Sustentabilidade (5Rs)", 0.0, 1.0, 0.5,
                       help="Quanto sustentável é o sistema?")
ODS = st.sidebar.slider("🌍 ODS Globais", 0.0, 1.0, 0.8,
                        help="Alinhamento com objetivos globais?")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configurações Avançadas")

num_simulations = st.sidebar.slider("Número de Simulações", 100, 5000, 1000)
show_advanced = st.sidebar.checkbox("Mostrar análises avançadas", True)
show_predictions = st.sidebar.checkbox("Mostrar predições GAT", True)

# =========================
# TÍTULO PRINCIPAL
# =========================
st.title("🌍 HK5-IPTS – Sistema Inteligente de Inclusão Educacional")

# =========================
# CÁLCULO E INTERPRETAÇÃO
# =========================
index = I(TH, TPs, TA, RS, ODS)
interpretacao = interpretar_indice_detalhado(index)

# Dashboard Principal
col_metric1, col_metric2, col_metric3 = st.columns([2, 2, 2])

with col_metric1:
    st.metric(
        "📊 Índice de Inclusão I(t)",
        round(index, 4),
        delta=f"{interpretacao['nivel']}/4 - {interpretacao['status'].split(' - ')[0]}"
    )

with col_metric2:
    st.markdown(f"""
    <div style="background-color: {interpretacao['cor']}20; border-left: 4px solid {interpretacao['cor']}; padding: 20px; border-radius: 5px;">
        <h4 style="margin: 0; color: {interpretacao['cor']}">{interpretacao['status']}</h4>
        <p style="margin: 5px 0; font-size: 12px;">{interpretacao['descricao']}</p>
    </div>
    """, unsafe_allow_html=True)

with col_metric3:
    # Score visual
    score_pct = int(index * 100)
    st.markdown(f"""
    <div style="text-align: center;">
        <div style="font-size: 48px; font-weight: bold; color: {interpretacao['cor']}">{score_pct}%</div>
        <div style="font-size: 12px; color: gray">Nível de Inclusão</div>
        <div style="width: 100%; background-color: #eee; border-radius: 10px; overflow: hidden; height: 8px; margin-top: 10px;">
            <div style="width: {score_pct}%; background-color: {interpretacao['cor']}; height: 100%; transition: width 0.3s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# AÇÕES PRIORITÁRIAS
# =========================
st.markdown("---")
st.subheader("🎯 Ações Prioritárias Recomendadas")

cols_acoes = st.columns(len(interpretacao['acoes_urgentes']))
for i, acao in enumerate(interpretacao['acoes_urgentes']):
    with cols_acoes[i]:
        st.info(f"**{i+1}.** {acao}")

# =========================
# ANÁLISE DE COMPONENTES
# =========================
st.markdown("---")
st.subheader("🔍 Análise Detalhada dos Componentes")

componentes = analisar_componentes(TH, TPs, TA, RS, ODS)
values = [TH, TPs, TA, RS, ODS]
keys = list(componentes.keys())

# Radar Chart
fig_radar = go.Figure(data=go.Scatterpolar(
    r=values,
    theta=keys,
    fill='toself',
    name='Inclusão',
    line_color='rgb(66, 135, 245)'
))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=False,
    height=450
)

col_radar, col_comp_text = st.columns([1.5, 1.5])

with col_radar:
    st.plotly_chart(fig_radar, use_container_width=True, key="radar_main")

with col_comp_text:
    st.markdown("### 📈 Resumo dos Componentes")
    
    for key, comp in componentes.items():
        valor = comp['valor']
        if valor < 0.3:
            nivel_texto = "🔴 Crítico"
        elif valor < 0.5:
            nivel_texto = "🟠 Baixo"
        elif valor < 0.7:
            nivel_texto = "🟡 Médio"
        else:
            nivel_texto = "🟢 Alto"
        
        st.markdown(f"""
        **{key}** - {comp['nome']}  
        Valor: `{valor:.2%}` | {nivel_texto}
        """)

# =========================
# DETALHES POR COMPONENTE
# =========================
if show_advanced:
    st.markdown("---")
    st.subheader("📋 Detalhes por Componente")
    
    for key, comp in componentes.items():
        with st.expander(f"🔹 {comp['nome']} (TH)" if key == "TH" else f"🔹 {comp['nome']}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Indicador circular
                valor = comp['valor']
                cor = "#FF4444" if valor < 0.3 else "#FF8844" if valor < 0.5 else "#FFBB33" if valor < 0.7 else "#44BB44"
                
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 60px; font-weight: bold; color: {cor}">{valor:.0%}</div>
                    <div style="font-size: 12px; margin-top: 10px">Peso: {comp['peso']:.0%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**Descrição:** {comp['descricao']}")
                st.markdown(f"**Valor Baixo:** _{comp['baixo']}_")
                st.markdown(f"**Valor Médio:** _{comp['medio']}_")
                st.markdown(f"**Valor Alto:** _{comp['alto']}_")
                
                st.markdown("**Métricas de Avaliação:**")
                for metrica in comp['metricas']:
                    st.markdown(f"- {metrica}")

# =========================
# GRAFO VISUAL AVANÇADO
# =========================
st.markdown("---")
st.subheader("🔗 Topologia da Rede Educacional")

pos = nx.spring_layout(G, seed=42, k=1, iterations=50)

@st.cache_data
def create_advanced_graph():
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_text, node_color = [], [], [], []
    
    valores_mapa = {"TH": TH, "TPs": TPs, "TA": TA, "DU": 0.5, "DUA": 0.6, "5Rs": RS, "ODS": ODS}
    
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)
        
        # Cor baseada no valor
        val = valores_mapa.get(n, 0.5)
        if val < 0.3:
            node_color.append("#FF4444")
        elif val < 0.5:
            node_color.append("#FF8844")
        elif val < 0.7:
            node_color.append("#FFBB33")
        else:
            node_color.append("#44BB44")

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=2.5, color="rgba(125,125,125,0.3)"),
        hoverinfo="none",
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=12, color="white"),
        marker=dict(
            size=30,
            color=node_color,
            line=dict(width=2, color="darkblue"),
            opacity=0.9
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False
    ))
    
    fig.update_layout(
        title="Rede Educacional HK5-IPTS (Cor = Nível de Inclusão)",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        height=500,
        plot_bgcolor="rgba(240, 240, 240, 0.5)"
    )
    
    return fig

st.plotly_chart(create_advanced_graph(), use_container_width=True, key="graph_advanced")

# Legenda interativa
col_leg1, col_leg2, col_leg3, col_leg4 = st.columns(4)
with col_leg1:
    st.markdown("**TH** - Tecnologia Humana")
with col_leg2:
    st.markdown("**TPs** - Tecnologias Pedagógicas")
with col_leg3:
    st.markdown("**TA** - Tecnologia Assistiva")
with col_leg4:
    st.markdown("**ODS** - Objetivos Globais")

# =========================
# BENCHMARKING
# =========================
st.markdown("---")
st.subheader("📊 Benchmarking e Comparação")

benchmark_data = {
    "Instituição": ["Sua Instituição", "Média Nacional", "Líderes Globais", "Meta ODS 2030"],
    "Índice": [index, 0.5, 0.75, 0.85],
    "Tipo": ["Sua Realidade", "Comparação", "Excelência", "Meta"]
}

df_bench = pd.DataFrame(benchmark_data)

fig_bench = go.Figure()

for tipo in df_bench["Tipo"].unique():
    dados_tipo = df_bench[df_bench["Tipo"] == tipo]
    cor = {
        "Sua Realidade": "#4285F4",
        "Comparação": "#EA4335",
        "Excelência": "#34A853",
        "Meta": "#FBBC04"
    }[tipo]
    
    fig_bench.add_trace(go.Bar(
        y=dados_tipo["Instituição"],
        x=dados_tipo["Índice"],
        orientation="h",
        name=tipo,
        marker=dict(color=cor),
        text=[f"{v:.0%}" for v in dados_tipo["Índice"]],
        textposition="auto"
    ))

fig_bench.update_layout(
    title="Posicionamento em Relação a Benchmarks",
    xaxis_title="Índice de Inclusão",
    yaxis_title="",
    barmode="group",
    height=350,
    xaxis=dict(range=[0, 1])
)

st.plotly_chart(fig_bench, use_container_width=True, key="bench_chart")

# Interpretação do benchmark
diferenca_nacional = index - 0.5
diferenca_lider = index - 0.75
diferenca_meta = index - 0.85

col_b1, col_b2, col_b3 = st.columns(3)

with col_b1:
    if diferenca_nacional > 0:
        st.success(f"✅ Acima da média nacional (+{diferenca_nacional:.1%})")
    else:
        st.warning(f"⚠️ Abaixo da média nacional ({diferenca_nacional:.1%})")

with col_b2:
    if diferenca_lider > 0:
        st.info(f"ℹ️ {diferenca_lider:.1%} abaixo dos líderes globais")
    else:
        st.success(f"✅ Próximo aos líderes globais ({abs(diferenca_lider):.1%})")

with col_b3:
    if diferenca_meta > 0:
        st.warning(f"⚠️ {diferenca_meta:.1%} abaixo da meta ODS 2030")
    else:
        st.success(f"✅ Acima da meta ODS 2030!")

# =========================
# IA (GAT CIENTÍFICO)
# =========================

st.markdown("---")
st.header("🧠 IA – Predição com Graph Attention Network")

st.markdown("""
O modelo **GAT (Graph Attention Network)** aprende automaticamente como os componentes 
da rede educacional se relacionam e influenciam o índice final de inclusão. 
Ele identifica padrões que podem não ser óbvios na análise linear.
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
# MODELO GAT AVANÇADO
# =========================
class GATHK5Advanced(nn.Module):
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
    st.session_state.model = GATHK5Advanced()
    st.session_state.loss_history = []
    st.session_state.training_complete = False
    st.session_state.training_count = 0

model = st.session_state.model
target = torch.tensor([[index] for _ in range(len(nodes))], dtype=torch.float)

# =========================
# TREINAMENTO
# =========================
col_train1, col_train2, col_train3 = st.columns([1, 2, 1])

with col_train1:
    train = st.button("🚀 Treinar GAT", use_container_width=True, key="train_btn")

with col_train2:
    if st.session_state.training_complete:
        st.success(f"✅ Modelo treinado {st.session_state.training_count}x")
    else:
        st.info("⏸️ Clique em 'Treinar GAT' para iniciar")

with col_train3:
    if st.button("🔄 Resetar", use_container_width=True):
        st.session_state.model = GATHK5Advanced()
        st.session_state.loss_history = []
        st.session_state.training_complete = False
        st.session_state.training_count = 0
        st.rerun()

if train:
    st.session_state.training_complete = False
    st.session_state.loss_history = []
    st.session_state.training_count += 1
    
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
            status_text.markdown(f"⏳ **Época {epoch}/70** | Loss: `{loss.item():.6f}`")
        
        if epoch % 5 == 0 and len(st.session_state.loss_history) > 0:
            loss_df = pd.DataFrame({
                "Epoch": range(len(st.session_state.loss_history)),
                "Loss": st.session_state.loss_history
            })
            loss_chart.line_chart(loss_df.set_index("Epoch"), use_container_width=True)
    
    st.session_state.training_complete = True
    status_text.success("✅ **Treinamento concluído!** Modelo aprendeu os padrões.")

# =========================
# RESULTADOS GAT
# =========================
if show_predictions:
    st.markdown("---")
    st.subheader("🔮 Predições do Modelo GAT")
    
    model.eval()
    with torch.no_grad():
        out, emb = model(data.x, data.edge_index)

    col_pred1, col_pred2, col_pred3 = st.columns([1.5, 1.5, 1])

    with col_pred1:
        st.markdown("**📊 Predições por Nó**")
        predictions = torch.clamp(out, 0, 1).detach().numpy()
        results_df = pd.DataFrame({
            "Nó": nodes,
            "Predição GAT": [f"{p:.4f}" for p in predictions.flatten()],
            "Status": [
                "🟢" if p >= 0.7 else "🟡" if p >= 0.4 else "🔴"
                for p in predictions.flatten()
            ]
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    with col_pred2:
        st.markdown("**📈 Análise Comparativa**")
        avg_index_gat = float(out.mean())
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("🧠 GAT", round(avg_index_gat, 4))
        with col_p2:
            st.metric("📊 Calculado", round(index, 4))
        
        delta = avg_index_gat - index
        delta_pct = (delta / index * 100) if index > 0 else 0
        
        if abs(delta) < 0.01:
            st.success("✅ Bem calibrado!")
        elif delta > 0:
            st.info(f"ℹ️ GAT prevê {abs(delta_pct):.1f}% maior")
        else:
            st.warning(f"⚠️ GAT prevê {abs(delta_pct):.1f}% menor")

    with col_pred3:
        st.markdown("**⚡ Insights**")
        st.markdown(f"""
        - **Diferença:** {delta:+.4f}
        - **Variação:** {delta_pct:+.2f}%
        - **Confiança:** {"Alta" if abs(delta) < 0.05 else "Média" if abs(delta) < 0.1 else "Baixa"}
        """)

# =========================
# HISTÓRICO DE TREINAMENTO
# =========================
if st.session_state.loss_history:
    st.markdown("---")
    st.subheader("📉 Histórico de Treinamento")
    
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
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(255, 0, 0, 0.1)"
    ))
    
    # Tendência
    z = np.polyfit(loss_df["Epoch"], loss_df["Loss"], 2)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=loss_df["Epoch"],
        y=p(loss_df["Epoch"]),
        mode="lines",
        name="Tendência",
        line=dict(color="rgba(0, 100, 255, 0.7)", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title="Convergência do Modelo Durante Treinamento",
        xaxis_title="Época",
        yaxis_title="Erro (MSE)",
        hovermode="x unified",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, key="loss_chart")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.metric("Perda Inicial", f"{st.session_state.loss_history[0]:.6f}")
    with col_s2:
        st.metric("Perda Final", f"{st.session_state.loss_history[-1]:.6f}")
    with col_s3:
        improvement = ((st.session_state.loss_history[0] - st.session_state.loss_history[-1]) / st.session_state.loss_history[0] * 100)
        st.metric("Melhoria", f"{improvement:.1f}%", delta=f"{improvement:+.1f}%")

# =========================
# CENÁRIOS E SIMULAÇÕES
# =========================
st.markdown("---")
st.header("🎮 Simulações e Cenários")

st.markdown("""
Explore cenários "e se?" (What-if) para entender como mudanças nos componentes 
afetam o índice de inclusão.
""")

col_scen1, col_scen2 = st.columns(2)

with col_scen1:
    st.subheader("📌 Cenários Pré-definidos")
    
    cenario_selecionado = st.radio(
        "Escolha um cenário:",
        [
            "Cenário Atual",
            "Foco em TH",
            "Foco em TPs",
            "Foco em TA",
            "Foco em Sustentabilidade",
            "Foco em ODS",
            "Cenário Otimista",
            "Cenário Pessimista"
        ]
    )
    
    cenarios = {
        "Cenário Atual": (TH, TPs, TA, RS, ODS),
        "Foco em TH": (0.9, TPs, TA, RS, ODS),
        "Foco em TPs": (TH, 0.9, TA, RS, ODS),
        "Foco em TA": (TH, TPs, 0.9, RS, ODS),
        "Foco em Sustentabilidade": (TH, TPs, TA, 0.9, ODS),
        "Foco em ODS": (TH, TPs, TA, RS, 0.9),
        "Cenário Otimista": (0.9, 0.9, 0.9, 0.9, 0.9),
        "Cenário Pessimista": (0.3, 0.3, 0.3, 0.3, 0.3)
    }
    
    th_cen, tps_cen, ta_cen, rs_cen, ods_cen = cenarios[cenario_selecionado]
    index_cen = I(th_cen, tps_cen, ta_cen, rs_cen, ods_cen)
    
    st.metric("Índice no Cenário", round(index_cen, 4), 
              delta=round(index_cen - index, 4))

with col_scen2:
    st.subheader("🎯 Ajuste Manual")
    
    th_manual = st.slider("TH no cenário", 0.0, 1.0, TH, key="th_manual")
    tps_manual = st.slider("TPs no cenário", 0.0, 1.0, TPs, key="tps_manual")
    ta_manual = st.slider("TA no cenário", 0.0, 1.0, TA, key="ta_manual")
    rs_manual = st.slider("5Rs no cenário", 0.0, 1.0, RS, key="rs_manual")
    ods_manual = st.slider("ODS no cenário", 0.0, 1.0, ODS, key="ods_manual")
    
    index_manual = I(th_manual, tps_manual, ta_manual, rs_manual, ods_manual)
    
    st.metric("Índice Customizado", round(index_manual, 4),
              delta=round(index_manual - index, 4))

# Comparação Visual
fig_comp = go.Figure()

fig_comp.add_trace(go.Bar(
    x=["TH", "TPs", "TA", "5Rs", "ODS"],
    y=[TH, TPs, TA, RS, ODS],
    name="Atual",
    marker_color="rgb(100, 150, 200)"
))

fig_comp.add_trace(go.Bar(
    x=["TH", "TPs", "TA", "5Rs", "ODS"],
    y=[th_manual, tps_manual, ta_manual, rs_manual, ods_manual],
    name="Cenário",
    marker_color="rgb(200, 150, 100)"
))

fig_comp.update_layout(
    title="Comparação: Situação Atual vs Cenário",
    barmode="group",
    height=350,
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(fig_comp, use_container_width=True, key="comp_chart")

# =========================
# RECOMENDAÇÕES FINAIS
# =========================
st.markdown("---")
st.header("📋 Relatório Executivo e Recomendações")

with st.expander("📄 Gerar Relatório Completo", expanded=False):
    relatorio = f"""
    # RELATÓRIO DE INCLUSÃO EDUCACIONAL - HK5-IPTS
    
    **Data:** {datetime.now().strftime('%d/%m/%Y %H:%M')}
    
    ## 1. SITUAÇÃO ATUAL
    
    - **Índice de Inclusão:** {index:.4f} ({int(index*100)}%)
    - **Status:** {interpretacao['status']}
    - **Nível:** {interpretacao['nivel']}/4
    
    ## 2. COMPONENTES
    
    | Componente | Valor | Contribuição | Status |
    |-----------|-------|-------------|--------|
    | TH (Tecnologia Humana) | {TH:.2%} | {0.3*TH:.4f} | {'🟢' if TH >= 0.7 else '🟡' if TH >= 0.4 else '🔴'} |
    | TPs (Pedagogias) | {TPs:.2%} | {0.25*TPs:.4f} | {'🟢' if TPs >= 0.7 else '🟡' if TPs >= 0.4 else '🔴'} |
    | TA (Assistiva) | {TA:.2%} | {0.2*TA:.4f} | {'🟢' if TA >= 0.7 else '🟡' if TA >= 0.4 else '🔴'} |
    | 5Rs (Sustentabilidade) | {RS:.2%} | {0.15*RS:.4f} | {'🟢' if RS >= 0.7 else '🟡' if RS >= 0.4 else '🔴'} |
    | ODS (Objetivos) | {ODS:.2%} | {0.1*ODS:.4f} | {'🟢' if ODS >= 0.7 else '🟡' if ODS >= 0.4 else '🔴'} |
    
    ## 3. PROBLEMAS IDENTIFICADOS
    
    """
    
    for problema in interpretacao['problemas']:
        relatorio += f"- {problema}\n"
    
    relatorio += "\n## 4. AÇÕES RECOMENDADAS\n\n"
    
    for acao in interpretacao['acoes_urgentes']:
        relatorio += f"- {acao}\n"
    
    relatorio += f"""
    
    ## 5. BENCHMARKING
    
    - Sua Instituição: {index:.4f}
    - Média Nacional: 0.5000
    - Líderes Globais: 0.7500
    - Meta ODS 2030: 0.8500
    
    """
    
    st.text_area("Relatório:", value=relatorio, height=400, disabled=True)
    
    col_down1, col_down2 = st.columns(2)
    with col_down1:
        st.download_button(
            label="📥 Baixar Relatório (TXT)",
            data=relatorio,
            file_name=f"relatorio_hk5_ipts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col_down2:
        # CSV com dados
        csv_data = f"""Componente,Valor,Peso,Contribuição
TH,{TH},0.30,{0.3*TH}
TPs,{TPs},0.25,{0.25*TPs}
TA,{TA},0.20,{0.2*TA}
5Rs,{RS},0.15,{0.15*RS}
ODS,{ODS},0.10,{0.1*ODS}
TOTAL,{index},1.00,{index}
"""
        st.download_button(
            label="📊 Baixar Dados (CSV)",
            data=csv_data,
            file_name=f"dados_hk5_ipts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# =========================
# RODAPÉ
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: gray; font-size: 12px;">
    <p><b>HK5-IPTS</b> - Sistema Inteligente de Inclusão Educacional</p>
    <p>Desenvolvido com Streamlit, PyTorch e Graph Neural Networks</p>
    <p>Alinhado com ODS 4 (Educação de Qualidade) e ODS 5 (Igualdade de Gênero)</p>
</div>
""", unsafe_allow_html=True)

