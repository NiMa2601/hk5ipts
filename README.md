# HK5-IPTS – Sistema de Inclusão Educacional em Rede

## Visão Geral

O HK5-IPTS é um sistema computacional baseado em redes complexas aplicado à educação inclusiva. O modelo integra tecnologia humana (TH), tecnologias pedagógicas (TPs), tecnologia assistiva (TA), sustentabilidade (5Rs) e diretrizes globais (ODS) em uma estrutura dinâmica representada por grafos.

O objetivo é modelar a inclusão educacional como uma propriedade emergente de um sistema em rede.

---

## Arquitetura do Sistema

O sistema é estruturado como um grafo:

- Nós: TH, TPs, TA, DU, DUA, 5Rs, ODS  
- Arestas: relações educacionais e estruturais entre os componentes  

A dinâmica é analisada por meio de um índice de inclusão:

I(t) = αTH + βTPs + γTA + δ5Rs + εODS

---

## Funcionalidades

- Simulação interativa de inclusão educacional
- Visualização de rede (grafo dinâmico)
- Ajuste de parâmetros em tempo real
- Cálculo do índice de inclusão I(t)
- Dossiê científico embutido na aplicação

---

## Fundamentação Teórica

O modelo é baseado em:

- Teoria de redes complexas
- Sistemas dinâmicos aplicados à educação
- Educação inclusiva e acessibilidade
- Integração entre tecnologia humana e tecnológica
- Sustentabilidade educacional (ODS e 5Rs)

---

## Aplicação

O sistema pode ser utilizado para:

- Pesquisa em educação inclusiva
- Simulação de políticas educacionais
- Estudos de acessibilidade
- Modelagem de sistemas educacionais complexos

---

## Execução Local

```bash
pip install streamlit networkx plotly
streamlit run hk5ipts.py
