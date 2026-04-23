"""
Database - Dados Oficiais para HK5-IPTS
Baseado em INEP, UNESCO, ODS 2030
"""

import pandas as pd

# ========================
# INDICADORES ODS OFICIAIS
# ========================

ODS_INDICADORES = {
    "ODS_4": {
        "titulo": "Educação de Qualidade",
        "descricao": "Garantir educação inclusiva, equitativa e de qualidade",
        "meta_2030": 0.85,
        "indicadores": [
            {
                "codigo": "4.1.1",
                "nome": "Taxa de conclusão de ensino primário",
                "meta": 0.95,
                "peso": 0.15
            },
            {
                "codigo": "4.5.1",
                "nome": "Igualdade de acesso",
                "meta": 0.85,
                "peso": 0.25
            }
        ]
    },
    "ODS_5": {
        "titulo": "Igualdade de Gênero",
        "meta_2030": 0.90
    },
    "ODS_10": {
        "titulo": "Redução de Desigualdades",
        "meta_2030": 0.80
    }
}

# ========================
# DADOS BRASIL OFICIAL (INEP 2023)
# ========================

BRASIL_DADOS = {
    "fonte": "INEP - Instituto Nacional de Estudos e Pesquisas",
    "ano": 2023,
    "taxa_matricula": 0.92,
    "taxa_conclusao": 0.88,
    "professores_especializados": 0.28,
    "escolas_com_acessibilidade": 0.42,
    "alunos_com_deficiencia": 0.035
}

# ========================
# INDICADORES A COLETAR
# ========================

INDICADORES_COLETA = {
    "TH": {
        "nome": "Tecnologia Humana",
        "peso": 0.30,
        "indicadores": [
            {
                "codigo": "TH_001",
                "nome": "Política de Humanização Tecnológica",
                "tipo": "binario",
                "frequencia": "Anual"
            },
            {
                "codigo": "TH_002",
                "nome": "% Docentes Treinados em Ética",
                "tipo": "percentual",
                "frequencia": "Semestral"
            },
            {
                "codigo": "TH_003",
                "nome": "Satisfação de Alunos (1-10)",
                "tipo": "escala",
                "frequencia": "Semestral"
            }
        ]
    },
    "TPs": {
        "nome": "Tecnologias Pedagógicas",
        "peso": 0.25,
        "indicadores": [
            {
                "codigo": "TP_001",
                "nome": "% Metodologias Ativas",
                "tipo": "percentual",
                "frequencia": "Semestral"
            },
            {
                "codigo": "TP_002",
                "nome": "% Turmas com Aprendizagem Personalizada",
                "tipo": "percentual",
                "frequencia": "Semestral"
            },
            {
                "codigo": "TP_003",
                "nome": "% Docentes com Formação Digital",
                "tipo": "percentual",
                "frequencia": "Anual"
            }
        ]
    },
    "TA": {
        "nome": "Tecnologia Assistiva",
        "peso": 0.20,
        "indicadores": [
            {
                "codigo": "TA_001",
                "nome": "% Acessibilidade Física",
                "tipo": "percentual",
                "frequencia": "Anual"
            },
            {
                "codigo": "TA_002",
                "nome": "% Plataformas com WCAG 2.1",
                "tipo": "percentual",
                "frequencia": "Semestral"
            },
            {
                "codigo": "TA_003",
                "nome": "Tipos de TA Disponível",
                "tipo": "inteiro",
                "frequencia": "Anual"
            }
        ]
    },
    "5Rs": {
        "nome": "Sustentabilidade",
        "peso": 0.15,
        "indicadores": [
            {
                "codigo": "SUS_001",
                "nome": "% Redução de Papel",
                "tipo": "percentual",
                "frequencia": "Anual"
            },
            {
                "codigo": "SUS_002",
                "nome": "% Energia Renovável",
                "tipo": "percentual",
                "frequencia": "Anual"
            }
        ]
    },
    "ODS": {
        "nome": "Objetivos Globais",
        "peso": 0.10,
        "indicadores": [
            {
                "codigo": "ODS_001",
                "nome": "Conformidade ODS 4",
                "tipo": "binario",
                "frequencia": "Anual"
            }
        ]
    }
}

# ========================
# FUNÇÕES
# ========================

def get_ods_indicadores():
    return ODS_INDICADORES

def get_brasil_dados():
    return BRASIL_DADOS

def get_indicadores_coleta():
    return INDICADORES_COLETA

def criar_dataframe_indicadores():
    """Cria DataFrame com todos os indicadores a coletar"""
    data = []
    for categoria, info in INDICADORES_COLETA.items():
        for ind in info['indicadores']:
            data.append({
                'Categoria': categoria,
                'Código': ind['codigo'],
                'Indicador': ind['nome'],
                'Tipo': ind['tipo'],
                'Frequência': ind['frequencia']
            })
    return pd.DataFrame(data)
