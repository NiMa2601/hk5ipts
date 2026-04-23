"""
Módulo de Coleta de Indicadores Reais
"""

import pandas as pd
from datetime import datetime
from database import INDICADORES_COLETA

class ColectorIndicadores:
    def __init__(self):
        self.dados = {}
        self.timestamp = datetime.now()
    
    def coletar(self, codigo, valor):
        """Coleta um indicador"""
        self.dados[codigo] = {
            'valor': valor,
            'data': datetime.now().isoformat()
        }
        return True
    
    def gerar_relatorio(self):
        """Gera relatório dos dados coletados"""
        return {
            'total_coletado': len(self.dados),
            'data': self.timestamp.isoformat(),
            'dados': self.dados
        }
    
    def exportar_csv(self):
        """Exporta dados em CSV"""
        data = []
        for codigo, info in self.dados.items():
            data.append({
                'Código': codigo,
                'Valor': info['valor'],
                'Data': info['data']
            })
        return pd.DataFrame(data)

def criar_template_coleta():
    """Cria template para coleta de dados"""
    return pd.DataFrame([
        {
            'Categoria': cat,
            'Código': ind['codigo'],
            'Indicador': ind['nome'],
            'Tipo': ind['tipo'],
            'Valor Coletado': '',
            'Data': ''
        }
        for cat, info in INDICADORES_COLETA.items()
        for ind in info['indicadores']
    ])
