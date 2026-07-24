#Pré-processamento comum: rótulos, filtragem de variáveis, janelas e estatísticas.

import numpy as np
import pandas as pd

from config import TAMANHO_JANELA


def normalizar_rotulos(serie: pd.Series) -> pd.Series:
    """
    Mapeia a convenção de rótulos do 3W v2.0:
      0           -> 0  (normal)
      X (1-9)     -> X  (estado estável da anomalia X)
      10X (>=100) -> X  (transiente, tratado como o mesmo tipo do estado estável)
      -1          -> -1 (não rotulado, ignorado no pipeline)
    """
    def mapear(v):
        if v == 0 or v == -1:
            return v
        return v - 100 if v >= 100 else v
    return serie.map(mapear).astype(int)


def filtrar_variaveis(janela_bruta: np.ndarray, janela_norm: np.ndarray) -> np.ndarray:
    """
    Critérios de Vargas (2019) avaliados nos dados brutos, aplicados na janela
    normalizada: colunas com mais de 10% de valores ausentes ou desvio padrão
    menor que 1 são zeradas.
    """
    resultado = janela_norm.copy().astype(np.float64)
    n = janela_bruta.shape[0]
    for j in range(janela_bruta.shape[1]):
        col = janela_bruta[:, j]
        if np.isnan(col).sum() / n > 0.10:
            resultado[:, j] = 0.0
            continue
        if np.std(np.nan_to_num(col, nan=0.0)) < 1.0:
            resultado[:, j] = 0.0
            continue
        resultado[:, j] = np.nan_to_num(resultado[:, j], nan=0.0)
    return resultado


def extrair_estatisticas(janela_2d: np.ndarray) -> np.ndarray:
    """42 features: mediana, média, desvio padrão, variância, máximo e mínimo por variável."""
    feats = []
    for j in range(janela_2d.shape[1]):
        s = janela_2d[:, j]
        feats.extend([np.median(s), np.mean(s), np.std(s), np.var(s), np.max(s), np.min(s)])
    return np.array(feats, dtype=np.float32)


def extrair_janelas(dados_brutos, dados_norm, passo, max_amostras=None, privilegiar_ultimas=False):
    """
    Percorre a série temporal criando janelas de TAMANHO_JANELA observações.
    privilegiar_ultimas=True começa a contagem pelo final da série, a região
    mais próxima do início da anomalia (Vargas, 2019).
    """
    n = len(dados_norm)
    indices = list(range(0, n - TAMANHO_JANELA, passo))
    if privilegiar_ultimas:
        indices = list(reversed(indices))
    janelas = []
    for i in indices:
        if max_amostras is not None and len(janelas) >= max_amostras:
            break
        janelas.append(filtrar_variaveis(
            dados_brutos[i:i + TAMANHO_JANELA],
            dados_norm[i:i + TAMANHO_JANELA],
        ))
    return janelas


def preparar_janelas_do_poco(df, variaveis, passo_treino, passo_teste, max_amostras_teste):
    """
    A partir do dataframe bruto de um poço, separa normal/anômalo, normaliza
    com StandardScaler (ajustado só no treino) e extrai as janelas de treino,
    teste-normal e teste-anômalo. Retorna None se o poço não tiver dados
    suficientes para as três partições.
    """
    from sklearn.preprocessing import StandardScaler

    df_normal = df[df['class'] == 0]
    df_anomalia = df[(df['class'] != 0) & (df['class'] != -1)]

    corte = int(len(df_normal) * 0.6)
    bruto_treino = df_normal.iloc[:corte][variaveis].values
    bruto_val_neg = df_normal.iloc[corte:][variaveis].values
    bruto_val_pos = df_anomalia[variaveis].values

    scaler = StandardScaler()
    scaler.fit(np.nan_to_num(bruto_treino, nan=0.0))

    norm_treino = scaler.transform(np.nan_to_num(bruto_treino, nan=0.0))
    norm_val_neg = scaler.transform(np.nan_to_num(bruto_val_neg, nan=0.0))
    norm_val_pos = scaler.transform(np.nan_to_num(bruto_val_pos, nan=0.0))

    janelas_treino = extrair_janelas(
        bruto_treino, norm_treino, passo=passo_treino,
        max_amostras=None, privilegiar_ultimas=True)
    janelas_neg = extrair_janelas(
        bruto_val_neg, norm_val_neg, passo=passo_teste,
        max_amostras=max_amostras_teste)
    janelas_pos = extrair_janelas(
        bruto_val_pos, norm_val_pos, passo=passo_teste,
        max_amostras=max_amostras_teste)

    if not janelas_treino or not janelas_neg or not janelas_pos:
        return None

    n_bal = min(len(janelas_neg), len(janelas_pos))
    return janelas_treino, janelas_neg[:n_bal], janelas_pos[:n_bal], n_bal
