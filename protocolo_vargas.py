# Amostragem, filtro e padronização do Benchmark 2, portados da extract_samples
# e do laço principal do notebook demo_2_benchmark_anomaly_detection.ipynb
# (repositório ricardovvargas/3w_dataset). Copyright (c) 2019 Ricardo Emanuel
# Vaz Vargas, licença MIT. verificar_contra_original() confere as janelas.
#
# Única diferença em relação ao notebook, seguindo a tese: o critério de 10% de
# ausentes é avaliado antes de os ausentes virarem zero.

from math import ceil

import numpy as np
from sklearn.preprocessing import StandardScaler

TAMANHO_AMOSTRA = 3 * 60          # sample_size
MAX_AMOSTRAS_POR_PERIODO = 15     # max_samples_per_period
FRACAO_TREINO = 0.6               # split_range
MAX_FRAC_AUSENTES = 0.1           # max_nan_percent
MIN_STD = 0.01                    # std_vars_min
CODIGO_NORMAL = 0


def _primeiro(ols, v):
    return ols.index(v)


def _ultimo(ols, v):
    return len(ols) - 1 - ols[::-1].index(v)


def indices_amostras(ols, class_code):
    """
    Porta de extract_samples: devolve as posicoes iniciais das janelas de
    treino, teste normal, teste transiente e teste em estado estavel.
    `ols` e a lista de rotulos brutos (0, N, 100+N, NaN).
    """
    ss = TAMANHO_AMOSTRA
    set_ols = set()
    for ol in ols:
        if ol in set_ols or np.isnan(ol):
            continue
        set_ols.add(int(ol))

    # Numeros maximos de amostras por periodo
    f_idx = _primeiro(ols, CODIGO_NORMAL)
    l_idx = _ultimo(ols, CODIGO_NORMAL)
    max_samples_normal = l_idx - f_idx + 1 - ss
    if max_samples_normal > 0:
        num_normal_samples = min(MAX_AMOSTRAS_POR_PERIODO, max_samples_normal)
        num_train_samples = int(FRACAO_TREINO * num_normal_samples)
        num_test_samples = num_normal_samples - num_train_samples
    else:
        num_normal_samples = 0
        num_train_samples = 0
        num_test_samples = 0

    transient_code = class_code + 100
    if transient_code in set_ols:
        f_idx = _primeiro(ols, transient_code)
        f_idx = f_idx - (ss - 1) if f_idx - (ss - 1) > 0 else 0
        l_idx = _ultimo(ols, transient_code)
        max_transient_samples = l_idx - f_idx + 1 - ss
    else:
        max_transient_samples = 0

    if class_code in set_ols:
        f_idx = _primeiro(ols, class_code)
        f_idx = f_idx - (ss - 1) if f_idx - (ss - 1) > 0 else 0
        l_idx = _ultimo(ols, class_code)
        l_idx = l_idx + (ss - 1) if l_idx + (ss - 1) < len(ols) - 1 else len(ols) - 1
        max_in_regime_samples = l_idx - f_idx + 1 - ss
    else:
        max_in_regime_samples = 0

    # Numeros efetivos, buscando o balanceamento
    num_transient_samples = ceil(num_test_samples / 2)
    num_in_regime_samples = num_test_samples - num_transient_samples
    if (max_transient_samples >= num_transient_samples) and \
       (max_in_regime_samples < num_in_regime_samples):
        num_in_regime_samples = max_in_regime_samples
        num_transient_samples = min(num_test_samples - num_in_regime_samples,
                                    max_transient_samples)
    elif (max_transient_samples < num_transient_samples) and \
         (max_in_regime_samples >= num_in_regime_samples):
        num_transient_samples = max_transient_samples
        num_in_regime_samples = min(num_test_samples - num_transient_samples,
                                    max_in_regime_samples)
    elif (max_transient_samples < num_transient_samples) and \
         (max_in_regime_samples < num_in_regime_samples):
        num_transient_samples = max_transient_samples
        num_in_regime_samples = max_in_regime_samples
        num_test_samples = num_transient_samples + num_in_regime_samples

    treino, teste_normal, transiente, regime = [], [], [], []

    # Periodo normal: janelas contadas a partir do fim do periodo
    f_idx = _primeiro(ols, CODIGO_NORMAL)
    l_idx = _ultimo(ols, CODIGO_NORMAL)
    if num_normal_samples > 0:
        if num_normal_samples == max_samples_normal:
            step_max = 1
        else:
            step_max = (max_samples_normal - 1) // (MAX_AMOSTRAS_POR_PERIODO - 1)
        step = min(ss, step_max)
        for idx in range(num_train_samples):
            treino.append(l_idx - ss + 1 - (num_normal_samples - 1 - idx) * step)
        for idx in range(num_train_samples, num_train_samples + num_test_samples):
            teste_normal.append(l_idx - ss + 1 - (num_normal_samples - 1 - idx) * step)

    # Transiente: espalhado ao longo do periodo
    if num_transient_samples > 0:
        if num_transient_samples == max_transient_samples:
            step_max = 1
        else:
            step_max = (max_transient_samples - 1) // (MAX_AMOSTRAS_POR_PERIODO - 1)
        step = min(np.inf, step_max)
        f_idx = _primeiro(ols, transient_code)
        f_idx = f_idx - (ss - 1) if f_idx - (ss - 1) > 0 else 0
        for idx in range(num_transient_samples):
            transiente.append(int(f_idx + idx * step))

    # Estado estavel: primeiras observacoes privilegiadas
    if num_in_regime_samples > 0:
        if num_in_regime_samples == max_in_regime_samples:
            step_max = 1
        else:
            step_max = (max_in_regime_samples - 1) // (MAX_AMOSTRAS_POR_PERIODO - 1)
        step = min(ss, step_max)
        f_idx = _primeiro(ols, class_code)
        f_idx = f_idx - (ss - 1) if f_idx - (ss - 1) > 0 else 0
        for idx in range(num_in_regime_samples):
            regime.append(f_idx + idx * step)

    return treino, teste_normal, transiente, regime


def _estatisticas(janela):
    """mediana, media, desvio padrao, variancia, maximo e minimo por variavel."""
    feats = []
    for j in range(janela.shape[1]):
        s = janela[:, j]
        feats.extend([np.median(s), np.mean(s), np.std(s), np.var(s), np.max(s), np.min(s)])
    return np.array(feats, dtype=np.float32)


def preparar_instancia(dados, rotulos, class_code):
    """
    dados: array (n_obs, n_vars) com os valores brutos (NaN para ausentes).
    rotulos: rotulos brutos como float (NaN para nao rotulado).
    Devolve (X_train, X_neg, X_pos, info) ou None se faltar treino ou teste.
    """
    ols = [float(v) for v in rotulos]
    treino, neg, trans, reg = indices_amostras(ols, int(class_code))
    pos = trans + reg
    if not treino or not neg or not pos:
        return None

    ss = TAMANHO_AMOSTRA
    bruto_treino = np.vstack([dados[i:i + ss] for i in treino])

    # Filtro de variáveis, uma vez por instância, sobre as janelas de treino.
    frac_ausentes = np.isnan(bruto_treino).mean(axis=0)
    std = np.nanstd(np.nan_to_num(bruto_treino, nan=0.0), axis=0)
    boas = (frac_ausentes <= MAX_FRAC_AUSENTES) & (std > MIN_STD)
    if not boas.any():
        return None

    dados_bons = np.nan_to_num(dados[:, boas], nan=0.0)
    treino_bons = np.vstack([dados_bons[i:i + ss] for i in treino])
    scaler = StandardScaler().fit(treino_bons)

    def feats(posicoes):
        return np.array([_estatisticas(
            scaler.transform(dados_bons[i:i + ss]).astype('float32')) for i in posicoes])

    info = {'n_vars_boas': int(boas.sum()), 'n_transiente': len(trans), 'n_regime': len(reg)}
    return feats(treino), feats(neg), feats(pos), info


def verificar_contra_original(df_raw, class_code, extract_samples_original):
    """
    Executa a extract_samples do notebook de Vargas sobre o mesmo dataframe e
    compara as janelas escolhidas (posicao inicial e rotulo) com as desta porta.
    """
    import pandas as pd
    if not hasattr(pd.DataFrame, 'append'):
        pd.DataFrame.append = lambda self, outro, ignore_index=False: \
            pd.concat([self, outro], ignore_index=ignore_index)
    df = df_raw.reset_index(drop=True)
    tr, ytr, te, yte = extract_samples_original(df, class_code)

    ols = [float(v) for v in df['class']]
    treino, neg, trans, reg = indices_amostras(ols, class_code)
    orig_tr = [int(i) for i in tr.index[::TAMANHO_AMOSTRA]] if len(tr) else []
    orig_te = [int(i) for i in te.index[::TAMANHO_AMOSTRA]] if len(te) else []
    porta_te = neg + trans + reg
    rot_porta = [0] * len(neg) + [class_code + 100] * len(trans) + [class_code] * len(reg)
    return (orig_tr == treino and orig_te == porta_te and list(yte) == rot_porta,
            (orig_tr, treino, orig_te, porta_te))
