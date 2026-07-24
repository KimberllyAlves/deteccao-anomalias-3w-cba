"""
Benchmark do VAE no 3W — detectores D1, D2, D2* e D3.

  D1  — erro de reconstrução (MSE) > percentil 50 do treino
  D2  — distância de Mahalanobis no espaço latente > percentil 50 do treino
  D2* — distância de Mahalanobis > sqrt(chi2(0.95, df=LATENT_DIM))
  D3  — D1 OR D2*

O VAE é treinado uma única vez por poço; os quatro detectores são avaliados
em cima do mesmo modelo. Também mede tempo de treino e latência de
inferência (encoder + Mahalanobis) para a discussão de custo computacional.

Uso: ajuste config.DIRETORIO_BASE para a pasta do 3W v2.0 e rode este arquivo.
"""

import os
import glob
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import chi2

from config import (
    VARIAVEIS, N_FEATURES, DIRETORIO_BASE, CLASSES_BENCHMARK, PASSO_TREINO,
    MAX_AMOSTRAS_TEST, MIN_NORMAL, MIN_ANOMALIA, PASSO_TESTE_POR_CLASSE,
    HIDDEN_DIM, LATENT_DIM, NIVEL_CONFIANCA, DEVICE, fixar_sementes,
)
from preprocessing import normalizar_rotulos, extrair_estatisticas, preparar_janelas_do_poco
from vae import treinar_vae, extrair_mus, calcular_erros_mse, calcular_mahalanobis

warnings.filterwarnings('ignore')

LIMIAR_CHI2 = np.sqrt(chi2.ppf(NIVEL_CONFIANCA, df=LATENT_DIM))


def metricas(y_true, y_pred):
    return {
        'f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precisao': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='micro', zero_division=0),
    }


def main():
    fixar_sementes()

    resultados = {'d1': [], 'd2': [], 'd2_estrela': [], 'd3': []}
    tempos_inferencia = []
    tempos_treino = []
    n_params = None
    descarte = {'normal_curto': 0, 'anomalia_curta': 0, 'janelas_insuf': 0, 'nao_finito': 0}

    print(f"VAE — D1 (MSE) / D2 (Mahalanobis P50) / D2* (Mahalanobis Chi²) / D3 | Device: {DEVICE}")
    print(f"D2*: Mahalanobis > sqrt(chi2({NIVEL_CONFIANCA}, df={LATENT_DIM})) = {LIMIAR_CHI2:.4f}")
    print(f"Classes: {CLASSES_BENCHMARK}\n")

    for classe in CLASSES_BENCHMARK:
        passo_teste = PASSO_TESTE_POR_CLASSE[classe]
        arquivos = glob.glob(os.path.join(DIRETORIO_BASE, classe, "WELL*.parquet"))
        print(f">>> Classe {classe}: {len(arquivos)} arquivo(s) | passo_teste={passo_teste}s")

        for caminho in arquivos:
            nome = os.path.basename(caminho)
            try:
                df = pd.read_parquet(caminho)
                for col in VARIAVEIS:
                    if col not in df.columns:
                        df[col] = np.nan
                df['class'] = normalizar_rotulos(df['class'].fillna(-1))

                if (df['class'] == 0).sum() < MIN_NORMAL:
                    descarte['normal_curto'] += 1
                    print(f"   {nome}: ignorado — normal curto")
                    continue
                if ((df['class'] != 0) & (df['class'] != -1)).sum() < MIN_ANOMALIA:
                    descarte['anomalia_curta'] += 1
                    print(f"   {nome}: ignorado — anomalia curta")
                    continue

                partes = preparar_janelas_do_poco(
                    df, VARIAVEIS, PASSO_TREINO, passo_teste, MAX_AMOSTRAS_TEST)
                if partes is None:
                    descarte['janelas_insuf'] += 1
                    print(f"   {nome}: ignorado — janelas insuficientes")
                    continue
                janelas_treino, janelas_neg, janelas_pos, n_bal = partes

                X_train = np.array([extrair_estatisticas(j) for j in janelas_treino])
                X_neg = np.array([extrair_estatisticas(j) for j in janelas_neg])
                X_pos = np.array([extrair_estatisticas(j) for j in janelas_pos])
                X_test = np.vstack([X_neg, X_pos])
                y_test = np.array([0] * n_bal + [1] * n_bal)

                t0 = time.perf_counter()
                modelo, erros_treino, mus_treino = treinar_vae(X_train, N_FEATURES, HIDDEN_DIM, LATENT_DIM)
                tempos_treino.append(time.perf_counter() - t0)

                if n_params is None:
                    n_params = sum(p.numel() for p in modelo.parameters())

                # D1 — erro de reconstrução
                erros_neg = calcular_erros_mse(modelo, X_neg)
                erros_pos = calcular_erros_mse(modelo, X_pos)
                erros_test = calcular_erros_mse(modelo, X_test)

                if not np.isfinite(erros_neg).all() or not np.isfinite(erros_pos).all():
                    descarte['nao_finito'] += 1
                    print(f"   {nome}: DESCARTADO — erros não finitos")
                    continue

                limiar_d1 = np.percentile(erros_treino, 50)
                pred_d1 = (erros_test > limiar_d1).astype(int)

                # D2 / D2* — distância de Mahalanobis no espaço latente
                mu_ref = mus_treino.mean(axis=0)
                cov_reg = np.cov(mus_treino.T) + np.eye(LATENT_DIM) * 1e-6
                try:
                    cov_inv = np.linalg.inv(cov_reg)
                except np.linalg.LinAlgError:
                    cov_inv = np.linalg.pinv(cov_reg)

                dists_treino = calcular_mahalanobis(mus_treino, mu_ref, cov_inv)
                mus_test = extrair_mus(modelo, X_test)
                dists_test = calcular_mahalanobis(mus_test, mu_ref, cov_inv)

                limiar_d2 = np.percentile(dists_treino, 50)
                pred_d2 = (dists_test > limiar_d2).astype(int)
                pred_d2_estrela = (dists_test > LIMIAR_CHI2).astype(int)

                # mede latência de inferência (encoder + Mahalanobis), janela a janela
                for i in range(X_test.shape[0]):
                    t0_inf = time.perf_counter()
                    mu_i = extrair_mus(modelo, X_test[i:i + 1])
                    calcular_mahalanobis(mu_i, mu_ref, cov_inv)
                    tempos_inferencia.append((time.perf_counter() - t0_inf) * 1000)

                # D3 — combinação
                pred_d3 = np.logical_or(pred_d1, pred_d2_estrela).astype(int)

                base = {'classe': classe, 'arquivo': nome, 'n_treino': len(X_train), 'n_bal': n_bal}
                resultados['d1'].append({**base, **metricas(y_test, pred_d1)})
                resultados['d2'].append({**base, **metricas(y_test, pred_d2)})
                resultados['d2_estrela'].append({**base, **metricas(y_test, pred_d2_estrela)})
                resultados['d3'].append({**base, **metricas(y_test, pred_d3)})

                print(f"   {nome}: D1={resultados['d1'][-1]['f1']:.3f} | "
                      f"D2={resultados['d2'][-1]['f1']:.3f} | "
                      f"D2*={resultados['d2_estrela'][-1]['f1']:.3f} | "
                      f"D3={resultados['d3'][-1]['f1']:.3f} | n_bal={n_bal}")

            except Exception as e:
                print(f"   {nome}: ERRO — {e}")
                continue

    print("\n" + "=" * 60)
    print("RESULTADO FINAL — VAE (D1 / D2 / D2* / D3)")
    print("=" * 60)
    print(f"Descartes: {descarte}")

    for label, dados in [('D1 (MSE)', resultados['d1']),
                          ('D2 (Mahalanobis P50)', resultados['d2']),
                          ('D2* (Mahalanobis Chi²)', resultados['d2_estrela']),
                          ('D3 (D1 OR D2*)', resultados['d3'])]:
        if not dados:
            continue
        df_r = pd.DataFrame(dados)
        df_r = df_r[np.isfinite(df_r['f1'])]
        resumo = df_r.groupby('classe').agg(F1_media=('f1', 'mean'), F1_std=('f1', 'std'), rodadas=('f1', 'count'))
        print(f"\n--- {label} ---")
        print(resumo.to_string())
        print(f"Global F1: {df_r['f1'].mean():.4f} ± {df_r['f1'].std():.4f} | rodadas={len(df_r)}")

    for nome_csv, dados in [('d1_mse', resultados['d1']),
                             ('d2_mahal_p50', resultados['d2']),
                             ('d2_estrela_mahal_chi2', resultados['d2_estrela']),
                             ('d3_comb', resultados['d3'])]:
        if dados:
            path = os.path.join(DIRETORIO_BASE, f'resultados_vae_{nome_csv}.csv')
            pd.DataFrame(dados).to_csv(path, index=False)

    print(f"\nCSVs salvos em: {DIRETORIO_BASE}")

    tempos_inferencia = np.array(tempos_inferencia)
    tempos_treino = np.array(tempos_treino)
    print("\n" + "=" * 60)
    print("CUSTO COMPUTACIONAL — D2* (VAE + Mahalanobis)")
    print("=" * 60)
    print(f"Parâmetros treináveis do VAE: {n_params:,}")
    print(f"Tempo de treinamento por poço: {tempos_treino.mean():.2f} s ± {tempos_treino.std():.2f} s "
          f"(n={len(tempos_treino)} poços)")
    print(f"Latência de inferência por janela: {tempos_inferencia.mean():.4f} ms ± {tempos_inferencia.std():.4f} ms")
    print(f"P95: {np.percentile(tempos_inferencia, 95):.4f} ms")
    print(f"Throughput: {1000 / tempos_inferencia.mean():.1f} janelas/segundo")


if __name__ == '__main__':
    main()
