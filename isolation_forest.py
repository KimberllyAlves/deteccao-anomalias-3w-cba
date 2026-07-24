# Isolation Forest — linha de base do benchmark de detecção de anomalias.


import os
import glob
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

from config import (
    VARIAVEIS, DIRETORIO_BASE, CLASSES_BENCHMARK, PASSO_TREINO,
    MAX_AMOSTRAS_TEST, MIN_NORMAL, MIN_ANOMALIA, PASSO_TESTE_POR_CLASSE,
)
from preprocessing import normalizar_rotulos, extrair_estatisticas, preparar_janelas_do_poco


def main():
    resultados = []
    tempos_inferencia = []
    descarte = {'normal_curto': 0, 'anomalia_curta': 0, 'janelas_insuf': 0}

    print("Isolation Forest")
    print(f"Variáveis: {VARIAVEIS}")
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

                clf = IsolationForest(random_state=42)
                clf.fit(X_train)

                for i in range(X_test.shape[0]):
                    t0 = time.perf_counter()
                    clf.score_samples(X_test[i:i + 1])
                    tempos_inferencia.append((time.perf_counter() - t0) * 1000)

                scores_treino = clf.score_samples(X_train)
                scores_neg = clf.score_samples(X_neg)
                scores_pos = clf.score_samples(X_pos)
                scores_teste = clf.score_samples(X_test)

                limiar = np.percentile(scores_treino, 50)
                y_pred = (scores_teste < limiar).astype(int)

                f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
                prec = precision_score(y_test, y_pred, average='micro', zero_division=0)
                rec = recall_score(y_test, y_pred, average='micro', zero_division=0)
                sep = float(scores_neg.mean() - scores_pos.mean())

                resultados.append({
                    'classe': classe, 'arquivo': nome,
                    'f1': f1, 'precisao': prec, 'recall': rec,
                    'n_treino': len(X_train), 'n_bal': n_bal, 'sep_scores': sep,
                })
                print(f"   {nome}: F1={f1:.3f} | sep={sep:+.3f} | n_bal={n_bal}")

            except Exception as e:
                print(f"   {nome}: ERRO — {e}")
                continue

    print("\n" + "=" * 60)
    print("RESULTADO FINAL — ISOLATION FOREST")
    print("=" * 60)
    print(f"Descartes: {descarte}")

    if not resultados:
        print("Nenhuma rodada válida concluída.")
        return

    df_res = pd.DataFrame(resultados)
    resumo = df_res.groupby('classe').agg(
        F1_media=('f1', 'mean'), F1_std=('f1', 'std'),
        n_treino=('n_treino', 'mean'), n_bal=('n_bal', 'mean'),
        sep_media=('sep_scores', 'mean'), rodadas=('f1', 'count'),
    )
    print("\nPor classe (média ± desvio padrão de F1):")
    print(resumo.to_string())
    print(f"\nGlobal — F1: {df_res['f1'].mean():.4f} ± {df_res['f1'].std():.4f}")
    print(f"Total de rodadas: {len(df_res)}")

    csv_path = os.path.join(DIRETORIO_BASE, 'resultados_if.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")

    tempos = np.array(tempos_inferencia)
    print("\n" + "=" * 60)
    print("CUSTO COMPUTACIONAL — Isolation Forest")
    print("=" * 60)
    print(f"Latência de inferência por janela: {tempos.mean():.4f} ms ± {tempos.std():.4f} ms")
    print(f"P95: {np.percentile(tempos, 95):.4f} ms")
    print(f"Throughput: {1000 / tempos.mean():.1f} janelas/segundo")


if __name__ == '__main__':
    main()
