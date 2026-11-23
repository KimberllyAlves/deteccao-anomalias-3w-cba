# Laço do benchmark, comum aos quatro detectores, para que todos usem a mesma
# regra de decisão.
#
# Convenção de sinal (q = config.TAXA_ALARME):
#   sklearn (score_samples,     MENOR = anômalo): limiar = percentil(q)
#   PyOD    (decision_function, MAIOR = anômalo): limiar = percentil(100 - q)

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from config import DIRETORIO_RESULTADOS, TAXA_ALARME, PONTUACAO_POR_JANELA
from cache_features import carregar_cache, descartes_do_cache


def _q_da_linha_de_comando(padrao=TAXA_ALARME):
    """Permite sobrescrever a taxa de alarme: python hbos.py --q 50"""
    if '--q' in sys.argv:
        return float(sys.argv[sys.argv.index('--q') + 1])
    return padrao


def executar_benchmark(nome_detector, construir_detector, sufixo_csv,
                       metodo='decision_function', sinal='maior', q=None):
    """
    nome_detector      rotulo usado nos cabecalhos impressos
    construir_detector funcao sem argumentos que devolve um estimador novo
    sufixo_csv         sufixo do arquivo (resultados_<sufixo>.csv)
    metodo             'decision_function' (PyOD) ou 'score_samples' (sklearn)
    sinal              'maior' ou 'menor' indica anomalia
    q                  taxa de alarme no treino, em %; None usa config/CLI
    """
    q = _q_da_linha_de_comando() if q is None else q
    percentil = q if sinal == 'menor' else 100 - q

    print(f"{nome_detector}")
    print(f"Regra de decisao: taxa de alarme no treino q={q}% "
          f"(percentil {percentil} dos escores, {sinal} = anomalo)")
    print(f"Pontuacao do teste: {'uma janela por vez' if PONTUACAO_POR_JANELA else 'teste inteiro de uma vez'}\n")

    resultados, tempos_inferencia, tempos_treino, falhas = [], [], [], []

    for reg in carregar_cache():
        classe, nome, n_bal = reg['classe'], reg['arquivo'], reg['n_bal']
        X_train, X_neg, X_pos = reg['X_train'], reg['X_neg'], reg['X_pos']
        X_test = np.vstack([X_neg, X_pos])
        y_test = np.array([0] * n_bal + [1] * n_bal)

        try:
            clf = construir_detector()
            t0 = time.perf_counter()
            clf.fit(X_train)
            tempos_treino.append((time.perf_counter() - t0) * 1000)

            escorar = getattr(clf, metodo)
            scores_janela = []
            for i in range(X_test.shape[0]):
                t0 = time.perf_counter()
                scores_janela.append(escorar(X_test[i:i + 1])[0])
                tempos_inferencia.append((time.perf_counter() - t0) * 1000)

            scores_treino = escorar(X_train)
            if PONTUACAO_POR_JANELA:
                scores_teste = np.array(scores_janela)
            else:
                scores_teste = escorar(X_test)
            scores_neg, scores_pos = scores_teste[:n_bal], scores_teste[n_bal:]
        except Exception as e:
            falhas.append((classe, nome, str(e)[:90]))
            print(f"   classe {classe} {nome}: ERRO — {e}")
            continue

        limiar = np.percentile(scores_treino, percentil)
        y_pred = ((scores_teste < limiar) if sinal == 'menor'
                  else (scores_teste > limiar)).astype(int)

        # positiva quando as janelas anômalas pontuam acima das normais
        sep = float(scores_pos.mean() - scores_neg.mean())
        if sinal == 'menor':
            sep = -sep

        resultados.append({
            'classe': classe, 'arquivo': nome,
            'f1': f1_score(y_test, y_pred, average='micro', zero_division=0),
            'precisao': precision_score(y_test, y_pred, average='micro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='micro', zero_division=0),
            'f1_bin': f1_score(y_test, y_pred, average='binary', zero_division=0),
            'precisao_bin': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall_bin': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'degenerada': bool(y_pred.all() or not y_pred.any()),
            # no sklearn, MENOR = anômalo; o sinal é invertido para o AUC
            'auc': roc_auc_score(y_test, -scores_teste if sinal == 'menor' else scores_teste),
            'n_treino': len(X_train), 'n_bal': n_bal, 'sep_scores': sep, 'q': q,
        })

    print("\n" + "=" * 72)
    print(f"RESULTADO FINAL — {nome_detector.upper()} (q={q}%)")
    print("=" * 72)
    print(f"Descartes do protocolo: {descartes_do_cache()}")
    if falhas:
        print(f"Falhas do detector: {len(falhas)}")
        for c, n, e in falhas:
            print(f"   classe {c} {n}: {e}")

    if not resultados:
        print("Nenhuma rodada valida concluida.")
        return None

    df_res = pd.DataFrame(resultados)
    resumo = df_res.groupby('classe').agg(
        F1_media=('f1', 'mean'), F1_std=('f1', 'std'),
        F1bin_media=('f1_bin', 'mean'), F1bin_std=('f1_bin', 'std'),
        degeneradas=('degenerada', 'sum'), rodadas=('f1', 'count'),
    )
    print("\nPor classe:")
    print(resumo.to_string())
    print(f"\nGlobal — F1 (micro):   {df_res['f1'].mean():.4f} ± {df_res['f1'].std():.4f}")
    print(f"Global — F1 (binario): {df_res['f1_bin'].mean():.4f} ± {df_res['f1_bin'].std():.4f}")
    print(f"Global — AUC-ROC:      {df_res['auc'].mean():.4f} ± {df_res['auc'].std():.4f}")
    print(f"Rodadas: {len(df_res)} | degeneradas: {df_res['degenerada'].sum()}")

    os.makedirs(DIRETORIO_RESULTADOS, exist_ok=True)
    csv_path = os.path.join(DIRETORIO_RESULTADOS, f'resultados_{sufixo_csv}.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")

    tempos, treinos = np.array(tempos_inferencia), np.array(tempos_treino)
    print("\n" + "=" * 72)
    print(f"CUSTO COMPUTACIONAL — {nome_detector}")
    print("=" * 72)
    print(f"Latencia de inferencia por janela: {tempos.mean():.4f} ms ± {tempos.std():.4f} ms")
    print(f"P95: {np.percentile(tempos, 95):.4f} ms")
    print(f"Throughput: {1000 / tempos.mean():.1f} janelas/segundo")
    print(f"Tempo de treino por poco: {treinos.mean():.2f} ms ± {treinos.std():.2f} ms")
    return df_res
