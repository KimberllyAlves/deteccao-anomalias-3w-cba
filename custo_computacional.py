# Custo computacional dos quatro detectores, por instância: tempo de treino,
# latência de uma janela isolada (monitoramento em tempo real) e tempo por
# janela num lote de TAMANHO_LOTE janelas (análise de histórico). O lote é
# grande para diluir o custo fixo de cada chamada.
#
# Em uma thread, para comparação justa:
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#       python custo_computacional.py

import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD

from config import DIRETORIO_RESULTADOS, SEMENTE_IF
from cache_features import carregar_cache

DETECTORES = {
    'Isolation Forest': (lambda: IsolationForest(random_state=SEMENTE_IF), 'score_samples'),
    'HBOS':             (lambda: HBOS(),  'decision_function'),
    'ECOD':             (lambda: ECOD(),  'decision_function'),
    'COPOD':            (lambda: COPOD(), 'decision_function'),
}
REPETICOES_LOTE = 3
TAMANHO_LOTE = 10_000
SEMENTE_LOTE = 0


def medir(construir, metodo, X_train, X_test, rng):
    clf = construir()
    t0 = time.perf_counter()
    clf.fit(X_train)
    fit_ms = (time.perf_counter() - t0) * 1000

    escorar = getattr(clf, metodo)
    escorar(X_test[:1])  # aquecimento, nao medido

    lat = []
    for i in range(X_test.shape[0]):
        t0 = time.perf_counter()
        escorar(X_test[i:i + 1])
        lat.append((time.perf_counter() - t0) * 1000)

    X_lote = X_test[rng.integers(0, X_test.shape[0], TAMANHO_LOTE)]
    lote = []
    for _ in range(REPETICOES_LOTE):
        t0 = time.perf_counter()
        escorar(X_lote)
        lote.append((time.perf_counter() - t0) * 1000)
    lote_por_janela = min(lote) / TAMANHO_LOTE

    return fit_ms, np.array(lat), lote_por_janela


def main():
    linhas = []
    rng = np.random.default_rng(SEMENTE_LOTE)
    for reg in carregar_cache():
        X_train = reg['X_train']
        X_test = np.vstack([reg['X_neg'], reg['X_pos']])
        for nome, (construir, metodo) in DETECTORES.items():
            fit_ms, lat, lote = medir(construir, metodo, X_train, X_test, rng)
            linhas.append({
                'detector': nome, 'classe': reg['classe'], 'arquivo': reg['arquivo'],
                'n_treino': X_train.shape[0], 'n_teste': X_test.shape[0],
                'fit_ms': fit_ms,
                'lat_mediana_ms': float(np.median(lat)),
                'lat_p95_ms': float(np.percentile(lat, 95)),
                'lote_ms_por_janela': lote,
            })
        print(f"   {reg['arquivo']}: ok", flush=True)

    df = pd.DataFrame(linhas)
    os.makedirs(DIRETORIO_RESULTADOS, exist_ok=True)
    destino = os.path.join(DIRETORIO_RESULTADOS, 'custo_computacional.csv')
    df.to_csv(destino, index=False)

    print("\n" + "=" * 92)
    print("CUSTO COMPUTACIONAL (medianas entre as instâncias avaliadas)")
    print("=" * 92)
    print(f"{'Detector':18s} {'treino (ms)':>12s} {'janela isolada (ms)':>20s} "
          f"{'P95 (ms)':>10s} {'lote (ms/janela)':>20s} {'rho(lat, n_treino)':>19s}")
    for nome, g in df.groupby('detector', sort=False):
        rho = spearmanr(g['lat_mediana_ms'], g['n_treino']).correlation
        print(f"{nome:18s} {g.fit_ms.median():12.2f} {g.lat_mediana_ms.median():20.4f} "
              f"{g.lat_p95_ms.median():10.4f} {g.lote_ms_por_janela.median():20.4f} {rho:19.2f}")
    print(f"\nSalvo em: {destino}")


if __name__ == '__main__':
    main()
