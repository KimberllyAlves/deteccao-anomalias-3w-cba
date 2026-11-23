# Isolation Forest (Liu, Ting e Zhou, 2008), linha de base.

from sklearn.ensemble import IsolationForest

from benchmark_pyod import executar_benchmark
from config import SEMENTE_IF


def main():
    executar_benchmark(
        nome_detector='Isolation Forest',
        construir_detector=lambda: IsolationForest(random_state=SEMENTE_IF),
        sufixo_csv='if',
        metodo='score_samples',
        sinal='menor',
    )


if __name__ == '__main__':
    main()
