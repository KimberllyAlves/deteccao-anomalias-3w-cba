# HBOS (Goldstein e Dengel, 2012): densidade de cada variável por histograma.
# Usa o número de faixas padrão do PyOD (10).

from pyod.models.hbos import HBOS

from benchmark_pyod import executar_benchmark


def main():
    executar_benchmark(
        nome_detector='HBOS',
        construir_detector=lambda: HBOS(),
        sufixo_csv='hbos',
    )


if __name__ == '__main__':
    main()
