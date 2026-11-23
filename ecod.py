# ECOD (Li et al., 2022): caudas da distribuição cumulativa empírica de cada
# variável. Sem hiperparâmetros.

from pyod.models.ecod import ECOD

from benchmark_pyod import executar_benchmark


def main():
    executar_benchmark(
        nome_detector='ECOD',
        construir_detector=lambda: ECOD(),
        sufixo_csv='ecod',
    )


if __name__ == '__main__':
    main()
