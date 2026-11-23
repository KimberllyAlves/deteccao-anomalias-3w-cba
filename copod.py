# COPOD (Li et al., 2020): caudas da distribuição cumulativa empírica de cada
# variável, como o ECOD, combinadas de outro modo. Não usa a dependência entre
# variáveis. Sem hiperparâmetros.

from pyod.models.copod import COPOD

from benchmark_pyod import executar_benchmark


def main():
    executar_benchmark(
        nome_detector='COPOD',
        construir_detector=lambda: COPOD(),
        sufixo_csv='copod',
    )


if __name__ == '__main__':
    main()
