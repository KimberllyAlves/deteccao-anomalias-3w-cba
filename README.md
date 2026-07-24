# Detecção de anomalias em poços de petróleo offshore (3W)

Código do benchmark usado no artigo "Detecção de Anomalias em Poços de
Petróleo Offshore via Espaço Latente de Variational Autoencoders e
Distância de Mahalanobis" (CBA 2026), utilizando o [dataset 3W](https://github.com/petrobras/3W) v2.0.

Dois detectores são comparados:

- **Isolation Forest**, linha de base (`isolation_forest.py`)
- **VAE**, treinado por poço em regime *one-class*, avaliado com quatro
  critérios de decisão (`vae_benchmark.py`):
  - `D1`: erro de reconstrução (MSE) acima do percentil 50 do treino
  - `D2`: distância de Mahalanobis no espaço latente acima do percentil 50
  - `D2*`: distância de Mahalanobis acima de `sqrt(chi2(0.95, df=8))` —
    critério proposto no artigo
  - `D3`: `D1 OR D2*`

## Estrutura

```
config.py           constantes e hiperparâmetros compartilhados
preprocessing.py     rótulos, janelas, extração de estatísticas
vae.py                arquitetura do VAE, treino e inferência
isolation_forest.py   script principal — baseline
vae_benchmark.py      script principal — VAE (D1/D2/D2*/D3)
```

## Uso

Baixe o [3W v2.0](https://github.com/petrobras/3W) e aponte
`DIRETORIO_BASE`, em `config.py`, para a pasta com as subpastas `1/`, `2/`,
`5/`, `6/`, `7/`, `8/`, `9/` (uma por classe de anomalia).

```
pip install -r requirements.txt
python isolation_forest.py
python vae_benchmark.py
```

Cada script imprime o F1 médio por classe e global, salva um CSV por
detector em `DIRETORIO_BASE` e, ao final, um resumo de custo computacional
(latência de inferência, throughput e, no caso do VAE, tempo de treino e
número de parâmetros).

## Reprodutibilidade

O VAE fixa a seed (`config.SEED`, padrão 42) para `random`, NumPy e PyTorch
antes do treino, incluindo modo determinístico do cuDNN quando há GPU
disponível. O Isolation Forest usa `random_state=42`.
