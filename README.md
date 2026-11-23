# Detecção de anomalias em poços de petróleo offshore com HBOS, ECOD e COPOD (3W)

Código do Trabalho de Conclusão de Curso *Detecção de Anomalias em Poços de
Petróleo Offshore: Avaliação de Algoritmos de Detecção de Novidades no Dataset
3W*, de Mateus Cervino Bastos (Engenharia Elétrica, UFBA, 2026), com orientação
do Prof. Dr. Eduardo Simas e coorientação de Kimberlly Alves Santos.

O trabalho avalia três métodos estatísticos de detecção de anomalias, o
**HBOS**, o **ECOD** e o **COPOD**, e os compara com o **Isolation Forest**,
seguindo o Benchmark 2 do [Dataset 3W](https://github.com/petrobras/3W)
(versão 2.0.0), definido por Vargas (2019).

## Origem do código

Este repositório é um *fork* do código do artigo de Kimberlly Alves Santos,
Eduardo F. Simas Filho e Márcio A. F. Martins, *Detecção de anomalias em poços
de petróleo offshore via espaço latente de Variational Autoencoders* (XXVI
Congresso Brasileiro de Automática, 2026). Os arquivos `vae.py` e
`vae_benchmark.py` vêm desse trabalho e **não são usados no TCC**.

Em relação ao original, o TCC acrescenta:

- a implementação do protocolo do Benchmark 2 na forma do código original de
  Vargas (`protocolo_vargas.py`), conferida contra o notebook dele;
- os três detectores estatísticos (HBOS, ECOD e COPOD, da biblioteca PyOD) e um
  laço único de avaliação para os quatro detectores;
- a análise estatística por poço e a medição do custo computacional.

## Estrutura

```
config.py               constantes do protocolo e caminhos
protocolo_vargas.py     seleção das janelas, filtro e padronização (Benchmark 2, Vargas)
preprocessing.py        rótulos e estatísticas (do trabalho original)
cache_features.py       gera e lê os atributos de cada instância
benchmark_pyod.py       laço de avaliação comum aos quatro detectores
isolation_forest.py     Isolation Forest (scikit-learn), linha de base
hbos.py, ecod.py, copod.py   detectores estatísticos (PyOD)
analise_estatistica.py  médias, testes de Friedman e Wilcoxon por poço, Holm
custo_computacional.py  tempo de treino, de uma janela e de um lote
resultados/             resultados por instância usados no TCC
vae.py, vae_benchmark.py   VAE do trabalho original (não usado no TCC)
```

## Protocolo, em resumo

- Instâncias reais das classes 1, 2, 5, 6, 7 e 8 do 3W com pelo menos 20 minutos
  de operação normal: 89 instâncias de 26 poços. A classe 9, posterior ao
  benchmark, é avaliada à parte.
- Uma rodada por instância: 9 janelas normais de 180 s para treino e 12 de teste
  (6 normais e 6 anômalas), cada janela descrita por 6 estatísticas de até 7
  sensores (até 42 atributos).
- Detectores com os valores padrão das bibliotecas, sem ajuste de
  hiperparâmetros; Isolation Forest com semente 1.
- Limiar no maior escore das janelas de treino, a regra do código de Vargas.
- Cada janela de teste é pontuada separadamente. No PyOD, o ECOD e o COPOD
  calculam o escore de uma janela usando também as outras janelas da mesma
  chamada, o que infla os resultados quando o teste é pontuado de uma vez.
- Medidas: F1 micro (principal), F1, precisão e sensibilidade binárias e AUC.
  Testes estatísticos com o poço como unidade.

## Como reproduzir

1. Clone o [3W](https://github.com/petrobras/3W) (versão 2.0.0) ao lado deste
   repositório, ou indique a pasta `dataset` dele na variável de ambiente
   `DIRETORIO_3W`.
2. Instale as dependências (Python 3.12):
   ```
   pip install -r requirements.txt
   ```
3. Rode, nesta ordem:
   ```
   python cache_features.py        # atributos de cada instância (alguns minutos)
   python isolation_forest.py
   python hbos.py
   python ecod.py
   python copod.py
   python analise_estatistica.py   # resultados/tabelas_q0.md
   ```
4. Para o custo computacional, com uma única thread:
   ```
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python custo_computacional.py
   ```

Os arquivos em `resultados/` são os usados no TCC. Os CSV
`resultados_ecod_lote_vargas.csv` e `resultados_copod_lote_vargas.csv` vêm da
mesma avaliação com o teste pontuado de uma vez (`PONTUACAO_POR_JANELA = False`
em `config.py`) e servem apenas para medir o efeito descrito acima.

## Licença

MIT, mantida do trabalho original (ver `LICENSE.txt`). O arquivo
`protocolo_vargas.py` adapta o código do Benchmark 2 de Ricardo Emanuel Vaz
Vargas (repositório `ricardovvargas/3w_dataset`, licença MIT).
