#Configurações compartilhadas do pipeline de detecção de anomalias no 3W.

import os
import random
from pathlib import Path

import numpy as np

# torch só é usado pelo VAE, fora deste TCC; o import é opcional.
try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None
    _TORCH_OK = False

# Sete sensores: T-JUS-CKGL não existe no 3W v2.0.0.
VARIAVEIS = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'QGL']
N_VARS = len(VARIAVEIS)
N_FEATURES = N_VARS * 6  # mediana, média, desvio padrão, variância, máximo, mínimo

_RAIZ = Path(__file__).resolve().parent
# Pasta "dataset" do 3W v2.0.0. Por padrão, o clone do 3W ao lado deste
# repositório; DIRETORIO_3W aponta para outro lugar.
DIRETORIO_BASE = os.environ.get('DIRETORIO_3W', str(_RAIZ.parent / '3W' / 'dataset'))
# Saídas: CSV, tabelas e cache.
DIRETORIO_RESULTADOS = os.environ.get('DIRETORIO_RESULTADOS', str(_RAIZ / 'resultados'))
CLASSES_BENCHMARK = ['1', '2', '5', '6', '7', '8', '9']
# Resultado principal: as seis classes do Benchmark 2. A classe 9 é extensão,
# separada em analise_estatistica.py.
CLASSES_PRINCIPAIS = ['1', '2', '5', '6', '7', '8']
CLASSE_EXTENSAO = '9'

# Limiar: percentil dos escores de treino que marca TAXA_ALARME% do treino
# como anômalo. 0 é a regra de Vargas: limiar no extremo do treino, como
# contamination=0 no notebook dele.
TAXA_ALARME = 0

# Semente do Isolation Forest, a do notebook de Vargas.
SEMENTE_IF = 1

# Leituras com |x| > LIMITE_SENTINELA (sensor travado, por exemplo o P-PDG do
# WELL-00006 em -1,18e42) passam a ser ausentes. Sem isso, o HBOS quebra.
LIMPAR_SENTINELAS = True
LIMITE_SENTINELA = 1e30

# Pontua cada janela de teste sozinha. No PyOD, ECOD e COPOD recalculam as
# distribuições com todas as janelas da chamada; pontuar o teste de uma vez
# daria a eles informação das outras janelas.
PONTUACAO_POR_JANELA = True

# True: protocolo de Vargas (protocolo_vargas.py). False: implementação de
# Alves (preprocessing.py).
PROTOCOLO_VARGAS = True

TAMANHO_JANELA = 180
PASSO_TREINO = 10
MAX_AMOSTRAS_TEST = 15
MIN_NORMAL = 1200      # segundos mínimos de operação normal (Vargas, 2019)
MIN_ANOMALIA = 180

# Só a classe 2 (fechamento espúrio de DHSV) tem passo de teste diferente
# do padrão, conforme dataset.ini do 3W.
PASSO_TESTE_POR_CLASSE = {
    '1': 180, '2': 15, '5': 180, '6': 180, '7': 180, '8': 180, '9': 180,
}

HIDDEN_DIM = 24
LATENT_DIM = 8
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 16
PATIENCE = 15
LOGVAR_CLAMP = 4.0
GRAD_CLIP = 1.0
NIVEL_CONFIANCA = 0.95

SEED = 42
DEVICE = torch.device('cuda' if (_TORCH_OK and torch.cuda.is_available()) else 'cpu') if _TORCH_OK else None


def fixar_sementes(seed=SEED):
    """Fixa a aleatoriedade de random/numpy/torch para tornar os resultados reprodutíveis."""
    random.seed(seed)
    np.random.seed(seed)
    if not _TORCH_OK:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
