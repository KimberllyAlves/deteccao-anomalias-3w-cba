#Configurações compartilhadas do pipeline de detecção de anomalias no 3W.

import random

import numpy as np
import torch

# Sensores usados. T-JUS-CKGL fica de fora porque está ausente no
# config.ini do 3W v2.0, mesmo constando na documentação original do dataset.
VARIAVEIS = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'QGL']
N_VARS = len(VARIAVEIS)
N_FEATURES = N_VARS * 6  # mediana, média, desvio padrão, variância, máximo, mínimo

DIRETORIO_BASE = r'C:\Users\Kimberlly Alves\Desktop\UFBA\Pesquisa\3W-main\dataset'
CLASSES_BENCHMARK = ['1', '2', '5', '6', '7', '8', '9']

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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fixar_sementes(seed=SEED):
    """Fixa a aleatoriedade de random/numpy/torch para tornar os resultados reprodutíveis."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
