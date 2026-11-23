# Cache dos atributos por poço (.npz). O pré-processamento é o mesmo para os
# quatro detectores e é a parte lenta; feito uma vez, cada rodada leva segundos.
#
# Uso:
#   python cache_features.py              gera o cache (pula o que já existe)
#   python cache_features.py --forcar     regenera tudo
#   python cache_features.py --verificar  confere igualdade bit a bit

import os
import sys
import glob
import json
import time
import hashlib

import numpy as np
import pandas as pd

from config import (
    VARIAVEIS, DIRETORIO_BASE, DIRETORIO_RESULTADOS, CLASSES_BENCHMARK,
    PASSO_TREINO, MAX_AMOSTRAS_TEST, MIN_NORMAL, MIN_ANOMALIA,
    PASSO_TESTE_POR_CLASSE, TAMANHO_JANELA, LIMPAR_SENTINELAS, LIMITE_SENTINELA,
    PROTOCOLO_VARGAS,
)
from preprocessing import (normalizar_rotulos, extrair_estatisticas,
                           preparar_janelas_do_poco, limpar_sentinelas)
import protocolo_vargas

# Um subdiretório por variante do protocolo, nomeado pela impressão digital.
DIR_RAIZ_CACHE = os.path.join(DIRETORIO_RESULTADOS, 'cache_features')


def _dir_cache():
    return os.path.join(DIR_RAIZ_CACHE, impressao_protocolo()[0])


def _arq_manifesto():
    return os.path.join(_dir_cache(), 'manifesto.json')


def impressao_protocolo():
    """
    Identifica a configuracao que gerou o cache. Se qualquer parametro do
    protocolo mudar, a impressao muda e o cache e rejeitado em vez de ser
    usado silenciosamente com features desatualizadas.
    """
    corpo = {
        'variaveis': list(VARIAVEIS),
        'tamanho_janela': TAMANHO_JANELA,
        'passo_treino': PASSO_TREINO,
        'passo_teste_por_classe': PASSO_TESTE_POR_CLASSE,
        'max_amostras_teste': MAX_AMOSTRAS_TEST,
        'min_normal': MIN_NORMAL,
        'min_anomalia': MIN_ANOMALIA,
        'estatisticas': ['mediana', 'media', 'std', 'var', 'max', 'min'],
        'split_treino': 0.6,
        'filtro_vargas': {'max_frac_ausentes': 0.10, 'min_std': 1.0},
        'limpar_sentinelas': bool(LIMPAR_SENTINELAS),
        'limite_sentinela': float(LIMITE_SENTINELA) if LIMPAR_SENTINELAS else None,
        # Codigo do pre-processamento: qualquer alteracao invalida o cache.
        'hash_preprocessing': hashlib.sha256(
            open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'preprocessing.py'), 'rb').read()).hexdigest()[:16],
    }
    if PROTOCOLO_VARGAS:
        # Protocolo de Vargas: impressão própria, com o hash de protocolo_vargas.py.
        pv = protocolo_vargas
        corpo = {
            'protocolo': 'vargas',
            'variaveis': list(VARIAVEIS),
            'tamanho_amostra': pv.TAMANHO_AMOSTRA,
            'max_amostras_por_periodo': pv.MAX_AMOSTRAS_POR_PERIODO,
            'fracao_treino': pv.FRACAO_TREINO,
            'min_normal': MIN_NORMAL,
            'filtro': {'max_frac_ausentes': pv.MAX_FRAC_AUSENTES, 'min_std': pv.MIN_STD},
            'estatisticas': ['mediana', 'media', 'std', 'var', 'max', 'min'],
            'limpar_sentinelas': bool(LIMPAR_SENTINELAS),
            'limite_sentinela': float(LIMITE_SENTINELA) if LIMPAR_SENTINELAS else None,
            'hash_protocolo_vargas': hashlib.sha256(
                open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'protocolo_vargas.py'), 'rb').read()).hexdigest()[:16],
        }
    chave = hashlib.sha256(json.dumps(corpo, sort_keys=True).encode()).hexdigest()[:16]
    return chave, corpo


def _features_do_poco(caminho, classe):
    """Devolve (X_train, X_neg, X_pos, n_bal) ou (None, motivo)."""
    df = pd.read_parquet(caminho)
    for col in VARIAVEIS:
        if col not in df.columns:
            df[col] = np.nan
    if LIMPAR_SENTINELAS:
        df, n_limpas = limpar_sentinelas(df, VARIAVEIS, LIMITE_SENTINELA)
        if n_limpas:
            print(f"      [A2] {n_limpas} leituras > {LIMITE_SENTINELA:.0e} "
                  f"tratadas como ausentes", flush=True)
    if PROTOCOLO_VARGAS:
        rotulos = df['class'].astype('float64').values
        if (rotulos == 0).sum() < MIN_NORMAL:
            return None, 'normal_curto'
        res = protocolo_vargas.preparar_instancia(df[VARIAVEIS].values, rotulos, int(classe))
        if res is None:
            return None, 'janelas_insuf'
        X_train, X_neg, X_pos, _ = res
        return (X_train, X_neg, X_pos, len(X_neg)), None

    df['class'] = normalizar_rotulos(df['class'].fillna(-1))

    if (df['class'] == 0).sum() < MIN_NORMAL:
        return None, 'normal_curto'
    if ((df['class'] != 0) & (df['class'] != -1)).sum() < MIN_ANOMALIA:
        return None, 'anomalia_curta'

    partes = preparar_janelas_do_poco(
        df, VARIAVEIS, PASSO_TREINO, PASSO_TESTE_POR_CLASSE[classe], MAX_AMOSTRAS_TEST)
    if partes is None:
        return None, 'janelas_insuf'

    janelas_treino, janelas_neg, janelas_pos, n_bal = partes
    X_train = np.array([extrair_estatisticas(j) for j in janelas_treino])
    X_neg = np.array([extrair_estatisticas(j) for j in janelas_neg])
    X_pos = np.array([extrair_estatisticas(j) for j in janelas_pos])
    return (X_train, X_neg, X_pos, n_bal), None


def gerar(forcar=False):
    chave, corpo = impressao_protocolo()
    os.makedirs(_dir_cache(), exist_ok=True)

    # Impressao diferente invalida tudo o que estava gravado.
    if os.path.exists(_arq_manifesto()):
        antigo = json.load(open(_arq_manifesto()))
        if antigo.get('chave') != chave and not forcar:
            print(f"Cache existente foi gerado com outro protocolo "
                  f"({antigo.get('chave')} != {chave}).")
            print("Rode com --forcar para regenerar.")
            return
    if forcar:
        for f in glob.glob(os.path.join(_dir_cache(), '*', '*.npz')):
            os.remove(f)

    t0 = time.perf_counter()
    entradas, descartes = [], {}
    for classe in CLASSES_BENCHMARK:
        os.makedirs(os.path.join(_dir_cache(), classe), exist_ok=True)
        arquivos = sorted(glob.glob(os.path.join(DIRETORIO_BASE, classe, "WELL*.parquet")))
        print(f">>> Classe {classe}: {len(arquivos)} arquivo(s)", flush=True)
        for caminho in arquivos:
            nome = os.path.basename(caminho)
            destino = os.path.join(_dir_cache(), classe, nome.replace('.parquet', '.npz'))
            if os.path.exists(destino) and not forcar:
                entradas.append({'classe': classe, 'arquivo': nome, 'npz': destino})
                continue
            try:
                res, motivo = _features_do_poco(caminho, classe)
            except Exception as e:
                descartes[nome] = f'erro: {e}'
                print(f"   {nome}: ERRO — {e}", flush=True)
                continue
            if res is None:
                descartes[nome] = motivo
                print(f"   {nome}: descartado ({motivo})", flush=True)
                continue
            X_train, X_neg, X_pos, n_bal = res
            np.savez_compressed(destino, X_train=X_train, X_neg=X_neg,
                                X_pos=X_pos, n_bal=n_bal)
            entradas.append({'classe': classe, 'arquivo': nome, 'npz': destino})
            print(f"   {nome}: {X_train.shape[0]} treino, n_bal={n_bal}", flush=True)

    json.dump({'chave': chave, 'protocolo': corpo, 'entradas': entradas,
               'descartes': descartes, 'gerado_em': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(_arq_manifesto(), 'w'), indent=2)

    tam = sum(os.path.getsize(e['npz']) for e in entradas) / 1e6
    print(f"\nCache: {len(entradas)} poços | {tam:.1f} MB | "
          f"{(time.perf_counter()-t0)/60:.1f} min")
    from collections import Counter
    print(f"Descartes: {dict(Counter(descartes.values()))}")


def descartes_do_cache():
    """Contagem dos descartes registrados quando o cache foi gerado."""
    from collections import Counter
    man = json.load(open(_arq_manifesto()))
    return dict(Counter(man['descartes'].values()))


def carregar_cache():
    """Itera o cache validando a impressao do protocolo."""
    if not os.path.exists(_arq_manifesto()):
        raise RuntimeError("Cache ausente. Rode: python cache_features.py")
    man = json.load(open(_arq_manifesto()))
    chave, _ = impressao_protocolo()
    if man['chave'] != chave:
        raise RuntimeError(
            f"Cache obsoleto: gerado com protocolo {man['chave']}, "
            f"atual {chave}. Rode: python cache_features.py --forcar")
    for e in man['entradas']:
        d = np.load(e['npz'])
        yield {'classe': e['classe'], 'arquivo': e['arquivo'],
               'X_train': d['X_train'], 'X_neg': d['X_neg'],
               'X_pos': d['X_pos'], 'n_bal': int(d['n_bal'])}


def verificar(n_pocos=3):
    """Recalcula do Parquet bruto e exige igualdade bit a bit com o cache."""
    man = json.load(open(_arq_manifesto()))
    alvos = man['entradas'][:1] + man['entradas'][len(man['entradas'])//2:
                                                  len(man['entradas'])//2+1] \
        + man['entradas'][-1:]
    alvos = alvos[:n_pocos]
    ok = True
    for e in alvos:
        caminho = os.path.join(DIRETORIO_BASE, e['classe'], e['arquivo'])
        res, _ = _features_do_poco(caminho, e['classe'])
        Xtr, Xn, Xp, nb = res
        d = np.load(e['npz'])
        iguais = (np.array_equal(Xtr, d['X_train']) and
                  np.array_equal(Xn, d['X_neg']) and
                  np.array_equal(Xp, d['X_pos']) and nb == int(d['n_bal']))
        print(f"   {e['arquivo']}: {'IDENTICO' if iguais else 'DIVERGENTE'} "
              f"(treino {Xtr.shape})")
        ok = ok and iguais
    print(f"\nVerificação: {'OK, cache confiável' if ok else 'FALHOU'}")
    return ok


if __name__ == '__main__':
    if '--verificar' in sys.argv:
        verificar()
    else:
        gerar(forcar='--forcar' in sys.argv)
