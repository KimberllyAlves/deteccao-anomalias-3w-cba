# Tabelas e testes estatísticos a partir dos CSV dos quatro detectores:
# desempenho global e por classe, Friedman e Wilcoxon com Holm, com o poço
# como unidade (média das instâncias de cada poço), e o Dummy de Vargas como
# referência. Seis classes no resultado principal; a classe 9 vem à parte, sem
# testes.
#
# Uso:
#   python analise_estatistica.py                           # resultado principal
#   python analise_estatistica.py --variante lote_vargas    # pontuação em lote
#
# Com --variante X, lê resultados_<det>_X.csv quando existir e, se não, o
# arquivo principal. Saída em resultados/tabelas_q<q>[_X].md.

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

from config import DIRETORIO_RESULTADOS, CLASSES_PRINCIPAIS, CLASSE_EXTENSAO

DETECTORES = {'Isolation Forest': 'if', 'HBOS': 'hbos', 'ECOD': 'ecod', 'COPOD': 'copod'}
COMPARACOES = [('COPOD', 'Isolation Forest'), ('ECOD', 'Isolation Forest'),
               ('COPOD', 'HBOS'), ('ECOD', 'HBOS'),
               ('HBOS', 'Isolation Forest'), ('COPOD', 'ECOD')]
METRICAS_TESTE = [('f1', 'F1 micro'), ('f1_bin', 'F1 binário'), ('auc', 'AUC-ROC')]
# Dummy de Vargas: rotula tudo como normal (F1 micro 0,5, F1 binário 0, AUC 0,5).
DUMMY = 'Dummy'
COMPARACOES_DUMMY = [(d, DUMMY) for d in DETECTORES]


def _variante():
    if '--variante' in sys.argv:
        return sys.argv[sys.argv.index('--variante') + 1]
    return None


def carregar(variante=None):
    partes = []
    for nome, suf in DETECTORES.items():
        caminho = os.path.join(DIRETORIO_RESULTADOS, f'resultados_{suf}.csv')
        if variante:
            alt = os.path.join(DIRETORIO_RESULTADOS, f'resultados_{suf}_{variante}.csv')
            caminho = alt if os.path.exists(alt) else caminho
        df = pd.read_csv(caminho)
        if 'auc' not in df.columns:
            # CSV antigos sem a coluna auc
            auc = pd.read_csv(os.path.join(
                DIRETORIO_RESULTADOS, f'auc_roc_{variante}.csv' if variante else 'auc_roc.csv'))
            col = 'auc_' + nome.lower().replace(' ', '_')
            df = df.merge(auc[['arquivo', col]].rename(columns={col: 'auc'}), on='arquivo')
        df['detector'] = nome
        partes.append(df)
    base = partes[0][['classe', 'arquivo']].copy()
    dummy = base.assign(f1=0.5, precisao=0.5, recall=0.5, f1_bin=0.0, precisao_bin=0.0,
                        recall_bin=0.0, auc=0.5, degenerada=True, detector=DUMMY)
    df = pd.concat(partes + [dummy], ignore_index=True)
    df['classe'] = df['classe'].astype(str)
    df['poco'] = df['arquivo'].str.split('_').str[0]
    return df


def holm(pvalores):
    p = np.asarray(pvalores, dtype=float)
    ordem = np.argsort(p)
    ajust = np.empty_like(p)
    acumulado = 0.0
    for k, i in enumerate(ordem):
        acumulado = max(acumulado, (len(p) - k) * p[i])
        ajust[i] = min(1.0, acumulado)
    return ajust


def testes_por_poco(df, comparacoes=COMPARACOES):
    """Wilcoxon pareado sobre as medias por poco, Holm dentro de cada metrica."""
    linhas = {f'{a} vs. {b}': {} for a, b in comparacoes}
    for met, rotulo in METRICAS_TESTE:
        tab = df.pivot_table(index='poco', columns='detector', values=met, aggfunc='mean')
        brutos, difs = [], []
        for a, b in comparacoes:
            d = tab[a] - tab[b]
            brutos.append(wilcoxon(tab[a], tab[b]).pvalue if (d != 0).any() else 1.0)
            difs.append(d.mean())
        for (a, b), p, d in zip(comparacoes, holm(brutos), difs):
            linhas[f'{a} vs. {b}'][rotulo] = (p, d)
    return linhas, df['poco'].nunique()


def fmt(m, s):
    return f"{m:.4f} ± {s:.3f}".replace('.', ',')


def tabela_global(df):
    out = ["| Detector | F1 micro | F1 binário | Precisão | Sensibilidade | AUC-ROC | Degeneradas |",
           "|---|---|---|---|---|---|---|"]
    for nome in list(DETECTORES) + [DUMMY]:
        g = df[df.detector == nome]
        out.append(f"| {nome} | " + " | ".join(
            fmt(g[c].mean(), g[c].std()) for c in
            ['f1', 'f1_bin', 'precisao_bin', 'recall_bin', 'auc']) +
            f" | {int(g.degenerada.sum())}/{len(g)} |")
    return out


def tabela_testes(df, comparacoes):
    linhas, _ = testes_por_poco(df, comparacoes)
    out = ["| Comparação | " + " | ".join(r for _, r in METRICAS_TESTE) + " |",
           "|---|" + "---|" * len(METRICAS_TESTE)]
    for comp, vals in linhas.items():
        cel = []
        for _, r in METRICAS_TESTE:
            p, d = vals[r]
            cel.append(f"{p:.3f}{' *' if p < 0.05 else ''} ({d:+.3f})".replace('.', ','))
        out.append(f"| {comp} | " + " | ".join(cel) + " |")
    return out


def tabela_friedman(df):
    """Friedman entre os quatro detectores, por poco, como em Vargas (2019)."""
    out = ["| Métrica | Estatística | p |", "|---|---|---|"]
    for met, rotulo in METRICAS_TESTE:
        tab = df[df.detector != DUMMY].pivot_table(
            index='poco', columns='detector', values=met, aggfunc='mean')
        r = friedmanchisquare(*[tab[d] for d in DETECTORES])
        out.append(f"| {rotulo} | {r.statistic:.2f} | {r.pvalue:.4f} |".replace('.', ','))
    return out


def tabela_por_classe(df):
    out = ["| Classe | n | " + " | ".join(DETECTORES) + " |",
           "|---|---|" + "---|" * len(DETECTORES)]
    por_classe = df.pivot_table(index='classe', columns='detector', values='f1', aggfunc='mean')
    n_classe = df[df.detector == DUMMY].groupby('classe').size()
    for c in por_classe.index:
        out.append(f"| {c} | {n_classe[c]} | " + " | ".join(
            f"{por_classe.loc[c, d]:.4f}".replace('.', ',') for d in DETECTORES) + " |")
    return out


def secao(df, titulo, testes=True):
    out = [f"## {titulo}", "",
           f"{df['arquivo'].nunique()} instâncias de {df['poco'].nunique()} poços.", "",
           "### Desempenho global (média ± desvio entre instâncias)", ""] + tabela_global(df)
    if testes:
        n = df['poco'].nunique()
        out += ["", f"### Friedman entre os quatro detectores ({n} poços)", ""] + tabela_friedman(df)
        out += ["", f"### Wilcoxon pareado entre detectores, poço como unidade ({n} poços), Holm", "",
                "Valor de p corrigido; entre parênteses, diferença média por poço (A menos B). "
                "Asterisco: p < 0,05.", ""] + tabela_testes(df, COMPARACOES)
        out += ["", "### Cada detector contra o Dummy, Holm (controle, como em Vargas)", ""] \
            + tabela_testes(df, COMPARACOES_DUMMY)
    out += ["", "### F1 micro por classe", ""] + tabela_por_classe(df)
    return out


def main():
    variante = _variante()
    df = carregar(variante)
    q = df.loc[df.detector != DUMMY, 'q'].iloc[0] if 'q' in df.columns else None
    q_txt = f"{q:g}" if q is not None and not pd.isna(q) else "?"

    out = [f"# Tabelas, q = {q_txt}%" + (f" (variante {variante})" if variante else ""), "",
           "Gerado por `analise_estatistica.py`. q = 0 é a regra de Vargas: limiar no "
           "extremo dos escores de treino.", ""]
    out += secao(df[df.classe.isin(CLASSES_PRINCIPAIS)],
                 "Resultado principal: seis classes do Benchmark 2 (" + ", ".join(CLASSES_PRINCIPAIS) + ")")
    ext = df[df.classe == CLASSE_EXTENSAO]
    if len(ext):
        out += [""] + secao(ext, f"Extensão: classe {CLASSE_EXTENSAO}, posterior ao benchmark",
                            testes=False)

    texto = "\n".join(out) + "\n"
    destino = os.path.join(DIRETORIO_RESULTADOS,
                           f"tabelas_q{q_txt}{'_' + variante if variante else ''}.md")
    with open(destino, 'w', encoding='utf-8') as f:
        f.write(texto)
    print(texto)
    print(f"Salvo em: {destino}")


if __name__ == '__main__':
    main()
