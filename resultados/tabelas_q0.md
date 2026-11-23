# Tabelas, q = 0%

Gerado por `analise_estatistica.py`. q = 0 é a regra de Vargas: limiar no extremo dos escores de treino.

## Resultado principal: seis classes do Benchmark 2 (1, 2, 5, 6, 7, 8)

89 instâncias de 26 poços.

### Desempenho global (média ± desvio entre instâncias)

| Detector | F1 micro | F1 binário | Precisão | Sensibilidade | AUC-ROC | Degeneradas |
|---|---|---|---|---|---|---|
| Isolation Forest | 0,6648 ± 0,177 | 0,5722 ± 0,301 | 0,6551 ± 0,344 | 0,5674 ± 0,339 | 0,7814 ± 0,200 | 14/89 |
| HBOS | 0,5084 ± 0,036 | 0,6705 ± 0,017 | 0,5053 ± 0,024 | 0,9981 ± 0,018 | 0,8291 ± 0,150 | 83/89 |
| ECOD | 0,7191 ± 0,153 | 0,7431 ± 0,143 | 0,7495 ± 0,197 | 0,8165 ± 0,201 | 0,8519 ± 0,135 | 7/89 |
| COPOD | 0,7341 ± 0,145 | 0,7658 ± 0,136 | 0,7333 ± 0,199 | 0,8596 ± 0,179 | 0,8881 ± 0,115 | 9/89 |
| Dummy | 0,5000 ± 0,000 | 0,0000 ± 0,000 | 0,0000 ± 0,000 | 0,0000 ± 0,000 | 0,5000 ± 0,000 | 89/89 |

### Friedman entre os quatro detectores (26 poços)

| Métrica | Estatística | p |
|---|---|---|
| F1 micro | 41,40 | 0,0000 |
| F1 binário | 29,05 | 0,0000 |
| AUC-ROC | 25,64 | 0,0000 |

### Wilcoxon pareado entre detectores, poço como unidade (26 poços), Holm

Valor de p corrigido; entre parênteses, diferença média por poço (A menos B). Asterisco: p < 0,05.

| Comparação | F1 micro | F1 binário | AUC-ROC |
|---|---|---|---|
| COPOD vs. Isolation Forest | 0,144 (+0,054) | 0,006 * (+0,173) | 0,001 * (+0,093) |
| ECOD vs. Isolation Forest | 0,144 (+0,055) | 0,004 * (+0,166) | 0,030 * (+0,061) |
| COPOD vs. HBOS | 0,000 * (+0,212) | 0,000 * (+0,098) | 0,002 * (+0,051) |
| ECOD vs. HBOS | 0,000 * (+0,213) | 0,000 * (+0,092) | 0,509 (+0,019) |
| HBOS vs. Isolation Forest | 0,001 * (-0,158) | 0,679 (+0,074) | 0,057 (+0,042) |
| COPOD vs. ECOD | 0,528 (-0,001) | 0,679 (+0,006) | 0,005 * (+0,032) |

### Cada detector contra o Dummy, Holm (controle, como em Vargas)

| Comparação | F1 micro | F1 binário | AUC-ROC |
|---|---|---|---|
| Isolation Forest vs. Dummy | 0,001 * (+0,169) | 0,000 * (+0,596) | 0,000 * (+0,305) |
| HBOS vs. Dummy | 0,043 * (+0,011) | 0,000 * (+0,671) | 0,000 * (+0,348) |
| ECOD vs. Dummy | 0,000 * (+0,223) | 0,000 * (+0,762) | 0,000 * (+0,367) |
| COPOD vs. Dummy | 0,000 * (+0,223) | 0,000 * (+0,769) | 0,000 * (+0,398) |

### F1 micro por classe

| Classe | n | Isolation Forest | HBOS | ECOD | COPOD |
|---|---|---|---|---|---|
| 1 | 4 | 0,7500 | 0,5000 | 0,7708 | 0,7500 |
| 2 | 20 | 0,7708 | 0,5125 | 0,7667 | 0,8042 |
| 5 | 9 | 0,6204 | 0,5000 | 0,6296 | 0,6481 |
| 6 | 6 | 0,5694 | 0,5694 | 0,7639 | 0,7083 |
| 7 | 36 | 0,6227 | 0,5023 | 0,6968 | 0,7176 |
| 8 | 14 | 0,6667 | 0,5000 | 0,7321 | 0,7381 |

## Extensão: classe 9, posterior ao benchmark

13 instâncias de 6 poços.

### Desempenho global (média ± desvio entre instâncias)

| Detector | F1 micro | F1 binário | Precisão | Sensibilidade | AUC-ROC | Degeneradas |
|---|---|---|---|---|---|---|
| Isolation Forest | 0,5000 ± 0,144 | 0,2178 ± 0,285 | 0,2718 ± 0,375 | 0,2179 ± 0,315 | 0,5887 ± 0,282 | 8/13 |
| HBOS | 0,5000 ± 0,000 | 0,6667 ± 0,000 | 0,5000 ± 0,000 | 1,0000 ± 0,000 | 0,6784 ± 0,227 | 13/13 |
| ECOD | 0,6282 ± 0,134 | 0,4628 ± 0,349 | 0,5279 ± 0,393 | 0,4744 ± 0,390 | 0,7906 ± 0,156 | 5/13 |
| COPOD | 0,5321 ± 0,093 | 0,3457 ± 0,296 | 0,4189 ± 0,351 | 0,3846 ± 0,405 | 0,7179 ± 0,202 | 6/13 |
| Dummy | 0,5000 ± 0,000 | 0,0000 ± 0,000 | 0,0000 ± 0,000 | 0,0000 ± 0,000 | 0,5000 ± 0,000 | 13/13 |

### F1 micro por classe

| Classe | n | Isolation Forest | HBOS | ECOD | COPOD |
|---|---|---|---|---|---|
| 9 | 13 | 0,5000 | 0,5000 | 0,6282 | 0,5321 |
