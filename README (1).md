# Laboratório 6 — Construindo um Tokenizador BPE e Explorando o WordPiece

**Disciplina:** Tópicos em Inteligência Artificial 2026.1
**Instituição:** iCEV — Instituto de Ensino Superior
**Professor:** Dimmy Magalhães

> **Nota sobre uso de IA Generativa:** A expressão regular utilizada na
> função `merge_vocab()` (Tarefa 2) para garantir que apenas pares isolados
> sejam substituídos foi gerada/complementada com IA, revisada por Arthur.
> As demais funções (`get_stats`, o loop de treinamento e a análise WordPiece)
> foram implementadas manualmente com base na teoria das aulas.

---

## Contexto

Os modelos de linguagem e a arquitetura Transformer não leem strings de
texto, mas sim tensores numéricos mapeados a partir de um vocabulário. A
abordagem de sub-palavras (como **Byte Pair Encoding** e **WordPiece**)
tornou-se o padrão da indústria para lidar com palavras raras e manter o
tamanho do vocabulário otimizado, como visto nos modelos de tradução
originais que usaram vocabulários de 32.000 a 37.000 tokens.

---

## Estrutura do Repositório

```
lab6-bpe/
│
├── tarefa1_motor_frequencias.py   # get_stats() — contagem de pares adjacentes
├── tarefa2_loop_fusao.py          # merge_vocab() + loop BPE (K=5 iterações)
├── tarefa3_wordpiece.py           # WordPiece com bert-base-multilingual-cased
└── README.md
```

---

## Como Executar

> Tarefas 1 e 2 exigem apenas **Python 3** (sem dependências externas).
> Tarefa 3 exige `pip install transformers`.

```bash
# Tarefa 1 — Motor de frequências
python tarefa1_motor_frequencias.py

# Tarefa 2 — Loop de fusão BPE (K=5)
python tarefa2_loop_fusao.py

# Tarefa 3 — WordPiece com Hugging Face
python tarefa3_wordpiece.py
```

---

## Tarefa 1 — O Motor de Frequências

A função `get_stats(vocab)` recebe o corpus segmentado em caracteres e
retorna a frequência de todos os pares adjacentes ponderada pela frequência
de cada palavra.

**Corpus de entrada (exatamente conforme o enunciado):**

```python
vocab = {
    'l o w </w>'      : 5,
    'l o w e r </w>'  : 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3,
}
```

**Validação obrigatória:** o par `('e', 's')` retorna contagem **9**
(6 de *newest* + 3 de *widest*) — confirmado ✓

---

## Tarefa 2 — O Loop de Fusão

A função `merge_vocab(pair, v_in)` substitui todas as ocorrências isoladas
do par mais frequente pela versão fundida. O loop principal executa **K = 5**
fusões sucessivas.

**Evolução das 5 iterações:**

| Iteração | Par fundido            | Token gerado  | Frequência |
|----------|------------------------|---------------|------------|
| 1        | `e` + `s`              | `es`          | 9          |
| 2        | `es` + `t`             | `est`         | 9          |
| 3        | `est` + `</w>`         | `est</w>`     | 9          |
| 4        | `l` + `o`              | `lo`          | 7          |
| 5        | `lo` + `w`             | `low`         | 7          |

**Validação obrigatória:** o sufixo morfológico `est</w>` é formado
na iteração 3 — confirmado ✓

---

## Tarefa 3 — Integração Industrial e WordPiece

### Resultado da tokenização

Frase de teste:
```
"Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
```

Executar `python tarefa3_wordpiece.py` para ver a tokenização completa.
Exemplo de tokens esperados (podem variar conforme versão da biblioteca):
```
['Os', 'hi', '##per', '-', 'par', '##â', '##metro', '##s', 'do', 'transform',
 '##er', 'são', 'inc', '##ons', '##titu', '##cion', '##almente', 'difíceis',
 'de', 'ajust', '##ar', '.']
```

---

### Relatório: O que significam os sinais `##` nos tokens WordPiece?

O prefixo `##` nos tokens do WordPiece indica que aquele fragmento é uma
**continuação** de uma palavra — ou seja, ele não começa uma nova palavra,
mas se conecta ao token anterior. Por exemplo, ao tokenizar a palavra
`"inconstitucionalmente"`, o modelo não consegue representá-la como um único
token (ela não existe no vocabulário de 119.547 entradas do
`bert-base-multilingual-cased`). Em vez disso, a palavra é desmembrada em
fragmentos como `inc`, `##ons`, `##titu`, `##cion`, `##almente`. O fragmento
`inc` marca o início da palavra (sem `##`), enquanto `##ons`, `##titu`,
`##cion` e `##almente` indicam que são continuações — devem ser concatenados
ao token anterior para reconstruir a palavra original.

**Por que isso impede o travamento do modelo diante de vocabulário
desconhecido?**

Modelos que operam com vocabulários de palavras inteiras sofrem do problema
do **token desconhecido** (`[UNK]`): qualquer palavra não vista durante o
treinamento é substituída por um único token genérico, perdendo toda a
informação morfológica. Com o WordPiece, uma palavra nova como
`"inconstitucionalmente"` — mesmo nunca vista — pode ser decomposta em
fragmentos presentes no vocabulário (`##mente`, `##cion`, etc.), que carregam
informação semântica e morfológica real. O modelo consegue, portanto,
processar e raciocinar sobre palavras desconhecidas com base nos seus
componentes, sem ser forçado a produzir representações degeneradas via
`[UNK]`. Essa propriedade foi determinante para a escalabilidade dos modelos
BERT e seus sucessores em centenas de idiomas com um único vocabulário.

---

## Fundamentos Matemáticos

**BPE — Critério de fusão:**

$$\text{merge} = \arg\max_{(a,b)} \; \text{count}(a, b)$$

**WordPiece — Critério de fusão (probabilístico):**

$$\text{merge} = \arg\max_{(a,b)} \; \frac{P(ab)}{P(a) \cdot P(b)}$$

A diferença central: o BPE maximiza a **frequência absoluta** do par,
enquanto o WordPiece maximiza o **ganho de verossimilhança** — favorecendo
pares cujos componentes isolados são raros mas que juntos são comuns.

---

## Referências

- Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with
  Subword Units*. ACL.
- Schuster, M. & Nakamura, K. (2012). *Japanese and Korean Voice Search*.
  ICASSP. (WordPiece original)
- Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding*.
- Notas de aula — Prof. Dimmy Magalhães, iCEV 2026.1
