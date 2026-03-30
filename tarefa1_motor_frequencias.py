"""
Laboratório 6 — Tarefa 1: O Motor de Frequências
=================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Fundamento
----------
O algoritmo BPE inicia com um corpus onde as palavras já estão separadas
em caracteres e acrescidas de um símbolo especial de fim de palavra </w>.
O primeiro passo é varrer esse corpus para contar a frequência de cada par
de símbolos adjacentes.
"""

# ---------------------------------------------------------------------------
# Corpus de treinamento (exatamente conforme especificado no enunciado)
# ---------------------------------------------------------------------------

vocab = {
    'l o w </w>'      : 5,
    'l o w e r </w>'  : 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3,
}


# ---------------------------------------------------------------------------
# Tarefa 1 — get_stats
# ---------------------------------------------------------------------------

def get_stats(vocab):
    """
    Varre o corpus e conta a frequência de cada par de símbolos adjacentes.

    Para cada palavra do vocabulário (representada como uma string de símbolos
    separados por espaço), percorre todos os pares consecutivos e acumula a
    contagem ponderada pela frequência da palavra no corpus.

    Parâmetros
    ----------
    vocab : dict[str, int]
        Dicionário onde a chave é a palavra segmentada em símbolos separados
        por espaço (ex: 'l o w </w>') e o valor é a frequência dessa palavra
        no corpus.

    Retorna
    -------
    pairs : dict[tuple[str, str], int]
        Dicionário mapeando cada par adjacente à sua frequência total.
        Ex: {('e', 's'): 9, ('s', 't'): 9, ...}
    """
    pairs = {}

    for word, freq in vocab.items():
        symbols = word.split()

        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq

    return pairs


# ---------------------------------------------------------------------------
# Demonstração e validação
# ---------------------------------------------------------------------------

def demo():
    print("=" * 60)
    print("TAREFA 1 — O Motor de Frequências")
    print("=" * 60)

    print("\nCorpus de treinamento:")
    for word, freq in vocab.items():
        print(f"  '{word}' : {freq}")

    pairs = get_stats(vocab)

    print("\nFrequências de todos os pares adjacentes:")
    for pair, freq in sorted(pairs.items(), key=lambda x: -x[1]):
        print(f"  {pair} : {freq}")

    # Validação obrigatória: ('e', 's') deve retornar 9
    assert pairs[('e', 's')] == 9, (
        f"FALHA: esperado 9 para ('e', 's'), obtido {pairs[('e', 's')]}"
    )

    par_max   = max(pairs, key=pairs.get)
    freq_max  = pairs[par_max]

    print(f"\nPar mais frequente : {par_max} → {freq_max}")
    print(f"\n✓ Validação: ('e', 's') = {pairs[('e', 's')]} (esperado: 9) — APROVADO")
    print("=" * 60)

    return pairs


if __name__ == "__main__":
    demo()
