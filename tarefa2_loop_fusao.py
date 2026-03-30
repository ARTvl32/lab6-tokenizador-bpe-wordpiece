"""
Laboratório 6 — Tarefa 2: O Loop de Fusão
==========================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Fundamento
----------
Após identificar o par mais frequente, o algoritmo realiza uma fusão (merge),
criando um novo token que passará a fazer parte do vocabulário do modelo.
A eficácia desse método em modelos de tradução de ponta se dá por sua
capacidade de criar representações eficientes para sequências de caracteres
comuns.

Nota sobre uso de IA
--------------------
A expressão regular utilizada na função merge_vocab() para garantir que
apenas pares isolados (delimitados por espaço ou borda de string) sejam
substituídos foi gerada/complementada com IA, revisada por Arthur.
"""

import re
from tarefa1_motor_frequencias import vocab as vocab_inicial, get_stats


# ---------------------------------------------------------------------------
# Tarefa 2.1 — merge_vocab
# ---------------------------------------------------------------------------

def merge_vocab(pair, v_in):
    """
    Substitui todas as ocorrências isoladas de um par adjacente de símbolos
    pela versão fundida (merged), retornando o vocabulário atualizado.

    Exemplo:
        pair  = ('e', 's')
        antes : 'n e w e s t </w>'
        depois: 'n e w es t </w>'

    A expressão regular garante que apenas o par ISOLADO seja substituído —
    ou seja, delimitado por espaço ou borda da string — evitando substituições
    parciais incorretas dentro de tokens já fundidos.

    Parâmetros
    ----------
    pair  : tuple[str, str]
        Par de símbolos adjacentes a fundir (ex: ('e', 's')).
    v_in  : dict[str, int]
        Vocabulário atual com palavras segmentadas por espaço.

    Retorna
    -------
    v_out : dict[str, int]
        Vocabulário atualizado com o par fundido em todos os contextos.

    Nota sobre IA
    -------------
    O padrão regex abaixo foi gerado/complementado com IA, revisado por Arthur.
    O padrão usa lookahead/lookbehind para garantir que somente pares
    delimitados por espaço ou borda da string sejam substituídos.
    """
    v_out   = {}
    bigram  = re.escape(' '.join(pair))          # ex: 'e s'
    pattern = re.compile(
        r'(?<!\S)' + bigram + r'(?!\S)'
    )
    replacement = ''.join(pair)                   # ex: 'es'

    for word in v_in:
        new_word = pattern.sub(replacement, word)
        v_out[new_word] = v_in[word]

    return v_out


# ---------------------------------------------------------------------------
# Tarefa 2.2 — Loop Principal de Treinamento do Tokenizador (K = 5)
# ---------------------------------------------------------------------------

def training_loop(vocab, num_merges=5):
    """
    Executa o loop principal de treinamento do tokenizador BPE.

    A cada iteração:
        1. Chama get_stats() para obter as frequências de todos os pares
        2. Seleciona o par mais frequente
        3. Chama merge_vocab() para fundir o par no vocabulário
        4. Imprime o par fundido e o estado do vocabulário atualizado

    Parâmetros
    ----------
    vocab      : dict[str, int]  — vocabulário inicial com corpus segmentado
    num_merges : int             — número de fusões a realizar (K = 5)

    Retorna
    -------
    vocab : dict[str, int]  — vocabulário final após todas as fusões
    merges : list[tuple]    — lista dos pares fundidos em ordem
    """
    merges = []

    for i in range(1, num_merges + 1):
        pairs      = get_stats(vocab)
        best_pair  = max(pairs, key=pairs.get)
        best_freq  = pairs[best_pair]

        vocab  = merge_vocab(best_pair, vocab)
        merges.append(best_pair)

        print(f"\n--- Iteração {i} ---")
        print(f"Par fundido : {best_pair}  (frequência: {best_freq})")
        print(f"Vocabulário após fusão:")
        for word, freq in vocab.items():
            print(f"  '{word}' : {freq}")

    return vocab, merges


# ---------------------------------------------------------------------------
# Demonstração e validação
# ---------------------------------------------------------------------------

def demo():
    print("=" * 60)
    print("TAREFA 2 — O Loop de Fusão")
    print("=" * 60)

    print("\nVocabulário inicial:")
    for word, freq in vocab_inicial.items():
        print(f"  '{word}' : {freq}")

    # Executar K = 5 fusões
    vocab_final, merges = training_loop(dict(vocab_inicial), num_merges=5)

    print("\n" + "=" * 60)
    print("RESUMO DAS 5 FUSÕES:")
    for i, pair in enumerate(merges, 1):
        print(f"  {i}. {pair[0]} + {pair[1]} → {''.join(pair)}")

    # Validação: verificar formação do sufixo est</w>
    tokens_formados = set()
    for word in vocab_final:
        tokens_formados.update(word.split())

    print(f"\nTokens morfológicos formados: {sorted(tokens_formados)}")

    sufixo_ok = 'est</w>' in tokens_formados
    print(f"\n✓ Validação: sufixo 'est</w>' formado = {sufixo_ok} — "
          f"{'APROVADO' if sufixo_ok else 'FALHOU'}")
    print("=" * 60)

    return vocab_final, merges


if __name__ == "__main__":
    demo()
