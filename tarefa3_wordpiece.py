"""
Laboratório 6 — Tarefa 3: Integração Industrial e WordPiece
============================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Fundamento
----------
O artigo Attention Is All You Need e os modelos que o sucederam (como o
BERT) popularizaram algoritmos de sub-palavras como o WordPiece, que
operam com regras probabilísticas ligeiramente diferentes do BPE clássico.

Enquanto o BPE seleciona o par mais FREQUENTE para fundir, o WordPiece
seleciona o par que maximiza a probabilidade do corpus de treinamento —
favorecendo pares cujos componentes isolados são raros mas juntos são
comuns, o que gera tokens morfologicamente mais coerentes.

Dependências
------------
    pip install transformers
"""

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Tarefa 3.1 — Instanciar tokenizador multilíngue do BERT (WordPiece)
# ---------------------------------------------------------------------------

def load_wordpiece_tokenizer():
    """
    Carrega o tokenizador bert-base-multilingual-cased do Hugging Face.
    Este tokenizador usa o algoritmo WordPiece internamente.
    """
    print("Carregando tokenizador 'bert-base-multilingual-cased'...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    print(f"✓ Tokenizador carregado. Vocab size: {tokenizer.vocab_size:,} tokens\n")
    return tokenizer


# ---------------------------------------------------------------------------
# Tarefa 3.2/3.3 — Segmentar frase de teste com .tokenize()
# ---------------------------------------------------------------------------

# Frase de teste exatamente conforme especificada no enunciado
FRASE_TESTE = (
    "Os hiper-parâmetros do transformer são inconstitucionalmente "
    "difíceis de ajustar."
)


def tokenizar_frase(tokenizer, frase=FRASE_TESTE):
    """
    Utiliza o método .tokenize() para segmentar a frase em sub-palavras.

    Parâmetros
    ----------
    tokenizer : AutoTokenizer  — tokenizador WordPiece do BERT
    frase     : str            — frase a segmentar

    Retorna
    -------
    tokens : list[str]  — lista de sub-palavras geradas pelo WordPiece
    """
    tokens = tokenizer.tokenize(frase)
    return tokens


# ---------------------------------------------------------------------------
# Demonstração e análise
# ---------------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("TAREFA 3 — Integração Industrial e WordPiece")
    print("=" * 65)

    tokenizer = load_wordpiece_tokenizer()

    print(f"Frase de teste:")
    print(f"  \"{FRASE_TESTE}\"\n")

    tokens = tokenizar_frase(tokenizer, FRASE_TESTE)

    # Tarefa 3.4 — Imprimir resultado no terminal
    print("Resultado da tokenização WordPiece (.tokenize()):")
    print(f"  {tokens}")

    print(f"\nNúmero de tokens gerados: {len(tokens)}")

    # Destacar tokens com prefixo ##
    tokens_continuacao = [t for t in tokens if t.startswith("##")]
    print(f"\nTokens de continuação (prefixo ##): {tokens_continuacao}")

    # Análise token a token
    print("\nAnálise token a token:")
    for i, tok in enumerate(tokens):
        tipo = "continuação (##)" if tok.startswith("##") else "início de palavra"
        print(f"  [{i:2d}] '{tok}'  ← {tipo}")

    print("=" * 65)

    return tokens


if __name__ == "__main__":
    demo()
