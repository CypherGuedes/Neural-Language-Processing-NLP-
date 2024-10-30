from collections import defaultdict
from tqdm import tqdm

class BPE_Tokenizer:
    """
    Implementação do Tokenizador BPE (Byte Pair Encoding).
    """

    def __init__(self):
        self.vocab = {}  # Vocabulário para mapear IDs para bytes.
        self.merges = {}  # Dicionário para armazenar as mesclagens realizadas.

    def get_pairs(self, ids):
        """
        Gera todos os pares consecutivos de tokens da lista de IDs.
        """
        pairs = []
        for i in range(len(ids) - 1):
            pairs.append((ids[i], ids[i + 1]))  # Gera pares consecutivos.
        return pairs

    def get_stats(self, ids):
        """
        Calcula a frequência de pares consecutivos de tokens.
        """
        pairs = defaultdict(int)
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            pairs[pair] += 1  # Incrementa a contagem para o par.
        return pairs

    def get_frequency(self, pair, stats):
        """
        Retorna a frequência de um par específico de tokens.

        Parâmetros:
        -----------
        pair : tuple
            O par de tokens para o qual a frequência é solicitada.
        stats : dict
            Dicionário com as frequências de todos os pares.

        Retorna:
        --------
        int:
            A frequência do par fornecido.
        """
        return stats.get(pair, 0)  # Retorna 0 se o par não estiver presente.

    def merge(self, ids, pair, idx):
        """
        Mescla um par específico de tokens em um novo token.
        """
        merged_ids = []
        skip = False

        for i in range(len(ids) - 1):
            if skip:
                skip = False
                continue

            if (ids[i], ids[i + 1]) == pair:
                merged_ids.append(idx)  # Adiciona o novo token.
                skip = True  # Pula o próximo elemento.
            else:
                merged_ids.append(ids[i])

        if not skip:
            merged_ids.append(ids[-1])  # Adiciona o último token.

        return merged_ids

    def train(self, text, vocab_size):
        """
        Treina o tokenizador BPE no texto fornecido.
        """
        assert vocab_size >= 276, "O tamanho do vocabulário deve ser >= 276."

        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}

        pbar = tqdm(total=num_merges, desc="Treinamento do tokenizador", unit="iteração")

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                print(f"Treinamento interrompido na iteração {i}: nenhuma combinação possível.")
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)

            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            pbar.update(1)

        pbar.close()
        print("Treinamento do tokenizador BPE concluído!")

    def encode(self, text):
        """
        Codifica um texto em uma lista de IDs usando o BPE.
        """
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")), default=None)

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)

        return ids

    def decode(self, ids):
        """
        Decodifica uma lista de IDs em texto.
        """
        try:
            text_bytes = b"".join(self.vocab[idx] for idx in ids)
            return text_bytes.decode("utf-8", errors="replace")
        except KeyError as e:
            print(f"Erro: ID {e} não encontrado no vocabulário.")
            return ""

