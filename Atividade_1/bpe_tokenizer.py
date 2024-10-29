from collections import defaultdict
from tqdm import tqdm

class BPE_Tokenizer:

    def __init__(self):
        self.vocab = {}
        self.merges = {}

    def get_stats(self, ids):
        pairs = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])  # Usando tupla
            if pair in pairs:
                pairs[pair] += 1
            else:
                pairs[pair] = 1
        return pairs

    def merge(self, ids, pair, idx):
        merged_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                merged_ids.append(idx)
                i += 2  # Pula o par
            else:
                merged_ids.append(ids[i])
                i += 1
        return merged_ids

    def train(self, text, vocab_size):
        assert vocab_size >= 276, "O tamanho do vocabulário deve ser >= 276."

        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # Utilizando tqdm para acompanhar o progresso
        with tqdm(total=num_merges, desc="Treinamento do tokenizador", unit="iteração") as pbar:
            for i in range(num_merges):
                stats = self.get_stats(ids)
                if not stats:
                    print(f"Treinamento interrompido na iteração {i}: nenhuma combinação possível.")
                    break

                pair = max(stats, key=stats.get)
                idx = 256 + i
                ids = self.merge(ids, pair, idx)
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

                pbar.update(1)  # Atualiza a barra de progresso

        self.merges = merges
        self.vocab = vocab
        print("Treinamento do tokenizador BPE concluído!")

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text):
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