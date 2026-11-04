"""
ICIA NLP: Zero-Dependency Byte-Level Tokenizer
Pure Python UTF-8 byte tokenizer with special tokens.
"""

from typing import List, Union


class ByteTokenizer:
    """
    Byte-level UTF-8 tokenizer.
    Vocabulary:
    - 0: <pad>
    - 1: <bos>
    - 2: <eos>
    - 3: <sep>
    - 4: <query>
    - 5: <ans>
    - 6..261: raw byte values (0..255)
    """
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    SEP_TOKEN = 3
    QUERY_TOKEN = 4
    ANS_TOKEN = 5
    BYTE_OFFSET = 6
    VOCAB_SIZE = 262

    def __init__(self):
        self.vocab_size = self.VOCAB_SIZE

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        raw_bytes = text.encode("utf-8")
        tokens = []
        if add_bos:
            tokens.append(self.BOS_TOKEN)
        for b in raw_bytes:
            tokens.append(b + self.BYTE_OFFSET)
        if add_eos:
            tokens.append(self.EOS_TOKEN)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        byte_list = []
        for tid in token_ids:
            if tid >= self.BYTE_OFFSET:
                byte_list.append(tid - self.BYTE_OFFSET)
            elif tid == self.EOS_TOKEN:
                break
        return bytes(byte_list).decode("utf-8", errors="replace")
