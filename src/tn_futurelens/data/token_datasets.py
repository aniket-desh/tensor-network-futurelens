r"""Tokenized sequence loaders.

WikiText-103: we split the corpus into ARTICLES (top-level ``= Title =`` headers)
and chunk each article into fixed-length blocks, so every sequence lies within a
single article -> clean document boundaries for the position-correlation
diagnostics (briefing §5, §12). A BOS token is prepended to each block (GPT-2's
standard <|endoftext|>), so position 0 is BOS and position i (i>=0) predicts
token i+1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import Tensor

_ARTICLE_HEADER = re.compile(r"^ = [^=].* = $")


def iter_wikitext_articles(split: str = "train", max_articles: int | None = None):
    """Yield article strings from WikiText-103 (raw), grouping lines by top-level header."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    buf: list[str] = []
    n = 0
    for row in ds:
        line = row["text"]
        if _ARTICLE_HEADER.match(line):
            if buf:
                yield "".join(buf)
                n += 1
                if max_articles is not None and n >= max_articles:
                    return
                buf = []
        buf.append(line)
    if buf and (max_articles is None or n < max_articles):
        yield "".join(buf)


@dataclass
class TokenSequences:
    tokens: Tensor       # [S, seq_len] long, BOS-prefixed
    article_ids: Tensor  # [S] long, which article each sequence came from
    seq_len: int
    bos_token_id: int


def build_token_sequences(
    tokenizer,
    seq_len: int,
    num_sequences: int,
    split: str = "train",
    min_article_tokens: int = 64,
    bos_token_id: int | None = None,
) -> TokenSequences:
    """Build ``num_sequences`` BOS-prefixed blocks of length ``seq_len`` from articles."""
    if bos_token_id is None:
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = tokenizer.eos_token_id
    body = seq_len - 1  # leave room for the prepended BOS
    seqs: list[list[int]] = []
    article_ids: list[int] = []
    aid = 0
    for article in iter_wikitext_articles(split=split):
        ids = tokenizer.encode(article)
        if len(ids) < min_article_tokens:
            continue
        for start in range(0, len(ids) - body + 1, body):
            block = ids[start : start + body]
            seqs.append([bos_token_id] + block)
            article_ids.append(aid)
            if len(seqs) >= num_sequences:
                return TokenSequences(
                    tokens=torch.tensor(seqs, dtype=torch.long),
                    article_ids=torch.tensor(article_ids, dtype=torch.long),
                    seq_len=seq_len,
                    bos_token_id=bos_token_id,
                )
        aid += 1
    return TokenSequences(
        tokens=torch.tensor(seqs, dtype=torch.long),
        article_ids=torch.tensor(article_ids, dtype=torch.long),
        seq_len=seq_len,
        bos_token_id=bos_token_id,
    )
