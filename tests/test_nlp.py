"""
Unit Tests: ICIA NLP Subsystem (Tokenizer, Generator, Latent Reasoner)
"""

import torch
import pytest
from icia.nlp.tokenizer import ByteTokenizer
from icia.nlp.tasks.symbolic_reasoning_generator import SymbolicReasoningGenerator
from icia.nlp.models.icia_nlp_reasoner import ICIANLPReasoner


def test_byte_tokenizer_roundtrip():
    """Verify lossless UTF-8 encoding and decoding."""
    tokenizer = ByteTokenizer()
    text = "alpha = 14 + beta! π ≈ 3.14159"
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    
    assert tokens[0] == tokenizer.BOS_TOKEN
    assert tokens[-1] == tokenizer.EOS_TOKEN
    
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_symbolic_generator_batch():
    """Verify generated batch dimensions and metadata validity."""
    generator = SymbolicReasoningGenerator(seed=123)
    batch = generator.generate_batch(batch_size=4, num_hops=2, num_distractors=1)
    
    assert batch["prompt_tokens"].size(0) == 4
    assert batch["prompt_mask"].size(0) == 4
    assert batch["answer_tokens"].size(0) == 4
    assert batch["answer_vals"].size(0) == 4


def test_icia_latent_reasoner_zero_scratchpad():
    """Verify that ICIANLPReasoner forward pass executes K steps with 0 scratchpad tokens."""
    tokenizer = ByteTokenizer()
    model = ICIANLPReasoner(vocab_size=tokenizer.vocab_size, d_model=32, d_latent=64, k_think_steps=3)
    
    prompt = "Set alpha = 5. Set beta = alpha + 3. What is beta?"
    toks = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=False)], dtype=torch.long)
    mask = torch.ones_like(toks, dtype=torch.float32)
    
    out = model(prompt_tokens=toks, prompt_mask=mask, k_steps=3)
    
    assert out["scratchpad_tokens_emitted"] == 0
    assert len(out["verifications"]) == 3
    assert out["pred_val"].shape == (1,)
    assert out["s_final"].shape == (1, 64)
