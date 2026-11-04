"""
ICIA NLP: Symbolic Multi-Step Reasoning Task Generator
Generates multi-hop variable binding, arithmetic deduction, and distractor-injected problems.
"""

import random
from typing import List, Dict, Any, Tuple, Optional
import torch
from icia.nlp.tokenizer import ByteTokenizer


class SymbolicReasoningGenerator:
    """
    Generates synthetic multi-hop symbolic reasoning problems.
    Supports variable binding graphs of depth H (hops) and D distractor statements.
    """
    VAR_NAMES = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        "kappa", "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
        "var_A", "var_B", "var_C", "var_D", "var_E", "var_F", "var_G", "var_H"
    ]
    DISTRACTOR_NAMES = [
        "noise_P", "noise_Q", "noise_R", "noise_S", "noise_T", "noise_U", "noise_V",
        "bogus_1", "bogus_2", "bogus_3", "dummy_X", "dummy_Y", "dummy_Z", "junk_K"
    ]
    OPERATIONS = ["+", "-"]

    def __init__(self, tokenizer: Optional[ByteTokenizer] = None, seed: int = 42):
        self.tokenizer = tokenizer or ByteTokenizer()
        self.rng = random.Random(seed)

    def generate_single_problem(
        self,
        num_hops: int = 2,
        num_distractors: int = 2,
        max_val: int = 30
    ) -> Dict[str, Any]:
        """
        Generates a single multi-hop reasoning problem.
        Example (hops=2, distractors=1):
          Premise 1: Set alpha = 7.
          Distractor: Set dummy_X = 42.
          Premise 2: Set beta = alpha + 4.
          Premise 3: Set gamma = beta - 3.
          Query: What is gamma?
          Answer: 8
        """
        used_vars = self.rng.sample(self.VAR_NAMES, num_hops + 1)
        used_distractors = self.rng.sample(self.DISTRACTOR_NAMES, min(num_distractors, len(self.DISTRACTOR_NAMES)))
        
        # Build the chain of computation
        # Step 0: Base variable
        base_val = self.rng.randint(2, max_val)
        val_map = {used_vars[0]: base_val}
        
        statements = []
        statements.append((f"Set {used_vars[0]} = {base_val}.", True))
        
        cot_scratchpad = [f"Step 0: {used_vars[0]} is {base_val}."]
        
        current_val = base_val
        for h in range(1, num_hops + 1):
            prev_var = used_vars[h - 1]
            curr_var = used_vars[h]
            op = self.rng.choice(self.OPERATIONS)
            delta = self.rng.randint(1, 15)
            
            if op == "+":
                current_val = val_map[prev_var] + delta
            else:
                # Ensure positive values
                if val_map[prev_var] <= delta:
                    op = "+"
                    current_val = val_map[prev_var] + delta
                else:
                    current_val = val_map[prev_var] - delta
            
            val_map[curr_var] = current_val
            stmt = f"Set {curr_var} = {prev_var} {op} {delta}."
            statements.append((stmt, True))
            cot_scratchpad.append(f"Step {h}: {curr_var} = {val_map[prev_var]} {op} {delta} = {current_val}.")

        # Generate distractors (irrelevant variable assignments or noise arithmetic)
        for d_name in used_distractors:
            d_val = self.rng.randint(10, 99)
            if self.rng.random() < 0.5:
                stmt = f"Set {d_name} = {d_val}."
            else:
                d_delta = self.rng.randint(1, 20)
                d_op = self.rng.choice(self.OPERATIONS)
                stmt = f"Set {d_name} = {d_val} {d_op} {d_delta}."
            statements.append((stmt, False))

        # Interleave distractors into the statement sequence (preserving dependency order for real variables)
        real_stmts = [s for s, is_real in statements if is_real]
        dist_stmts = [s for s, is_real in statements if not is_real]
        
        # Insert distractors at random positions after the first statement
        final_sequence = list(real_stmts)
        for d_stmt in dist_stmts:
            pos = self.rng.randint(1, len(final_sequence))
            final_sequence.insert(pos, d_stmt)

        target_var = used_vars[-1]
        target_answer = val_map[target_var]
        
        premise_text = " ".join(final_sequence)
        query_text = f"What is {target_var}?"
        full_prompt = f"{premise_text} {query_text}"
        answer_text = str(target_answer)
        cot_text = " ".join(cot_scratchpad) + f" Therefore {target_var} = {target_answer}."

        # Tokenize
        prompt_tokens = self.tokenizer.encode(full_prompt, add_bos=True, add_eos=False)
        # Append special query token
        prompt_tokens.append(self.tokenizer.QUERY_TOKEN)
        
        answer_tokens = self.tokenizer.encode(answer_text, add_bos=False, add_eos=True)
        cot_tokens = self.tokenizer.encode(cot_text, add_bos=False, add_eos=True)

        return {
            "prompt_text": full_prompt,
            "query_var": target_var,
            "answer_text": answer_text,
            "answer_val": target_answer,
            "cot_text": cot_text,
            "num_hops": num_hops,
            "num_distractors": num_distractors,
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "cot_tokens": cot_tokens,
            "variables": used_vars,
            "distractors": used_distractors
        }

    def generate_batch(
        self,
        batch_size: int = 32,
        num_hops: int = 2,
        num_distractors: int = 2,
        max_prompt_len: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates a padded PyTorch batch of symbolic reasoning problems.
        """
        samples = [
            self.generate_single_problem(num_hops=num_hops, num_distractors=num_distractors)
            for _ in range(batch_size)
        ]

        # Determine max length
        p_lens = [len(s["prompt_tokens"]) for s in samples]
        a_lens = [len(s["answer_tokens"]) for s in samples]
        cot_lens = [len(s["cot_tokens"]) for s in samples]

        max_p = max_prompt_len or max(p_lens)
        max_a = max(a_lens)
        max_cot = max(cot_lens)

        # Pad tokens
        prompt_tensor = torch.zeros((batch_size, max_p), dtype=torch.long)
        prompt_mask = torch.zeros((batch_size, max_p), dtype=torch.float32)
        answer_tensor = torch.zeros((batch_size, max_a), dtype=torch.long)
        cot_tensor = torch.zeros((batch_size, max_cot), dtype=torch.long)

        answer_vals = torch.zeros((batch_size,), dtype=torch.long)

        for i, s in enumerate(samples):
            p_len = min(len(s["prompt_tokens"]), max_p)
            prompt_tensor[i, :p_len] = torch.tensor(s["prompt_tokens"][:p_len], dtype=torch.long)
            prompt_mask[i, :p_len] = 1.0

            a_len = min(len(s["answer_tokens"]), max_a)
            answer_tensor[i, :a_len] = torch.tensor(s["answer_tokens"][:a_len], dtype=torch.long)

            c_len = min(len(s["cot_tokens"]), max_cot)
            cot_tensor[i, :c_len] = torch.tensor(s["cot_tokens"][:c_len], dtype=torch.long)

            answer_vals[i] = s["answer_val"]

        return {
            "prompt_tokens": prompt_tensor,
            "prompt_mask": prompt_mask,
            "answer_tokens": answer_tensor,
            "cot_tokens": cot_tensor,
            "answer_vals": answer_vals,
            "samples": samples
        }
