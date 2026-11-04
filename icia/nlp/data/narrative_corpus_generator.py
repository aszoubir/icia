"""
ICIA NLP: Narrative English Corpus & Needle-in-a-Haystack Generator
Generates fluent English stories with planted factual attributes and controlled distractor paragraphs.
"""

import random
from typing import Dict, Any, List, Tuple, Optional
import torch
from icia.nlp.tokenizer import ByteTokenizer


class NarrativeCorpusGenerator:
    """
    Generates cohesive English narratives (TinyStories style) with controlled distractor paragraphs
    and planted factual needles for language modeling and retrieval benchmarking.
    """
    CHARACTERS = ["Lily", "Timmy", "Oliver", "Maya", "Lucas", "Sophie", "Ethan", "Chloe"]
    COMPANIONS = ["a fluffy cat named Barnaby", "a friendly puppy named Toby", "a tiny bluebird", "a brown rabbit named Pip"]
    LOCATIONS = ["a sunny green garden", "an ancient whispering forest", "a warm village bakery", "a cozy wooden treehouse", "a quiet dusty library"]
    ACTIVITIES = [
        "was searching for interesting stones near the clear brook",
        "was building a small model sailboat from pine bark",
        "was carefully painting colorful pictures of butterflies",
        "was gathering sweet red berries in a wicker basket",
        "was reading an old book of gentle adventurous tales"
    ]
    DISTRACTOR_EVENTS = [
        "The gentle wind blew through the tall leafy branches, creating a soothing sound like running water.",
        "A group of colorful butterflies danced gracefully above the yellow wildflowers near the fence.",
        "Soon, the warm afternoon sun began to shine brightly, warming the soft grass beneath their feet.",
        "They stopped for a short rest and shared a cool glass of sweet homemade lemonade together.",
        "In the distance, the village bell chimed three times, signaling the calm middle of the peaceful day.",
        "A small brown squirrel scampered quickly up an oak tree, carrying a tiny acorn between its paws.",
        "They noticed how the white clouds drifted slowly across the vast blue sky like fluffy cotton balls.",
        "Everything around them felt safe, quiet, and full of pleasant quiet wonder."
    ]

    FACT_TEMPLATES = [
        ("The secret number carved into the wooden chest was {num}.", "What was the secret number carved into the chest?", "{num}"),
        ("The key to the attic door was made of shiny {metal}.", "What was the key to the attic door made of?", "{metal}"),
        ("The name written inside the magical silver amulet was {name}.", "What was the name written inside the amulet?", "{name}"),
        ("The special flower that bloomed only at dusk had {color} petals.", "What color petals did the special flower have?", "{color}"),
        ("The code required to open the secret garden gate was {code}.", "What was the code to open the garden gate?", "{code}")
    ]

    METALS = ["bronze", "copper", "silver", "gold", "iron", "platinum"]
    NAMES = ["Aria", "Rowan", "Jasper", "Lyra", "Felix", "Cora", "Milo"]
    COLORS = ["purple", "crimson", "amber", "emerald", "silver", "golden"]

    def __init__(self, tokenizer: Optional[ByteTokenizer] = None, seed: int = 42):
        self.tokenizer = tokenizer or ByteTokenizer()
        self.rng = random.Random(seed)

    def generate_story(
        self,
        num_distractor_paragraphs: int = 2,
        with_needle: bool = True
    ) -> Dict[str, Any]:
        """
        Generates a cohesive English narrative.
        Optionally implants a factual needle within the story and appends a query.
        """
        char = self.rng.choice(self.CHARACTERS)
        comp = self.rng.choice(self.COMPANIONS)
        loc = self.rng.choice(self.LOCATIONS)
        act = self.rng.choice(self.ACTIVITIES)

        paragraphs = []

        # Paragraph 1: Introduction
        p1 = f"Once upon a time, {char} and {comp} were spending a quiet morning in {loc}. {char} {act}."
        paragraphs.append(p1)

        # Planted Factual Needle
        fact_sentence = ""
        query_text = ""
        answer_text = ""

        if with_needle:
            template, q_tmpl, a_tmpl = self.rng.choice(self.FACT_TEMPLATES)
            fillers = {
                "num": str(self.rng.randint(1000, 9999)),
                "metal": self.rng.choice(self.METALS),
                "name": self.rng.choice(self.NAMES),
                "color": self.rng.choice(self.COLORS),
                "code": f"{self.rng.randint(10, 99)}-{self.rng.randint(10, 99)}"
            }
            fact_sentence = template.format(**fillers)
            query_text = q_tmpl.format(**fillers)
            answer_text = a_tmpl.format(**fillers)

        # Decide where to place the needle (e.g. early or middle paragraph)
        needle_para_idx = self.rng.randint(1, max(1, num_distractor_paragraphs))

        # Generate distractor paragraphs
        for i in range(1, num_distractor_paragraphs + 1):
            sentences = self.rng.sample(self.DISTRACTOR_EVENTS, min(3, len(self.DISTRACTOR_EVENTS)))
            if with_needle and i == needle_para_idx:
                # Insert needle in this paragraph
                insert_pos = self.rng.randint(0, len(sentences))
                sentences.insert(insert_pos, fact_sentence)
            para = " ".join(sentences)
            paragraphs.append(para)

        # Final Paragraph: Resolution
        p_final = f"As the evening sun set softly behind the gentle green hills, {char} smiled happily and headed home with a peaceful heart."
        paragraphs.append(p_final)

        full_narrative = "\n\n".join(paragraphs)

        if with_needle:
            full_prompt = f"{full_narrative}\n\nQuestion: {query_text} Answer:"
            full_training_text = f"{full_prompt} {answer_text}"
            prompt_tokens = self.tokenizer.encode(full_training_text, add_bos=True, add_eos=False)
            prompt_only_tokens = self.tokenizer.encode(full_prompt, add_bos=True, add_eos=False)
        else:
            full_prompt = full_narrative
            prompt_tokens = self.tokenizer.encode(full_prompt, add_bos=True, add_eos=False)
            prompt_only_tokens = prompt_tokens

        answer_tokens = self.tokenizer.encode(answer_text, add_bos=False, add_eos=True) if with_needle else []

        return {
            "narrative_text": full_narrative,
            "full_prompt": full_prompt,
            "needle_sentence": fact_sentence,
            "query_text": query_text,
            "answer_text": answer_text,
            "num_distractor_paragraphs": num_distractor_paragraphs,
            "prompt_tokens": prompt_tokens,
            "prompt_only_tokens": prompt_only_tokens,
            "answer_tokens": answer_tokens,
            "token_count": len(prompt_only_tokens)
        }

    def generate_batch(
        self,
        batch_size: int = 16,
        num_distractor_paragraphs: int = 2,
        with_needle: bool = True,
        max_len: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates a padded batch for training or needle evaluation.
        """
        samples = [
            self.generate_story(num_distractor_paragraphs=num_distractor_paragraphs, with_needle=with_needle)
            for _ in range(batch_size)
        ]

        p_lens = [len(s["prompt_tokens"]) for s in samples]
        seq_len = max_len or max(p_lens)

        prompt_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long)
        prompt_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32)

        for i, s in enumerate(samples):
            l = min(len(s["prompt_tokens"]), seq_len)
            prompt_tensor[i, :l] = torch.tensor(s["prompt_tokens"][:l], dtype=torch.long)
            prompt_mask[i, :l] = 1.0

        return {
            "prompt_tokens": prompt_tensor,
            "prompt_mask": prompt_mask,
            "samples": samples
        }
