#!/usr/bin/env python3
"""Record what transformers' beam search answers on a table-driven model.

    python3 beam_golden.py > beam_transformers.json

The model has no weights: its next-token logits are integer arithmetic on the tokens so far
(`logits` below), which the Rust test repeats. Every setting of the search is run through
`generate` and the sequences and scores it returns are written down.
"""
import itertools, json, sys
import torch, transformers
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

VOCAB, PAD = 12, 99
PROMPT = [1, 4, 2, 9, 3]


def logits(seq):
    """The model: the logit of token v after `seq`."""
    s, n, last = sum(seq), len(seq), seq[-1]
    return [((17 * (v + 1) + 5 * s + 3 * n * v + 7 * last * v) % 29) * 0.25 + v * 0.0123 for v in range(VOCAB)]


class Toy(PreTrainedModel, GenerationMixin):
    config_class = PretrainedConfig

    def __init__(self):
        super().__init__(PretrainedConfig(vocab_size=VOCAB, is_decoder=True, pad_token_id=PAD))
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kw):
        b, s = input_ids.shape
        out = torch.zeros(b, s, VOCAB)
        for i in range(b):
            out[i, -1] = torch.tensor(logits(input_ids[i].tolist()), dtype=torch.float32)
        return CausalLMOutputWithPast(logits=out)

    def prepare_inputs_for_generation(self, input_ids, **kw):
        return {"input_ids": input_ids}


def main():
    model = Toy().eval()
    cases = []
    for width, early, lp, ends in itertools.product([1, 2, 3, 4], [False, True, "never"], [1.0, 0.0, -1.0, 2.0], [[], [7, 5]]):
        max_new = 6
        out = model.generate(
            torch.tensor([PROMPT]), num_beams=width, num_return_sequences=width, max_new_tokens=max_new,
            eos_token_id=ends or None, pad_token_id=PAD, length_penalty=lp, early_stopping=early,
            do_sample=False, use_cache=False, return_dict_in_generate=True, output_scores=True,
            min_length=0, no_repeat_ngram_size=0, repetition_penalty=1.0)
        seqs = [[t for t in s[len(PROMPT):].tolist() if t != PAD] for s in out.sequences]
        cases.append({"width": width, "early_stopping": early, "length_penalty": lp, "ends": ends,
                      "max_new": max_new, "sequences": seqs, "scores": (out.sequences_scores.tolist() if getattr(out, "sequences_scores", None) is not None else None)})
    json.dump({"transformers": transformers.__version__, "torch": torch.__version__, "vocab": VOCAB,
               "prompt": PROMPT, "cases": cases}, sys.stdout, indent=1)
    print()


main()
