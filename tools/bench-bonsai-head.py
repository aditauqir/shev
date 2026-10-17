#!/usr/bin/env python3
"""Compare the native tied 2-bit output head with a 4-bit repack in RAM.

Run with the pack's runtime requirements, after shev has created its worker.
Never edits weights or model configuration. This tests repacking an already
2-bit head, NOT the accuracy of a separately trained higher-precision model.
"""
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

root = Path.home() / ".local/share/shev/models"
pack = root / "Ternary-Bonsai-2-2B-mlx-2bit"
sys.argv = [str(root / "shev_bonsai_worker.py"), str(pack)]
spec = importlib.util.spec_from_file_location("worker", sys.argv[0])
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
import mlx.core as mx
from mlx import nn
from runtime import fwht


class RepackedEmbedding(nn.Module):
    """Keep input embeddings native; replace only their tied output operation."""

    def __init__(self, base, bits):
        super().__init__()
        self.base = base
        self.bits = bits
        dense = mx.dequantize(base.weight, base.scales, base.biases,
                              group_size=128, bits=2)
        self.weight, self.scales, self.biases = mx.quantize(
            dense, group_size=128, bits=bits)
        mx.eval(self.parameters())

    def __call__(self, ids):
        return self.base(ids)

    def as_linear(self, hidden):
        if self.base.block:
            hidden = fwht(hidden, self.base.block, self.base.signs)
        return mx.quantized_matmul(hidden, self.weight, self.scales, self.biases,
                                   transpose=True, group_size=128, bits=self.bits)


cases = [
    ("Pay the electricity bill by Friday.", True),
    ("The invoice is already paid. Nothing else is needed.", False),
    ("Please send the signed contract today.", True),
    ("Thanks, I received the document. No reply needed.", False),
    ("The deadline was cancelled; do not submit anything.", False),
    ("Your appointment needs confirmation by 5pm today.", True),
]
source = (Path(__file__).resolve().parents[1] / "src/lib.rs").read_text()
system_prompt = source.split('const LOCAL_SYSTEM_PROMPT: &str = r#"', 1)[1].split('"#;', 1)[0]
engine = worker.Engine()
lm = engine.model.language_model if engine.vision else engine.model
if not lm.args.tie_word_embeddings:
    raise RuntimeError("This benchmark expects the inspected tied-output pack")
base = lm.model.embed_tokens
print(json.dumps({"head": "tied embed_tokens.as_linear", "native_bits": 2,
                  "packed_shape": list(base.weight.shape)}), flush=True)
variants = {2: base, 4: RepackedEmbedding(base, 4)}
mx.random.seed(42)
x = mx.random.normal((1, 1, 2048)).astype(mx.float16)
reference = base.as_linear(x)
mx.eval(reference)
for bits, embedding in variants.items():
    lm.model.embed_tokens = embedding
    for _ in range(4):
        mx.eval(embedding.as_linear(x))
    timings = []
    for _ in range(30):
        started = time.perf_counter()
        logits = embedding.as_linear(x)
        mx.eval(logits)
        timings.append(1000 * (time.perf_counter() - started))
    rmse = mx.sqrt(mx.mean((logits.astype(mx.float32) -
                            reference.astype(mx.float32)) ** 2)).item()
    durations, correct, valid = [], 0, 0
    for message, expected in cases:
        prompt = (f'{system_prompt}\n\nWHAT THE USER WANTS TO DO WITH THIS MESSAGE:\n'
                  'Identify the outstanding action, if any.\n\n'
                  f'THE MESSAGE, AS A WHOLE:\n<message>\n{message}\n</message>')
        mx.random.seed(42)
        started = time.perf_counter()
        text = engine.generate(prompt)
        durations.append(1000 * (time.perf_counter() - started))
        try:
            decision = json.loads(text.strip().removeprefix('```json').removesuffix('```'))
            valid += isinstance(decision, dict)
            correct += decision.get('take_action') is expected
        except (ValueError, AttributeError):
            pass
    print(json.dumps({"bits": bits, "median_head_ms": statistics.median(timings),
                      "synthetic_logit_rmse": rmse,
                      "median_decision_ms": statistics.median(durations),
                      "valid_json": valid, "correct_actions": correct,
                      "cases": len(cases)}), flush=True)
lm.model.embed_tokens = base
