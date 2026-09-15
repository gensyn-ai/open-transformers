"""Emit same-input training digests for source/native wheel comparisons.

Run in separate processes with each installed package on PYTHONPATH. This is
a small QAT composition probe, not a substitute for a published replay unit.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import pretrain
import repop
from pretrain.config import load_config
from pretrain.model import build_model
from pretrain.model.fused_loss import fused_ce_z_loss
from pretrain.model.init import init_weights
from pretrain.optim.registry import build_optimizer
from pretrain.train.state_hash import compute_state_hash


def digest(tensor):
    return hashlib.sha256(
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--expect-native", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expect_native:
        assert pretrain.__file__.endswith(".so"), pretrain.__file__
        assert repop.__file__.endswith(".so"), repop.__file__
    torch.set_num_threads(2)
    cfg = load_config("100m_smoke_repop")
    cfg.model.d_model = 128
    cfg.model.n_heads = 2
    cfg.model.n_kv_heads = 1
    cfg.model.head_dim = 64
    cfg.model.ffn_intermediate = 256
    cfg.model.n_layers = 1
    cfg.model.vocab_size = 256
    cfg.model.qat.enabled = True
    cfg.optim.name = "adamw_repop"
    model = build_model(cfg.model)
    init_weights(model, seed=20260910)
    model = model.to(args.device)
    optimizer = build_optimizer(model, cfg.optim)
    tokens = (torch.arange(32, dtype=torch.int64).reshape(1, 32) % 256).to(args.device)
    labels = ((tokens + 1) % 256).contiguous()
    results = {
        "device": args.device,
        "torch": torch.__version__,
        "package_paths": {"repop": repop.__file__, "pretrain": pretrain.__file__},
        "input_hash": digest(tokens),
        "init_hash": compute_state_hash(
            model, optimizer=optimizer, include_grads=False, prev_hash=None
        ),
        "steps": [],
    }
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = model(tokens)
        ce, zloss = fused_ce_z_loss(out.logits, labels, cfg.model.z_loss.coeff)
        loss = ce + zloss
        loss.backward()
        assert torch.isfinite(loss)
        gradients = {
            name: digest(p.grad)
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        assert gradients
        optimizer.step()
        results["steps"].append(
            {
                "logits": digest(out.logits),
                "loss": digest(loss),
                "gradients": gradients,
                "weights": {name: digest(p) for name, p in model.named_parameters()},
                "optimizer": {
                    name: {
                        key: digest(value) if isinstance(value, torch.Tensor) else value
                        for key, value in optimizer.state[p].items()
                    }
                    for name, p in model.named_parameters()
                    if p in optimizer.state
                },
                "state_hash": compute_state_hash(
                    model, optimizer=optimizer, include_grads=False, prev_hash=None
                ),
            }
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
