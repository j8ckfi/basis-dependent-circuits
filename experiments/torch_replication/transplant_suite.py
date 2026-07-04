"""Cross-framework replication of the transplant experiment + the missing controls.

For each trained run, identifies the critical head by per-head value ablation on
the fixed eval set, then for each donor→recipient pair runs:

Static arms (replication of transplant_unified_eval.py):
  baseline, zero_critical, transplant (donor critical → recipient critical slot),
  shuffled_donor, random_head, donor_to_noncritical_slot, noncritical_donor_to_critical_slot.

Adapter arms (adjudicating the paper-vs-JSON tournament contradiction, with the
controls the original design lacked):
  adapter_donor          — linear head-interface adapters (WITH bias, identity-init)
                           around the transplanted donor head; recipient frozen.
  adapter_zero           — same adapters around a zeroed head. With biases this is a
                           meaningful control (the original bias-free control was
                           architecturally disconnected and could not learn).
  adapter_random         — same adapters around a fresh random-init head
                           (expressivity control: can adapters synthesize a new head?).
  adapter_donor_then_zero_probe — after training adapter_donor, zero the donor head
                           weights and re-evaluate with the trained adapters kept.
                           High retained accuracy ⇒ adapters bypassed the donor;
                           a large drop ⇒ the rescue was donor-mediated.

All evals use the fixed 1024-sequence set (seed 9999), macro per-sequence accuracy,
paired bootstrap + sign-flip tests against the recipient baseline.

Usage: python -m experiments.torch_replication.transplant_suite --pairs is0_ds42:is1_ds42 ...
"""

import argparse
import copy
import itertools
import json

import numpy as np
import torch

from .common import (CKPT_DIR, RESULTS_DIR, SMALL_CONFIG, evaluate_fixed,
                     fixed_eval_set, masked_loss, paired_bootstrap, train_stream)
from .torch_model import (GPT, create_model, get_head_weights, head_slices,
                          random_head_weights, set_head_weights,
                          shuffled_head_weights, zero_head_weights)

ADAPTER_STEPS = 1000
ADAPTER_LR = 1e-3
ADAPTER_DATA_SEED = 7777


def load_run(name):
    model = GPT(SMALL_CONFIG)
    model.load_state_dict(torch.load(CKPT_DIR / name / "final.pt", weights_only=True))
    model.eval()
    return model


def find_critical_head(model, eval_batches):
    """Per-head value-output ablation; returns ((layer, head), drops)."""
    base, _ = evaluate_fixed(model, eval_batches)
    drops = {}
    for layer in range(SMALL_CONFIG.n_layers):
        for head in range(SMALL_CONFIG.n_heads):
            mask = torch.ones(SMALL_CONFIG.n_heads)
            mask[head] = 0.0
            acc, _ = evaluate_fixed(model, eval_batches, head_value_masks={layer: mask})
            drops[f"L{layer}H{head}"] = base - acc
    crit = max(drops, key=drops.get)
    layer, head = int(crit[1]), int(crit[3])
    return (layer, head), base, drops


class HeadAdapters(torch.nn.Module):
    """Identity-initialized linear maps (with bias) on one head's read/write interfaces."""

    def __init__(self, d_model):
        super().__init__()
        self.a_in = torch.nn.Linear(d_model, d_model, bias=True)
        self.a_out = torch.nn.Linear(d_model, d_model, bias=True)
        with torch.no_grad():
            self.a_in.weight.copy_(torch.eye(d_model))
            self.a_in.bias.zero_()
            self.a_out.weight.copy_(torch.eye(d_model))
            self.a_out.bias.zero_()


def forward_with_adapters(model, idx, adapters, layer, head):
    """Forward pass where (layer, head) reads through a_in and writes through a_out."""
    cfg = model.config
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device)
    x = model.wte(idx) + model.wpe(pos)
    causal = torch.triu(torch.full((T, T), -1e9, device=idx.device), diagonal=1)
    for i, block in enumerate(model.blocks):
        xn = block.ln1(x)
        attn = block.attn
        if i == layer:
            sl = head_slices(cfg, head)
            qkv = attn.qkv_proj(xn).reshape(B, T, 3, cfg.n_heads, cfg.d_head)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0].clone(), qkv[1].clone(), qkv[2].clone()
            xa = adapters.a_in(xn)
            W = attn.qkv_proj.weight
            q[:, head] = xa @ W[sl["q"], :].T
            k[:, head] = xa @ W[sl["k"], :].T
            v[:, head] = xa @ W[sl["v"], :].T
            a = torch.softmax((q @ k.transpose(-2, -1)) * (cfg.d_head ** -0.5) + causal, dim=-1)
            out = a @ v  # (B, H, T, dh)
            contrib = torch.zeros(B, T, cfg.d_model, device=idx.device)
            Wo = attn.out_proj.weight
            for h in range(cfg.n_heads):
                c = out[:, h] @ Wo[:, head_slices(cfg, h)["out_cols"]].T
                contrib = contrib + (adapters.a_out(c) if h == head else c)
            x = x + contrib
        else:
            x = x + attn(xn)
        x = x + block.mlp(block.ln2(x))
    return model.ln_f(x) @ model.wte.weight.T


def train_adapters(model, layer, head, eval_batches, label):
    """Train head-interface adapters with the base model frozen."""
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = HeadAdapters(SMALL_CONFIG.d_model)
    opt = torch.optim.AdamW(adapters.parameters(), lr=ADAPTER_LR, weight_decay=0.0)
    stream = train_stream(ADAPTER_DATA_SEED)
    for step in range(ADAPTER_STEPS):
        x, y, m = next(stream)
        loss = masked_loss(forward_with_adapters(model, x, adapters, layer, head), y, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"    [{label}] adapter step {step} loss={loss.item():.4f}", flush=True)
    return adapters


@torch.no_grad()
def evaluate_adapted(model, adapters, layer, head, eval_batches):
    per_sequence = []
    for x_np, y_np, m_np in eval_batches:
        x = torch.from_numpy(x_np.astype(np.int64))
        preds = forward_with_adapters(model, x, adapters, layer, head).argmax(dim=-1).numpy()
        at_mask = m_np > 0.5
        correct = (preds == y_np) & at_mask
        for row in range(x_np.shape[0]):
            denom = int(at_mask[row].sum())
            per_sequence.append(float(correct[row].sum() / denom) if denom else 0.0)
    return float(np.mean(per_sequence)), per_sequence


def run_pair(donor_name, recip_name, eval_batches, results):
    donor = load_run(donor_name)
    recip = load_run(recip_name)
    (dl, dh), donor_base, donor_drops = find_critical_head(donor, eval_batches)
    (rl, rh), recip_base, recip_drops = find_critical_head(recip, eval_batches)
    pair_key = f"{donor_name}->{recip_name}"
    print(f"[{pair_key}] donor crit L{dl}H{dh} (base {donor_base:.4f}), "
          f"recipient crit L{rl}H{rh} (base {recip_base:.4f})", flush=True)

    _, base_seq = evaluate_fixed(recip, eval_batches)
    arms = {}

    def static_arm(name, weights):
        m = copy.deepcopy(recip)
        set_head_weights(m, rl, rh, weights)
        acc, seq = evaluate_fixed(m, eval_batches)
        arms[name] = {"acc": acc, "vs_baseline": paired_bootstrap(seq, base_seq)}
        print(f"  {name}: {acc:.4f}", flush=True)
        return m

    donor_w = get_head_weights(donor, dl, dh)
    recip_w = get_head_weights(recip, rl, rh)

    arms["baseline"] = {"acc": recip_base}
    static_arm("zero_critical", {k: torch.zeros_like(v) for k, v in recip_w.items()})
    transplanted = static_arm("transplant", donor_w)
    static_arm("shuffled_donor", shuffled_head_weights(donor_w, seed=123))
    static_arm("random_head", random_head_weights(SMALL_CONFIG, seed=456))

    # donor critical head into a non-critical recipient slot
    noncrit_slots = [h for h in range(SMALL_CONFIG.n_heads) if not (rl == 0 and h == rh)]
    nc = noncrit_slots[0]
    m = copy.deepcopy(recip)
    set_head_weights(m, rl, nc, donor_w)
    acc, seq = evaluate_fixed(m, eval_batches)
    arms["donor_to_noncritical_slot"] = {"slot": f"L{rl}H{nc}", "acc": acc,
                                         "vs_baseline": paired_bootstrap(seq, base_seq)}
    print(f"  donor_to_noncritical_slot: {acc:.4f}", flush=True)

    # non-critical donor head into the critical slot
    dnc = [h for h in range(SMALL_CONFIG.n_heads) if not (dl == 0 and h == dh)][0]
    static_arm("noncritical_donor_to_critical_slot", get_head_weights(donor, dl, dnc))

    # ---- adapter arms ----
    for arm_name, host in [
        ("adapter_donor", transplanted),
        ("adapter_zero", None),   # built below
        ("adapter_random", None),
    ]:
        if arm_name == "adapter_zero":
            host = copy.deepcopy(recip)
            zero_head_weights(host, rl, rh)
        elif arm_name == "adapter_random":
            host = copy.deepcopy(recip)
            set_head_weights(host, rl, rh, random_head_weights(SMALL_CONFIG, seed=789))
        host = copy.deepcopy(host)
        adapters = train_adapters(host, rl, rh, eval_batches, f"{pair_key}:{arm_name}")
        acc, seq = evaluate_adapted(host, adapters, rl, rh, eval_batches)
        arms[arm_name] = {"acc": acc, "vs_baseline": paired_bootstrap(seq, base_seq)}
        print(f"  {arm_name}: {acc:.4f}", flush=True)
        if arm_name == "adapter_donor":
            zero_head_weights(host, rl, rh)
            acc2, seq2 = evaluate_adapted(host, adapters, rl, rh, eval_batches)
            arms["adapter_donor_then_zero_probe"] = {
                "acc": acc2, "drop_from_adapter_donor": acc - acc2,
                "vs_baseline": paired_bootstrap(seq2, base_seq)}
            print(f"  adapter_donor_then_zero_probe: {acc2:.4f} "
                  f"(drop {acc - acc2:+.4f})", flush=True)

    results[pair_key] = {
        "donor_critical": f"L{dl}H{dh}", "recipient_critical": f"L{rl}H{rh}",
        "donor_baseline": donor_base, "recipient_baseline": recip_base,
        "donor_ablation_drops": donor_drops, "recipient_ablation_drops": recip_drops,
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", required=True,
                        help="donor:recipient run-name pairs, e.g. is0_ds42:is1_ds42")
    parser.add_argument("--out", default="transplant_suite_results.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_batches = fixed_eval_set()
    results = {}
    out_path = RESULTS_DIR / args.out
    for spec in args.pairs:
        donor_name, recip_name = spec.split(":")
        run_pair(donor_name, recip_name, eval_batches, results)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
