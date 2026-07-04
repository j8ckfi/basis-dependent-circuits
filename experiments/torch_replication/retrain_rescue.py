"""Partial-retraining rescue with the missing controls (AUDIT.md S4).

The paper freezes a transplanted donor head and retrains the host, recovering
>=80% of baseline in 100-500 steps, concluding the donor head "carries useful
structure". But without a control arm, fast recovery could equally mean the
host reroutes around a frozen junk head using its redundant capacity.

Arms (per donor->recipient pair, all with the critical-slot head frozen):
  frozen_donor    - donor critical head in the slot (paper's arm)
  frozen_random   - fresh random-init head in the slot
  frozen_zero     - zeroed head in the slot
  frozen_shuffled - element-shuffled donor head (matched weight statistics)

If frozen_donor recovers materially faster than the junk arms, the donor head
contributed function. Additionally, after each retraining run the frozen head
is value-ablated and re-evaluated: a large drop means the host actually came
to rely on the frozen head; no drop means it routed around it.

Usage: python -m experiments.torch_replication.retrain_rescue --pairs is0_ds42:is1_ds42 ...
"""

import argparse
import copy
import json

import torch

from .common import (CKPT_DIR, RESULTS_DIR, SMALL_CONFIG, evaluate_fixed,
                     fixed_eval_set, masked_loss, train_stream)
from .torch_model import (GPT, get_head_weights, head_slices,
                          random_head_weights, set_head_weights,
                          shuffled_head_weights)
from .transplant_suite import find_critical_head, load_run

RETRAIN_STEPS = 1000
RETRAIN_LR = 1e-3
RETRAIN_DATA_SEED = 4242
EVAL_EVERY = 25
RECOVERY_FRACTION = 0.8


def freeze_grad_hooks(model, layer, head):
    """Zero gradients on the frozen head's Q/K/V rows and out-proj columns."""
    sl = head_slices(model.config, head)
    attn = model.blocks[layer].attn

    def qkv_hook(grad):
        g = grad.clone()
        g[sl["q"], :] = 0
        g[sl["k"], :] = 0
        g[sl["v"], :] = 0
        return g

    def out_hook(grad):
        g = grad.clone()
        g[:, sl["out_cols"]] = 0
        return g

    attn.qkv_proj.weight.register_hook(qkv_hook)
    attn.out_proj.weight.register_hook(out_hook)


def retrain_arm(recip, head_weights, layer, head, eval_batches, recip_base, label):
    model = copy.deepcopy(recip)
    set_head_weights(model, layer, head, head_weights)
    frozen_before = get_head_weights(model, layer, head)
    freeze_grad_hooks(model, layer, head)
    # weight_decay must be 0: decoupled decay would mutate frozen weights.
    opt = torch.optim.AdamW(model.parameters(), lr=RETRAIN_LR, weight_decay=0.0)
    stream = train_stream(RETRAIN_DATA_SEED)

    start_acc, _ = evaluate_fixed(model, eval_batches)
    curve = [{"step": 0, "acc": start_acc}]
    k_star = None
    target = RECOVERY_FRACTION * recip_base
    for step in range(1, RETRAIN_STEPS + 1):
        x, y, m = next(stream)
        model.train()
        loss = masked_loss(model(x), y, m)
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.clamp_(-1.0, 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            acc, _ = evaluate_fixed(model, eval_batches)
            curve.append({"step": step, "acc": acc})
            if k_star is None and acc >= target:
                k_star = step
                if step >= 500:
                    break
    # verify the freeze actually held
    frozen_after = get_head_weights(model, layer, head)
    freeze_ok = all(torch.equal(frozen_before[k], frozen_after[k]) for k in frozen_before)

    final_acc = curve[-1]["acc"]
    # post-retraining reliance probe: value-ablate the frozen head
    mask = torch.ones(SMALL_CONFIG.n_heads)
    mask[head] = 0.0
    ablated_acc, _ = evaluate_fixed(model, eval_batches, head_value_masks={layer: mask})
    print(f"  [{label}] start={start_acc:.4f} k*={k_star} final={final_acc:.4f} "
          f"ablate_frozen={ablated_acc:.4f} freeze_ok={freeze_ok}", flush=True)
    return {"start_acc": start_acc, "k_star_80pct": k_star, "final_acc": final_acc,
            "frozen_head_ablation_acc": ablated_acc,
            "frozen_head_reliance_drop": final_acc - ablated_acc,
            "freeze_intact": freeze_ok, "curve": curve}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--out", default="retrain_rescue_results.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_batches = fixed_eval_set()
    results = {}
    out_path = RESULTS_DIR / args.out
    for spec in args.pairs:
        donor_name, recip_name = spec.split(":")
        donor, recip = load_run(donor_name), load_run(recip_name)
        (dl, dh), _, _ = find_critical_head(donor, eval_batches)
        (rl, rh), recip_base, _ = find_critical_head(recip, eval_batches)
        pair_key = f"{donor_name}->{recip_name}"
        print(f"[{pair_key}] donor crit L{dl}H{dh}, recipient crit L{rl}H{rh} "
              f"(base {recip_base:.4f})", flush=True)
        donor_w = get_head_weights(donor, dl, dh)
        arms = {
            "frozen_donor": donor_w,
            "frozen_random": random_head_weights(SMALL_CONFIG, seed=456),
            "frozen_zero": {k: torch.zeros_like(v) for k, v in donor_w.items()},
            "frozen_shuffled": shuffled_head_weights(donor_w, seed=123),
        }
        results[pair_key] = {"recipient_baseline": recip_base, "arms": {}}
        for arm_name, w in arms.items():
            results[pair_key]["arms"][arm_name] = retrain_arm(
                recip, w, rl, rh, eval_batches, recip_base, f"{pair_key}:{arm_name}")
        out_path.write_text(json.dumps(results, indent=2))
        print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
