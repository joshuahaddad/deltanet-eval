"""Memory per sequence for the three Qwen models in the benchmark: the reference for the post's KV cache table.

Config values are copied from each model's Hugging Face config.json (the exact checkpoints the eval called):
  https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/raw/main/config.json
  https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/raw/main/config.json
  https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json
Run this file to check every number the table shows.
"""

BF16, FP32 = 2, 4  # bytes per value
KiB, MiB, GiB = 2**10, 2**20, 2**30

MODELS = {  # full-attention layers, KV heads, head dim
    "Qwen3 30B-A3B": (48, 4, 128),
    "Qwen3 235B-A22B": (94, 4, 128),
    # 48 layers, full_attention_interval 4: every 4th layer is attention, the other 36 are Gated DeltaNet
    "Qwen3-Next 80B-A3B": (48 // 4, 2, 256),
}
DELTA_LAYERS, V_HEADS, K_DIM, V_DIM = 48 - 48 // 4, 32, 128, 128  # linear_num_value_heads, linear_key/value_head_dim
CONV_CHANNELS, CONV_KERNEL = 2 * 16 * 128 + 32 * 128, 4  # q, k, v channels into the short conv; linear_conv_kernel_dim


def kv_per_token(layers, kv_heads, head_dim, dtype=BF16):
    """K and V, for every attention layer, every KV head."""
    return 2 * layers * kv_heads * head_dim * dtype


def delta_state(state_dtype, conv_cols):
    """The fixed state of the Gated DeltaNet layers: one d_k x d_v matrix per value head, plus the short-conv cache
    (activations, bf16). It does not grow with context. Serving stacks differ in how they keep it:
    the Hugging Face reference holds the matrices in fp32 and caches all 4 conv columns; vLLM's default keeps them in
    the model dtype (bf16) and caches kernel - 1 = 3 columns."""
    return DELTA_LAYERS * (V_HEADS * K_DIM * V_DIM * state_dtype + CONV_CHANNELS * conv_cols * BF16)


HF_STATE, VLLM_STATE = delta_state(FP32, CONV_KERNEL), delta_state(BF16, CONV_KERNEL - 1)


if __name__ == "__main__":
    per_token = {m: kv_per_token(*c) for m, c in MODELS.items()}
    assert {m: b // KiB for m, b in per_token.items()} == {"Qwen3 30B-A3B": 96, "Qwen3 235B-A22B": 188, "Qwen3-Next 80B-A3B": 24}
    at_192k = {m: b * 192 * 1024 / GiB for m, b in per_token.items()}
    assert at_192k == {"Qwen3 30B-A3B": 18.0, "Qwen3 235B-A22B": 35.25, "Qwen3-Next 80B-A3B": 4.5}
    # claim: the hybrid's cache is 4x smaller than the 30B's per token and ~7.8x smaller than the 235B's
    assert per_token["Qwen3 30B-A3B"] / per_token["Qwen3-Next 80B-A3B"] == 4
    assert round(per_token["Qwen3 235B-A22B"] / per_token["Qwen3-Next 80B-A3B"], 1) == 7.8
    # claim: the delta layers add a fixed ~38 MiB (vLLM default) to ~74 MiB (HF reference), under 2% of the total at 192k
    assert (round(VLLM_STATE / MiB, 1), round(HF_STATE / MiB, 2)) == (37.7, 74.25)
    assert HF_STATE / (HF_STATE + at_192k["Qwen3-Next 80B-A3B"] * GiB) < 0.02
    for m, b in per_token.items():
        print(f"{m:20s} {b // KiB:4d} KiB/token  " + "  ".join(f"{k}k {b * k * 1024 / GiB:6.2f} GiB" for k in (8, 32, 128, 192)))
    print(f"Gated DeltaNet state (fixed): {VLLM_STATE / MiB:.1f} MiB (vLLM default) to {HF_STATE / MiB:.2f} MiB (HF reference)")
    print("all claims check out")
