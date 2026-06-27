# config.py

config = {
    "vocab_size": 8192,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "n_kv_heads": 2,
    "multiple_of": 256,
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-6,
    "max_seq_len": 512,
    "dropout": 0.1,
    "n_experts": 2,
    "num_experts_per_tok": 1
}