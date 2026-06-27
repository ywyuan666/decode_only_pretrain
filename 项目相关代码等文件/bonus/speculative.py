"""Speculative Decoding implementation."""
import torch
import torch.nn as nn
from typing import List, Tuple


class DraftModel(nn.Module):
    """A smaller, faster draft model for speculative decoding."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        # Use smaller dimensions
        draft_config = config.copy()
        draft_config["d_model"] = 64
        draft_config["n_layers"] = 2
        draft_config["n_heads"] = 4
        draft_config["n_kv_heads"] = 2

        from model.transformer import ModernTransformer
        self.model = ModernTransformer(draft_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SpeculativeDecoder:
    """Implements speculative decoding with draft model verification."""

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: DraftModel,
        tokenizer,
        gamma: int = 5,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.gamma = gamma  # Number of speculative tokens

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> str:
        """Generate text using speculative decoding.

        Returns:
            Generated text and speedup metrics.
        """
        device = next(self.target_model.parameters()).device
        self.target_model.eval()
        self.draft_model.eval()

        input_ids = self.tokenizer.encode(prompt).ids
        generated = input_ids.copy()
        total_draft_tokens = 0
        accepted_tokens = 0

        while len(generated) < len(input_ids) + max_new_tokens:
            # 1. Draft model generates gamma tokens
            draft_tokens = []
            draft_input = torch.tensor([generated], device=device)

            for _ in range(self.gamma):
                logits = self.draft_model(draft_input)
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                draft_tokens.append(next_token)
                draft_input = torch.cat(
                    [draft_input, torch.tensor([[next_token]], device=device)], dim=1
                )
                total_draft_tokens += 1
                if next_token == self.tokenizer.token_to_id("</s>"):
                    break

            if not draft_tokens:
                break

            # 2. Target model verifies in parallel
            verify_input = torch.tensor([generated + draft_tokens], device=device)
            target_logits = self.target_model(verify_input)  # (1, L+gamma, V)

            # 3. Rejection sampling
            for i, draft_token in enumerate(draft_tokens):
                # Draft probability
                draft_logits = self.draft_model(
                    torch.tensor([generated + draft_tokens[:i]], device=device)
                )
                draft_prob = torch.softmax(draft_logits[:, -1, :] / temperature, dim=-1)[
                    0, draft_token
                ]

                # Target probability
                target_prob = torch.softmax(
                    target_logits[:, len(generated) + i - 1, :] / temperature, dim=-1
                )[0, draft_token]

                # Accept with probability min(1, target_prob / draft_prob)
                r = torch.rand(1).item()
                if r < min(1.0, (target_prob / (draft_prob + 1e-8)).item()):
                    generated.append(draft_token)
                    accepted_tokens += 1
                else:
                    # Reject and sample from residual distribution
                    residual_prob = torch.softmax(
                        target_logits[:, len(generated) + i, :] / temperature, dim=-1
                    )
                    # Remove draft's contribution (simplified)
                    next_token = torch.multinomial(residual_prob, 1).item()
                    generated.append(next_token)
                    break  # Stop verifying further draft tokens
            else:
                # All draft tokens accepted, sample one more from target
                final_prob = torch.softmax(
                    target_logits[:, -1, :] / temperature, dim=-1
                )
                next_token = torch.multinomial(final_prob, 1).item()
                generated.append(next_token)

            if generated[-1] == self.tokenizer.token_to_id("</s>"):
                break

        acceptance_rate = accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0
        print(f"Speculative decoding: acceptance rate = {acceptance_rate:.2f}")
        return self.tokenizer.decode(generated)