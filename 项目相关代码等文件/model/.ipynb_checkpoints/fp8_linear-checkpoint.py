import torch
import torch.nn as nn

class FP8Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("weight_scale", torch.tensor(1.0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        self.weight = self.weight.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
        self.weight_scale = self.weight_scale.to(device)

        if x.dtype != torch.float8_e4m3fn:
            return super().forward(x)

        w = self.weight
        w_amax = w.abs().max()
        w_max = torch.finfo(torch.float8_e4m3fn).max
        scale_w = (w_amax / w_max).clamp(min=1e-8)
        w_q = (w / scale_w).to(torch.float8_e4m3fn)

        x_f = x.float()
        x_amax = x_f.abs().max()
        scale_x = (x_amax / w_max).clamp(min=1e-8)
        x_q = (x_f / scale_x).to(torch.float8_e4m3fn)

        out = torch._scaled_mm(
            x_q, w_q.t(),
            scale_a=scale_x,
            scale_b=scale_w,
            out_dtype=torch.float16
        )

        if self.bias is not None:
            out += self.bias
        return out