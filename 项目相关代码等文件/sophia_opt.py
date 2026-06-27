# sophia_opt.py
import torch
from torch.optim import Optimizer

class SophiaG(Optimizer):
    """简化版 SophiaG 优化器，用于对比实验。"""
    def __init__(self, params, lr=1e-3, betas=(0.965, 0.99), rho=0.01, weight_decay=0.1, eps=1e-12):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)
                state['step'] += 1
                exp_avg = state['exp_avg']
                hessian = state['hessian']
                beta1, beta2 = group['betas']
                rho = group['rho']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction = 1 - beta2 ** state['step']
                h = hessian / bias_correction
                p.mul_(1 - group['lr'] * group['weight_decay'])
                update = exp_avg / (torch.sqrt(h) + group['eps']).clamp(min=rho)
                p.add_(update, alpha=-group['lr'])
        return loss