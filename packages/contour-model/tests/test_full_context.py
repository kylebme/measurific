import torch
from contour_model.model import ContourTransformer,ModelConfig


def test_full_context_includes_every_token_and_checkpoint_backward():
    torch.set_num_threads(2)
    model=ContourTransformer(ModelConfig(width=16,global_width=32,local_depth=1,global_depth=1,
        pooled_tokens=8,direct_global_context=True,full_context=True,label_schema='primitives-compounds-v2'))
    with torch.no_grad():model.context_gate.fill_(.2)
    sizes=[]
    hook=model.context_attention.register_forward_pre_hook(lambda m,args:sizes.append(args[1].shape[1]))
    out=model(torch.randn(280,12),[0,140,280]);out['primary'].square().mean().backward()
    hook.remove()
    assert sizes and all(n==280 for n in sizes)
    assert torch.isfinite(model.context_attention.q_proj_weight.grad).all()
    assert model.context_attention.q_proj_weight.grad.abs().sum()>0
