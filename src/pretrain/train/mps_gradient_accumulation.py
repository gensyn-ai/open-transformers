"""Keep audit gradients on CPU between microbatches, adding on MPS."""

import torch


@torch.no_grad()
def accumulate_mps_fp32_gradients(
    model: torch.nn.Module, accumulator: dict[str, torch.Tensor]
) -> None:
    """Preserve autograd's old-plus-new MPS addition and release device grads.

    CPU storage is a byte-preserving transfer, not a CPU reduction: MPS float
    addition can differ from CPU addition for subnormal values and signed zeros.
    The caller owns one accumulator per virtual rank and folds it after the last
    microbatch. Parameters without a new gradient retain their previous total.
    """
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if grad.device.type != "mps" or grad.dtype != torch.float32:
            raise ValueError("MPS gradient offload requires FP32 MPS gradients")
        if name in accumulator:
            previous = accumulator[name]
            if previous.device.type != "cpu" or previous.dtype != torch.float32:
                raise ValueError("MPS gradient accumulator requires FP32 CPU storage")
            merged = previous.to(device=grad.device)
            merged.add_(grad)
            accumulator[name] = merged.cpu()
            del merged
        else:
            accumulator[name] = grad.detach().cpu()
        parameter.grad = None
