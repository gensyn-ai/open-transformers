# Pinned NGC PyTorch base. Bumping the tag is a separate validated PR
# (see plan/08_ops.md §2). The digest below is a placeholder — populate from
# `docker buildx imagetools inspect nvcr.io/nvidia/pytorch:25.02-py3` and
# commit with the lock.
ARG NGC_TAG=26.03-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

WORKDIR /work


RUN pip install --no-cache-dir --break-system-packages uv==0.5.11

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_BREAK_SYSTEM_PACKAGES=1


COPY pyproject.toml uv.lock* ./
RUN uv pip install --system --break-system-packages \
        "hydra-core==1.3.2" \
        "omegaconf==2.3.0" \
        "pydantic==2.9.2" \
        "tokenizers==0.20.3" \
        "datasets==3.0.1" \
        "wandb==0.18.5" \
        "tqdm" \
        "rich" \
        "pyarrow==17.0.0" \
        "zstandard>=0.22" \
        "boto3>=1.34"


RUN python -c "import torch, transformer_engine; \
    print('torch', torch.__version__, 'cuda', torch.version.cuda); \
    print('TE', transformer_engine.__version__)"

# lm-eval is optional; install on eval boxes.
ARG INSTALL_EVAL=0
RUN if [ "$INSTALL_EVAL" = "1" ]; then \
        uv pip install --system --break-system-packages "lm-eval==0.4.4"; \
    fi

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

ENV PYTHONPATH=/work/src
ENV PYTHONUNBUFFERED=1

# Default entrypoint = bash; launchers under scripts/ wrap torchrun.
CMD ["/bin/bash"]
