"""Model abstraction. A model is a typed view of model.yaml, plus a builder that
composes registered blocks into a runnable network."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .model_blocks import filter_block_config
from .registry import get_block


class BlockConfig(BaseModel):
    type: str
    name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class OutputHead(BaseModel):
    name: str
    task: str


class ModelSpec(BaseModel):
    id: str
    name: str
    version: str
    description: str | None = None
    blocks: list[BlockConfig]
    input_shape: str | None = None
    output_heads: list[OutputHead] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    # Declarative Try-it / Inference manifest for a NATIVE studio model (no
    # adapter). Maps a task key → {label, description, input_schema:{kind,…},
    # output_schema:{kind,field,…}, sample?} — the same shape an adapter's
    # capabilities() returns and the schema-driven Try-it renders. Empty (the
    # default) means the model declares no tasks, so the Try-it falls back to
    # the legacy per-eval form. The prior may declare this instead; the model
    # wins when both do. See predict.run_inference's __capabilities__ branch.
    capabilities: dict[str, Any] = Field(default_factory=dict)


class Model:
    """Constructs a network by instantiating each block from the registry, in order.

    Frameworks are deliberately not assumed — block constructors return whatever they
    return (PyTorch nn.Module, plain callable, JAX flax module, etc.). The local
    training adapter knows how to handle PyTorch; for other frameworks, plug in your
    own training adapter.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.modules: list[tuple[str, Any]] = []
        for i, block in enumerate(spec.blocks):
            cls = get_block(block.type)
            instance = cls(**filter_block_config(block.type, block.config))
            label = block.name or f"{block.type}_{i}"
            self.modules.append((label, instance))

    def by_name(self, name: str) -> Any:
        for label, mod in self.modules:
            if label == name:
                return mod
        raise KeyError(f"No block named '{name}' in model '{self.spec.id}'.")
