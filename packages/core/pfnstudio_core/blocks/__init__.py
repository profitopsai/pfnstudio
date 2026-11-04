"""Built-in architecture blocks. Imported eagerly to register them on package load."""

from . import (  # noqa: F401  (registers on import)
    cross_attention,
    grid,
    heads,
    helpers,
    tabular,
    time_encoding,
    transformer,
)
