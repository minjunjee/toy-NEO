"""Name -> class registries and the one strict builder used by every config block."""

from __future__ import annotations

import inspect

ENCODERS: dict[str, type] = {}
DECODERS: dict[str, type] = {}
PROGRAMMERS: dict[str, type] = {}
CODES: dict[str, type] = {}
EXECUTORS: dict[str, type] = {}
MODELS: dict[str, type] = {}
METHODS: dict[str, type] = {}
EVALS: dict[str, object] = {}
DATASETS: dict[str, type] = {}


def build(registry: dict, block: dict, **extra):
    """Instantiate ``registry[block["type"]]`` with the remaining keys as kwargs.

    Unknown keys raise so a stale yaml never runs silently with a default.
    """

    if not isinstance(block, dict) or "type" not in block:
        raise KeyError(f"config block must be a mapping with a 'type' key, got {block!r}")
    block = dict(block)
    name = block.pop("type")
    try:
        target = registry[name]
    except KeyError as error:
        raise KeyError(f"unknown type {name!r}; registered: {sorted(registry)}") from error
    parameters = inspect.signature(target).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        unknown = set(block) | set(extra)
        unknown -= set(parameters)
        if unknown:
            raise KeyError(f"{name}: unknown keys {sorted(unknown)}; accepts {sorted(parameters)}")
    return target(**block, **extra)


def import_all() -> None:
    """Import every module that registers something, so registries are populated."""

    import data.datasets  # noqa: F401
    import data.babytheorist  # noqa: F401
    import components.codes  # noqa: F401
    import components.codes_growable  # noqa: F401
    import components.codes_restart  # noqa: F401
    import components.codes_orbit  # noqa: F401
    import components.decoders  # noqa: F401
    import components.encoders  # noqa: F401
    import components.cnn_any_size  # noqa: F401
    import components.executors  # noqa: F401
    import components.executors_expert  # noqa: F401
    import components.programmers  # noqa: F401
    import components.programmer_pair  # noqa: F401
    import models.neo  # noqa: F401
    import models.neo_visible_state  # noqa: F401
    import models.neo_repeat  # noqa: F401
    import models.neo_counted  # noqa: F401
    import models.neo_expert  # noqa: F401
    import models.vae  # noqa: F401
    import methods.neo  # noqa: F401
    import methods.neo_warm  # noqa: F401
    import methods.neo_compose  # noqa: F401
    import methods.vae_pretrain  # noqa: F401
    import evals.self_explain_exact  # noqa: F401
    import evals.transfer_exact  # noqa: F401
    import evals.babytheorist  # noqa: F401
