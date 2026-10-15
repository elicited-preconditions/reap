# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from .base_encoder import EncoderFactory, GraphEncoderBase, PygData
from .hetero_encoder import HGraphEncoder
from .horizon_hetero_encoder import HorizonHGraphEncoder
from .ilg_hetero_encoder import ILGHGraphEncoder
from .relation_dict import RelationDict
from .relation_formatter import Node, RelationFormatter
from .transition_hetero_encoder import (
    TransitionEffectsHGraphEncoder,
    TransitionHGraphEncoder,
)

_MIFROST_ADAPTERS_REGISTERED = False


def _register_mifrost_xmimir_adapters() -> None:
    global _MIFROST_ADAPTERS_REGISTERED
    if _MIFROST_ADAPTERS_REGISTERED:
        return
    try:
        import mifrost
    except ModuleNotFoundError:
        return

    from reap.xmimir import XAction, XDomain, XLiteral, XState

    mifrost.register_domain_adapter(XDomain, lambda domain: domain.advanced)
    mifrost.register_state_adapter(XState, lambda state: state.base)
    mifrost.register_literal_adapter(XLiteral, lambda literal: literal.base)
    mifrost.register_action_adapter(XAction, lambda action: action.advanced)
    _MIFROST_ADAPTERS_REGISTERED = True


_register_mifrost_xmimir_adapters()


def make_default_encoder_factory() -> EncoderFactory:
    import mifrost

    return EncoderFactory(mifrost.HGraphEncoder)
