# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

# this is for saving / loading agents and training checkpoints
import os
import pathlib
import posixpath
import re
import time
from pathlib import Path
from typing import Optional

import torch
import yaml

import reap.xmimir as xmimir
from reap.encoding import HGraphEncoder, HorizonHGraphEncoder
from reap.encoding.base_encoder import GraphEncoderBase
from reap.models.aggr import AttentionAggregation
from reap.models.mlp import SimpleMLP
from reap.models.pooling import GlobalSumPool
from reap.models.relational_gnn import RelationalGNN
from reap.utils.object_embeddings import ActionPoolingModule


def _atomic_torch_save(obj, path):
    """Atomic torch.save: write to path.tmp first, then rename to path.

    Workers (re)load model state files from disk while the trainer may be
    mid-write; without an atomic rename a worker can read a partial or
    zero-byte file and crash (Model.load_from_folder also retries a failed
    torch.load after 2.5 s).  os.replace is atomic on POSIX filesystems, so
    once the rename completes the worker sees the new file in full or the
    old file in full, never a partial.
    """
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Model(torch.nn.Module):
    encoder: Optional[GraphEncoderBase]
    q_gnn: RelationalGNN
    policy_gnn: RelationalGNN
    q_pooling: torch.nn.Module
    policy_pooling: torch.nn.Module
    q_readout: torch.nn.Module
    policy_readout: torch.nn.Module
    domain: xmimir.XDomain

    def __init__(
        self,
        encoder=None,
        q_gnn=None,
        policy_gnn=None,
        q_pooling=None,
        policy_pooling=None,
        q_readout=None,
        policy_readout=None,
        domain=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.q_gnn = q_gnn
        self.policy_gnn = policy_gnn
        self.q_readout = q_readout
        self.policy_readout = policy_readout
        self.q_pooling = q_pooling
        self.policy_pooling = policy_pooling
        self.domain = domain

    @staticmethod
    def create_new(
        domain: xmimir.XDomain,
        embedding_size: int = 32,
        num_layer: int = 30,
        attn_pooling: bool = False,
        encoding: str = "default",
        layer_norm: bool = False,
        pooling_mode: str = "mask",
    ):
        config = {
            "encoder": {"type": encoding},
            "domain": domain,
            "gnn": {
                "embedding_size": embedding_size,
                "num_layer": num_layer,
                "layer_norm": layer_norm,
            },
            "pooling": {"parameters": attn_pooling, "mode": pooling_mode},
        }
        return Model.load_from_folder(config)

    @staticmethod
    def load_from_folder(load_folder: str | dict) -> "Model":
        if isinstance(load_folder, str) or isinstance(load_folder, pathlib.Path):
            # Handle "/model/best"
            load_folder = Path(load_folder)
            if load_folder.name == "best":
                parent = load_folder.parent  # /model

                if parent.name != "checkpoint":
                    parent = parent / "checkpoint"

                if not parent.exists():
                    print(
                        f"Checkpoint Model directory does not exist: {parent}, reverting to default"
                    )
                    load_folder = parent.parent
                else:
                    checkpoint_dirs = []
                    pattern = re.compile(r"^checkpoint_(\d+)$")

                    for d in parent.iterdir():
                        if d.is_dir():
                            match = pattern.match(d.name)
                            if match:
                                checkpoint_dirs.append((int(match.group(1)), d))

                    # Select highest checkpoint number
                    _, best_dir = max(checkpoint_dirs, key=lambda x: x[0])
                    load_folder = best_dir

                    if not checkpoint_dirs:
                        raise FileNotFoundError(
                            f"No checkpoint_* folders found in {parent}"
                        )

            config_path = os.path.join(load_folder, "config.yaml")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"No config.yaml found in {load_folder}")

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            config = load_folder

        # Load domain. A released checkpoint bundles its domain as <folder>/domain.pddl; that file takes precedence
        # over config.yaml's domain_path, which is relative to the repository root of the training run.
        domain = config.get("domain")
        domain_path = config.get("domain_path")
        if not isinstance(load_folder, dict):
            bundled = os.path.join(load_folder, "domain.pddl")
            if os.path.isfile(bundled):
                domain_path = bundled
        if domain_path is not None:
            domain = xmimir.XDomain.parse(str(domain_path))

        # Load encoder
        encoder_config = config.get("encoder")
        encoder_type = encoder_config.get("type")
        if encoder_type is None or encoder_type == "default":
            encoder = HGraphEncoder(
                domain=domain,
                symbol_type_id=encoder_config.get("symbol_type_id", "_symbol_"),
                ignore_actions=encoder_config.get("ignore_actions", False),
            )
        elif encoder_type == "horizon":
            encoder = HorizonHGraphEncoder(
                domain=domain,
                successor_mode="delta",
                enable_parent_relation=True,
                enable_sibling_relation=False,
                enable_cousin_relation=False,
            )

        # Load GNN
        gnn_config = config.get("gnn")
        if gnn_config is not None:
            q_gnn = RelationalGNN(
                embedding_size=gnn_config.get("embedding_size"),
                num_layer=gnn_config.get("num_layer"),
                aggr=None,
                symbol_type_ids=encoder.symbol_type_id,
                relation_dict=encoder.relation_dict,
            )
            policy_gnn = RelationalGNN(
                embedding_size=gnn_config.get("embedding_size"),
                num_layer=gnn_config.get("num_layer"),
                aggr=None,
                symbol_type_ids=encoder.symbol_type_id,
                relation_dict=encoder.relation_dict,
            )
            if not isinstance(load_folder, dict):
                gnn_path = os.path.join(load_folder, "q_gnn.pt")
                if os.path.exists(gnn_path):
                    try:
                        q_gnn.load_state_dict(torch.load(gnn_path, map_location="cpu"))
                    except:
                        time.sleep(2.5)
                        q_gnn.load_state_dict(torch.load(gnn_path, map_location="cpu"))
                gnn_path = os.path.join(load_folder, "policy_gnn.pt")
                if os.path.exists(gnn_path):
                    try:
                        policy_gnn.load_state_dict(
                            torch.load(gnn_path, map_location="cpu")
                        )
                    except:
                        time.sleep(2.5)
                        policy_gnn.load_state_dict(
                            torch.load(gnn_path, map_location="cpu")
                        )

        # Load pooling
        pooling_config = config.get("pooling")
        if pooling_config is not None:
            p = pooling_config.get("parameters")
            pooling_mode = pooling_config.get("mode", "mask")  # checkpoints without a saved mode used "mask"
            if p is not None and p:
                q_pooling = ActionPoolingModule(
                    AttentionAggregation(
                        gnn_config.get("embedding_size"), split_features=False
                    ),
                    mode=pooling_mode,
                )
                policy_pooling = ActionPoolingModule(
                    AttentionAggregation(
                        gnn_config.get("embedding_size"), split_features=False
                    ),
                    mode=pooling_mode,
                )
            else:
                q_pooling = ActionPoolingModule(GlobalSumPool(), mode=pooling_mode)
                policy_pooling = ActionPoolingModule(GlobalSumPool(), mode=pooling_mode)
            if not isinstance(load_folder, dict):
                q_pooling_path = os.path.join(load_folder, "q_pooling.pt")
                policy_pooling_path = os.path.join(load_folder, "policy_pooling.pt")
                if os.path.exists(q_pooling_path):
                    q_pooling.load_state_dict(
                        torch.load(q_pooling_path, map_location="cpu")
                    )
                    policy_pooling.load_state_dict(
                        torch.load(policy_pooling_path, map_location="cpu")
                    )
        # Load Q readout: input dim scales with pooling mode (2× or 3×)
        q_readout = SimpleMLP(
            gnn_config.get("embedding_size") * q_pooling.output_multiplier,
            [gnn_config.get("embedding_size")] * 2,
            1,
            activation="mish",
        )
        if not isinstance(load_folder, dict):
            q_path = os.path.join(load_folder, "q_readout.pt")
            if os.path.exists(q_path):
                try:
                    q_readout.load_state_dict(torch.load(q_path, map_location="cpu"))
                except:
                    time.sleep(2.5)
                    q_readout.load_state_dict(torch.load(q_path, map_location="cpu"))

        # Load policy readout: same dim scaling as q_readout
        policy_readout = SimpleMLP(
            gnn_config.get("embedding_size") * policy_pooling.output_multiplier,
            [gnn_config.get("embedding_size")] * 2,
            1,
            activation="mish",
        )
        if not isinstance(load_folder, dict):
            policy_path = os.path.join(load_folder, "policy_readout.pt")
            if os.path.exists(policy_path):
                try:
                    policy_readout.load_state_dict(
                        torch.load(policy_path, map_location="cpu")
                    )
                except:
                    time.sleep(2.5)
                    policy_readout.load_state_dict(
                        torch.load(policy_path, map_location="cpu")
                    )

        return Model(
            encoder,
            q_gnn,
            policy_gnn,
            q_pooling,
            policy_pooling,
            q_readout,
            policy_readout,
            domain,
        )

    def save_to_folder(self, save_folder: str) -> None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        config = {}

        if self.domain is not None:
            config["domain_path"] = self.domain.filepath

        if self.encoder is not None:
            encoder_config = {}

            if isinstance(self.encoder, HorizonHGraphEncoder):
                encoder_config["type"] = "horizon"
            else:
                encoder_config["type"] = "default"

            encoder_config["symbol_type_id"] = self.encoder.symbol_type_id
            encoder_config["ignore_actions"] = self.encoder.ignore_actions

            config["encoder"] = encoder_config

        if self.q_gnn is not None:
            gnn_config = {}

            gnn_config["embedding_size"] = self.q_gnn.embedding_size
            gnn_config["num_layer"] = self.q_gnn.num_layer

            _atomic_torch_save(self.q_gnn.state_dict(), os.path.join(save_folder, "q_gnn.pt"))
            _atomic_torch_save(
                self.policy_gnn.state_dict(), os.path.join(save_folder, "policy_gnn.pt")
            )

            config["gnn"] = gnn_config

        if self.q_pooling is not None:
            pooling_config = {}

            pooling_config["parameters"] = any(
                p.requires_grad for p in self.q_pooling.parameters()
            )
            # Persist the ActionPoolingModule mode so readout shapes match on load.
            pooling_config["mode"] = getattr(self.q_pooling, "mode", "mask")

            if any(p.requires_grad for p in self.q_pooling.parameters()):
                _atomic_torch_save(
                    self.q_pooling.state_dict(),
                    os.path.join(save_folder, "q_pooling.pt"),
                )
            if any(p.requires_grad for p in self.policy_pooling.parameters()):
                _atomic_torch_save(
                    self.policy_pooling.state_dict(),
                    os.path.join(save_folder, "policy_pooling.pt"),
                )

            config["pooling"] = pooling_config

        if self.q_readout is not None:
            _atomic_torch_save(
                self.q_readout.state_dict(), os.path.join(save_folder, "q_readout.pt")
            )

        if self.policy_readout is not None:
            _atomic_torch_save(
                self.policy_readout.state_dict(),
                os.path.join(save_folder, "policy_readout.pt"),
            )

        # config.yaml is written last after all weight files; on the worker
        # side, if config.yaml exists with the new pooling mode but the
        # weight file is stale, the load_from_folder retry guard catches it.
        # Atomic via tmp+rename for the config as well.
        tmp_config = os.path.join(save_folder, "config.yaml.tmp")
        with open(tmp_config, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        os.replace(tmp_config, os.path.join(save_folder, "config.yaml"))


import mifrost
import reap.xmimir as xmi

mifrost.register_state_adapter(xmi.XState, lambda s: s.base)
mifrost.register_domain_adapter(xmi.XDomain, lambda d: d.advanced)
mifrost.register_literal_adapter(xmi.XLiteral, lambda d: d.advanced)
mifrost.register_action_adapter(xmi.XAction, lambda d: d.advanced)
