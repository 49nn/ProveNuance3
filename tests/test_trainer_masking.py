from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch
from torch_geometric.data import HeteroData

from data_model.cluster import ClusterSchema
from nn.config import NNConfig
from nn.gating import ExceptionGateBank
from nn.graph_builder import GraphNodeIndex
from nn.message_passing import HeteroMessagePassingBank
from nn.proposer import NeuralProposer
from nn.trainer import ProposerTrainer


def test_train_on_case_masks_clamped_cluster_logits() -> None:
    schema = ClusterSchema(
        cluster_id=1,
        name="customer_type",
        entity_type="CUSTOMER",
        domain=["CONSUMER", "BUSINESS"],
        entity_role="CUSTOMER",
        value_role="VALUE",
    )
    config = NNConfig(T=1, beta_sparse=0.0)
    proposer = NeuralProposer(
        config=config,
        mp_bank=HeteroMessagePassingBank([]),
        gate_bank=ExceptionGateBank(gate_specs=[]),
        cluster_type_dims={schema.name: schema.dim},
    )
    trainer = ProposerTrainer(
        proposer=proposer,
        cluster_schemas=[schema],
        config=config,
        seed=42,
    )

    data = HeteroData()
    data["c_customer_type"].x = torch.tensor([[10.0, -10.0]])
    data["c_customer_type"].memory_bias = torch.zeros(1, 2)
    data["c_customer_type"].is_clamped = torch.tensor([True])
    data["c_customer_type"].clamp_hard = torch.tensor([True])

    original_x = data["c_customer_type"].x.clone()

    metrics = trainer.train_on_case(data, GraphNodeIndex())

    assert metrics["L_mask"] == pytest.approx(math.log(2.0), rel=1e-3)
    assert metrics["L_total"] == pytest.approx(metrics["L_mask"], rel=1e-6)
    assert torch.equal(data["c_customer_type"].x, original_x)
    assert data["c_customer_type"].is_clamped.tolist() == [True]
    assert data["c_customer_type"].clamp_hard.tolist() == [True]
