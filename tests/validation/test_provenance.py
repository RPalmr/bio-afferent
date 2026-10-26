"""Tests for parameter provenance completeness."""

from __future__ import annotations

import pytest

from bioafferent import GTOConfig, SpindleConfig
from bioafferent.models.provenance import (
    GTO_PROVENANCE,
    SPINDLE_PROVENANCE,
    ProvenanceKind,
    check_provenance_complete,
    parameter_table,
)


@pytest.mark.parametrize("config,registry", [(SpindleConfig(), SPINDLE_PROVENANCE), (GTOConfig(), GTO_PROVENANCE)])
def test_every_field_has_provenance(config, registry):
    check_provenance_complete(config)
    table = parameter_table(config)
    assert {row["field"] for row in table} == set(registry)
    assert all(row["kind"] in {kind.value for kind in ProvenanceKind} for row in table)


def test_published_fields_are_limited_to_sourced_values():
    published = {name for name, item in SPINDLE_PROVENANCE.items() if item.kind is ProvenanceKind.PUBLISHED}
    assert published == {"tau_activation_dynamic", "tau_activation_static"}
    assert "Vannucci" in SPINDLE_PROVENANCE["tau_activation_dynamic"].source


def test_unsupported_config_type_is_rejected():
    with pytest.raises(TypeError):
        check_provenance_complete(object())  # type: ignore[arg-type]
