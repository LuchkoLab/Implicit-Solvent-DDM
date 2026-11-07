"""Unit tests for packing several GPU windows onto one device."""
import pytest

from implicit_solvent_ddm.implicit_ddm_workflow import (
    _reserve_cores_for_gpu_slots,
    _slots_to_visible_devices,
)


@pytest.mark.parametrize("slot, device", [(0, "0"), (1, "1")])
def test_one_window_per_gpu_keeps_identity_mapping(slot, device):
    assert _slots_to_visible_devices([slot], ["0", "1"], 1) == device


@pytest.mark.parametrize("slot, device", [(0, "0"), (3, "0"), (4, "1"), (7, "1")])
def test_four_windows_per_gpu_share_a_device(slot, device):
    assert _slots_to_visible_devices([slot], ["0", "1"], 4) == device


def test_multi_slot_request_across_devices_lists_each_once():
    assert _slots_to_visible_devices([2, 3, 4, 5], ["0", "1"], 4) == "0,1"


def test_non_contiguous_allocation_maps_to_real_ids():
    assert _slots_to_visible_devices([0, 5], ["2", "5"], 4) == "2,5"


def test_reserve_cores_subtracts_slots():
    assert _reserve_cores_for_gpu_slots(90, 8) == 82


def test_reserve_cores_reads_slurm_when_unset(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "20")
    assert _reserve_cores_for_gpu_slots(None, 4) == 16


def test_reserve_cores_raises_when_none_left():
    with pytest.raises(ValueError):
        _reserve_cores_for_gpu_slots(4, 4)

