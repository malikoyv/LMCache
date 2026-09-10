# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.admission_policy import (
    EphemeralWriteVeto,
    ReuseAdmissionPolicy,
    VetoMode,
    create_admission_policy,
    create_ephemeral_veto,
)
from lmcache.v1.storage_backend.storage_manager import select_admitted_objects


def create_test_key(key_id: int) -> CacheEngineKey:
    """Create a distinct CacheEngineKey for the given id."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=key_id,
        dtype=torch.bfloat16,
    )


def offer(policy: ReuseAdmissionPolicy, key: CacheEngineKey) -> bool:
    """Offer a single key to the policy and report whether it was admitted."""
    return key in policy.admit([key])


class TestReuseAdmissionPolicy:
    def test_disabled_admits_everything(self):
        policy = ReuseAdmissionPolicy(min_reuse=0)

        keys = [create_test_key(i) for i in range(4)]

        assert policy.guarded_backends == frozenset()
        assert policy.admit(keys) == set(keys)
        assert policy.admit(keys) == set(keys)

    def test_first_offer_is_rejected_second_is_admitted(self):
        policy = ReuseAdmissionPolicy(min_reuse=1)

        key = create_test_key(1)

        assert not offer(policy, key)
        assert offer(policy, key)
        assert offer(policy, key)

    def test_min_reuse_controls_number_of_offers_before_admission(self):
        policy = ReuseAdmissionPolicy(min_reuse=3)

        key = create_test_key(1)

        assert [offer(policy, key) for _ in range(5)] == [
            False,
            False,
            False,
            True,
            True,
        ]

    def test_single_use_keys_are_never_admitted(self):
        policy = ReuseAdmissionPolicy(min_reuse=1)

        single_use = [create_test_key(i) for i in range(1000)]

        assert policy.admit(single_use) == set()
        assert policy.stats().written == 0
        assert policy.stats().offered == 1000

    def test_reused_keys_are_admitted_while_single_use_keys_are_not(self):
        policy = ReuseAdmissionPolicy(min_reuse=1)

        reused = [create_test_key(i) for i in range(10)]
        single_use = [create_test_key(1000 + i) for i in range(90)]

        policy.admit(reused + single_use)
        admitted = policy.admit(reused)

        assert admitted == set(reused)

    def test_counters_are_tracked_per_key(self):
        policy = ReuseAdmissionPolicy(min_reuse=1)

        hot = create_test_key(1)
        cold = create_test_key(2)

        policy.admit([hot])
        admitted = policy.admit([hot, cold])

        assert admitted == {hot}

    def test_stats_report_write_reduction(self):
        policy = ReuseAdmissionPolicy(min_reuse=1)

        keys = [create_test_key(i) for i in range(10)]
        policy.admit(keys)
        policy.admit(keys)

        stats = policy.stats()

        assert stats.offered == 20
        assert stats.written == 10
        assert stats.suppressed == 10
        assert stats.write_fraction == pytest.approx(0.5)

    def test_write_fraction_defaults_to_one_when_nothing_offered(self):
        assert ReuseAdmissionPolicy(min_reuse=1).stats().write_fraction == 1.0

    def test_bounded_history_forgets_evicted_keys(self):
        # A two-slot table cannot remember 1000 distinct keys, so none of them
        # should survive long enough to be admitted on a second pass.
        policy = ReuseAdmissionPolicy(min_reuse=1, history_size=2)

        keys = [create_test_key(i) for i in range(1000)]
        policy.admit(keys)

        assert len(policy.admit(keys)) < len(keys)

    def test_guarded_backends_are_configurable(self):
        policy = ReuseAdmissionPolicy(
            min_reuse=1, guarded_backends=["LocalDiskBackend", "GdsBackend"]
        )

        assert policy.guarded_backends == frozenset({"LocalDiskBackend", "GdsBackend"})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_reuse": -1},
            {"min_reuse": 255},
            {"min_reuse": 1, "history_size": 0},
        ],
    )
    def test_invalid_settings_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ReuseAdmissionPolicy(**kwargs)

    @pytest.mark.parametrize(
        "backend", ["PDBackend", "P2PBackend", "LocalCPUBackend", "MaruBackend"]
    )
    def test_transfer_and_staging_backends_cannot_be_guarded(self, backend):
        with pytest.raises(ValueError, match=backend):
            ReuseAdmissionPolicy(min_reuse=1, guarded_backends=[backend])


class TestCreateAdmissionPolicy:
    def test_disabled_by_default(self):
        config = LMCacheEngineConfig.from_defaults(chunk_size=256)

        assert create_admission_policy(config).guarded_backends == frozenset()

    def test_enabled_via_extra_config(self):
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={"admission_min_reuse": 2},
        )

        policy = create_admission_policy(config)
        key = create_test_key(1)

        assert policy.guarded_backends == frozenset({"LocalDiskBackend", "GdsBackend"})
        assert [offer(policy, key) for _ in range(3)] == [False, False, True]

    def test_guarded_backends_from_extra_config(self):
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={
                "admission_min_reuse": 1,
                "admission_backends": ["GdsBackend"],
            },
        )

        assert create_admission_policy(config).guarded_backends == frozenset(
            {"GdsBackend"}
        )

    def test_non_string_backend_names_are_rejected(self):
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={
                "admission_min_reuse": 1,
                "admission_backends": [42],
            },
        )

        with pytest.raises(ValueError):
            create_admission_policy(config)

    def test_transfer_backend_is_rejected(self):
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={
                "admission_min_reuse": 1,
                "admission_backends": ["PDBackend"],
            },
        )

        with pytest.raises(ValueError, match="PDBackend"):
            create_admission_policy(config)

    def test_bare_string_backend_is_rejected(self):
        # "LocalDiskBackend" would otherwise decompose into single characters
        # and silently guard nothing.
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={
                "admission_min_reuse": 1,
                "admission_backends": "LocalDiskBackend",
            },
        )

        with pytest.raises(ValueError, match="admission_backends"):
            create_admission_policy(config)

    @pytest.mark.parametrize("value", ["2", 1.5, None, True])
    def test_non_integer_min_reuse_is_rejected(self, value):
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={"admission_min_reuse": value},
        )

        with pytest.raises(ValueError, match="admission_min_reuse"):
            create_admission_policy(config)


class TestEphemeralWriteVeto:
    def test_off_never_denies_and_records_nothing(self):
        veto = EphemeralWriteVeto(mode=VetoMode.OFF, min_ttl_s=0)

        assert veto.guarded_backends == frozenset()
        assert not veto.evaluate_put({"lmcache.kv_ttl_s": 1}, 4)
        assert veto.stats().puts == 0

    def test_short_lived_request_is_denied(self):
        veto = EphemeralWriteVeto(mode=VetoMode.ENFORCE, min_ttl_s=300)

        assert veto.evaluate_put({"lmcache.kv_ttl_s": 60}, 4)

    def test_long_lived_request_is_stored(self):
        veto = EphemeralWriteVeto(mode=VetoMode.ENFORCE, min_ttl_s=300)

        assert not veto.evaluate_put({"lmcache.kv_ttl_s": 3600}, 4)

    def test_threshold_is_exclusive(self):
        veto = EphemeralWriteVeto(mode=VetoMode.ENFORCE, min_ttl_s=300)

        assert not veto.evaluate_put({"lmcache.kv_ttl_s": 300}, 1)
        assert veto.evaluate_put({"lmcache.kv_ttl_s": 299}, 1)

    def test_observe_counts_without_suppressing(self):
        veto = EphemeralWriteVeto(mode=VetoMode.OBSERVE, min_ttl_s=300)

        assert not veto.evaluate_put({"lmcache.kv_ttl_s": 60}, 10)
        assert not veto.evaluate_put({"lmcache.kv_ttl_s": 3600}, 30)
        assert veto.guarded_backends == frozenset()

        stats = veto.stats()

        assert stats.mode is VetoMode.OBSERVE
        assert stats.chunks == 40
        assert stats.matched_chunks == 10
        assert stats.matched_chunk_fraction == pytest.approx(0.25)

    @pytest.mark.parametrize(
        "request_configs",
        [
            None,
            {},
            {"lmcache.skip_save": True},
            {"lmcache.kv_ttl_s": "not-a-number"},
            {"lmcache.kv_ttl_s": -1},
            {"lmcache.kv_ttl_s": True},
            {"lmcache.kv_ttl_s": 1.5},
        ],
    )
    def test_undeclared_or_malformed_fails_open(self, request_configs):
        veto = EphemeralWriteVeto(mode=VetoMode.ENFORCE, min_ttl_s=300)

        assert not veto.evaluate_put(request_configs, 4)

    def test_string_declaration_is_accepted(self):
        # request_configs survives the lookup-client wire format as strings.
        veto = EphemeralWriteVeto(mode=VetoMode.ENFORCE, min_ttl_s=300)

        assert veto.evaluate_put({"lmcache.kv_ttl_s": "60"}, 4)

    def test_tag_prefix_is_not_used(self):
        # "lmcache.tag." keys are folded into CacheEngineKey.tags, which would
        # fork the namespace per class and destroy cross-request sharing.
        veto = EphemeralWriteVeto(mode=VetoMode.ENFORCE, min_ttl_s=300)

        assert not veto.evaluate_put({"lmcache.tag.kv_ttl_s": 60}, 4)

    def test_stats_count_puts_chunks_and_malformed(self):
        veto = EphemeralWriteVeto(mode=VetoMode.ENFORCE, min_ttl_s=300)

        veto.evaluate_put({"lmcache.kv_ttl_s": 60}, 10)
        veto.evaluate_put({"lmcache.kv_ttl_s": 3600}, 20)
        veto.evaluate_put({"lmcache.kv_ttl_s": "oops"}, 5)
        veto.evaluate_put(None, 5)

        stats = veto.stats()

        assert stats.puts == 4
        assert stats.chunks == 40
        assert stats.matched_puts == 1
        assert stats.matched_chunks == 10
        assert stats.malformed == 1
        assert stats.matched_chunk_fraction == pytest.approx(0.25)

    def test_stats_fraction_is_zero_when_nothing_offered(self):
        veto = EphemeralWriteVeto(mode=VetoMode.ENFORCE, min_ttl_s=300)

        assert veto.stats().matched_chunk_fraction == 0.0

    def test_negative_threshold_is_rejected(self):
        with pytest.raises(ValueError):
            EphemeralWriteVeto(mode=VetoMode.OFF, min_ttl_s=-1)

    @pytest.mark.parametrize("mode", [VetoMode.OBSERVE, VetoMode.ENFORCE])
    def test_active_mode_requires_a_threshold(self, mode):
        with pytest.raises(ValueError, match="admission_flash_min_ttl_s"):
            EphemeralWriteVeto(mode=mode, min_ttl_s=0)

    @pytest.mark.parametrize(
        "backend", ["PDBackend", "P2PBackend", "LocalCPUBackend", "MaruBackend"]
    )
    def test_transfer_and_staging_backends_cannot_be_vetoed(self, backend):
        with pytest.raises(ValueError, match=backend):
            EphemeralWriteVeto(
                mode=VetoMode.ENFORCE, min_ttl_s=300, guarded_backends=[backend]
            )


class TestCreateEphemeralVeto:
    def test_disabled_by_default(self):
        config = LMCacheEngineConfig.from_defaults(chunk_size=256)

        veto = create_ephemeral_veto(config)

        assert veto.mode is VetoMode.OFF
        assert veto.guarded_backends == frozenset()

    def test_enabled_via_extra_config(self):
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={
                "admission_flash_veto_mode": "enforce",
                "admission_flash_min_ttl_s": 300,
            },
        )

        veto = create_ephemeral_veto(config)

        assert veto.guarded_backends == frozenset({"LocalDiskBackend", "GdsBackend"})
        assert veto.evaluate_put({"lmcache.kv_ttl_s": 60}, 1)

    def test_observe_mode_from_extra_config(self):
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={
                "admission_flash_veto_mode": "observe",
                "admission_flash_min_ttl_s": 300,
            },
        )

        veto = create_ephemeral_veto(config)

        assert veto.mode is VetoMode.OBSERVE
        assert not veto.evaluate_put({"lmcache.kv_ttl_s": 60}, 1)
        assert veto.stats().matched_chunks == 1

    def test_shares_admission_backends_setting(self):
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={
                "admission_flash_veto_mode": "enforce",
                "admission_flash_min_ttl_s": 300,
                "admission_backends": ["LocalDiskBackend"],
            },
        )

        assert create_ephemeral_veto(config).guarded_backends == frozenset(
            {"LocalDiskBackend"}
        )

    def test_independent_of_the_reuse_filter(self):
        # The veto must work with admission_min_reuse left at its default 0.
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={
                "admission_flash_veto_mode": "enforce",
                "admission_flash_min_ttl_s": 300,
            },
        )

        assert create_admission_policy(config).guarded_backends == frozenset()
        assert create_ephemeral_veto(config).guarded_backends != frozenset()

    def test_unknown_mode_is_rejected(self):
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={"admission_flash_veto_mode": "maybe"},
        )

        with pytest.raises(ValueError, match="admission_flash_veto_mode"):
            create_ephemeral_veto(config)

    @pytest.mark.parametrize("value", ["300", 1.5, None, True])
    def test_non_integer_threshold_is_rejected(self, value):
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            extra_config={"admission_flash_min_ttl_s": value},
        )

        with pytest.raises(ValueError, match="admission_flash_min_ttl_s"):
            create_ephemeral_veto(config)


class TestSelectAdmittedObjects:
    def test_keeps_only_admitted_pairs(self):
        keys = [create_test_key(i) for i in range(4)]
        objs = [f"obj{i}" for i in range(4)]

        selected_keys, selected_objs = select_admitted_objects(
            keys, objs, {keys[1], keys[3]}
        )

        assert selected_keys == [keys[1], keys[3]]
        assert selected_objs == ["obj1", "obj3"]

    def test_returns_inputs_unchanged_when_all_admitted(self):
        keys = [create_test_key(i) for i in range(3)]
        objs = [f"obj{i}" for i in range(3)]

        selected_keys, selected_objs = select_admitted_objects(keys, objs, set(keys))

        assert selected_keys is keys
        assert selected_objs is objs

    def test_returns_empty_when_nothing_admitted(self):
        keys = [create_test_key(i) for i in range(3)]
        objs = [f"obj{i}" for i in range(3)]

        selected_keys, selected_objs = select_admitted_objects(keys, objs, set())

        assert selected_keys == []
        assert selected_objs == []
