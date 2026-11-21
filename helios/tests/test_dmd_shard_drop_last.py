"""Regression test for DMD SP-group sharding with drop_last."""

from helios.dataset.dataloader_dmd import BucketedSampler


def _sampler(ith):
    s = object.__new__(BucketedSampler)
    s.num_sp_groups = 3
    s.ith_sp_group = ith
    s.drop_last = True
    return s


def test_shard_truncates_evenly_when_drop_last():
    # 10 indices do not divide evenly into 3 groups. drop_last must drop
    # the remainder so every group receives the same number of indices.
    shards = [_sampler(i)._shard_indices_for_sp_group(list(range(10))) for i in range(3)]

    assert shards == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
    assert len({len(s) for s in shards}) == 1
