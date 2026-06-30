from support_capacity_reliability.queueing.erlang import (
    erlang_a,
    erlang_c,
    required_agents_erlang_a,
)


def test_erlang_c_more_agents_reduce_wait():
    few = erlang_c(0.02, 1 / 300, 7)
    more = erlang_c(0.02, 1 / 300, 10)
    assert more.average_wait_seconds < few.average_wait_seconds


def test_erlang_a_more_agents_reduce_abandonment():
    few = erlang_a(0.03, 1 / 420, 1 / 240, 10)
    more = erlang_a(0.03, 1 / 420, 1 / 240, 16)
    assert more.abandonment_rate <= few.abandonment_rate


def test_required_agents_meets_or_exhausts_bound():
    result = required_agents_erlang_a(
        0.015,
        420,
        240,
        service_level_target=0.8,
        abandonment_target=0.15,
        service_level_seconds=120,
        max_agents=50,
    )
    assert 1 <= result.agents <= 50
