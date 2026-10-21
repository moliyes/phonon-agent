from phonon_agent.entropy import recommended_count


def test_raw_95_percent_recommendation_handles_non_monotonic_curve():
    assert recommended_count([0.0, 0.91, 1.0, 0.94], 0.95) == 3
    assert recommended_count([1.0, 2.0, 1.9, 1.8], 0.95) == 2
    assert recommended_count([0.0, 0.96, 1.0], 0.95, [1, 5, 10]) == 5
