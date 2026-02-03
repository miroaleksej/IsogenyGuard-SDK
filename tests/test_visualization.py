from isogenyguard.visualization import (
    quantize_linear_value,
    quantize_linear,
    quantize_uruz,
    heatmap_counts,
)


def test_quantize_linear_value_edges():
    max_range = 100
    bins = 10
    assert quantize_linear_value(0, max_range, bins) == 1
    assert quantize_linear_value(9, max_range, bins) == 1
    assert quantize_linear_value(10, max_range, bins) == 2
    assert quantize_linear_value(99, max_range, bins) == 10
    assert quantize_linear_value(100, max_range, bins) == 10
    assert quantize_linear_value(101, max_range, bins) == 10


def test_quantize_linear_vector():
    assert quantize_linear([0, 50, 99], 100, 10) == [1, 6, 10]


def test_heatmap_counts():
    uruz = [(0, 0), (99, 99), (50, 50)]
    grid = heatmap_counts(uruz, max_range=100, bins=10)
    assert grid[0][0] == 1
    assert grid[5][5] == 1
    assert grid[9][9] == 1


def test_quantize_uruz():
    uruz = [(0, 0), (99, 99)]
    assert quantize_uruz(uruz, max_range=100, bins=10) == [(1, 1), (10, 10)]
