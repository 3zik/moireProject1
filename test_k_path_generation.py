import numpy as np

def high_symmetry_path(num_k=60):
    # Moiré reciprocal lattice vectors (dimensionless)
    b1 = np.array([np.sqrt(3)/2, 1/2])
    b2 = np.array([0, 1])

    # High-symmetry points in moiré BZ
    Γ = np.array([0.0, 0.0])
    M = 0.5 * b1
    K = (2 * b1 + b2) / 3

    path = []
    for start, end in [(Γ, K), (K, M), (M, Γ)]:
        for t in np.linspace(0, 1, num_k, endpoint=False):
            path.append((1 - t) * start + t * end)
    path.append(Γ)
    return np.array(path)

# ----------------- Tests -----------------

def test_path_length():
    num_k = 60
    k_path = high_symmetry_path(num_k=num_k)
    expected_length = 3 * num_k + 1  # 3 segments each with num_k points + last Γ point
    assert len(k_path) == expected_length, f"Expected {expected_length} points, got {len(k_path)}"

def test_start_and_end_points():
    num_k = 60
    k_path = high_symmetry_path(num_k=num_k)
    np.testing.assert_array_almost_equal(k_path[0], np.array([0.0, 0.0]), decimal=10, err_msg="Start point is not Γ")
    np.testing.assert_array_almost_equal(k_path[-1], np.array([0.0, 0.0]), decimal=10, err_msg="End point is not Γ")

def test_segment_endpoints():
    num_k = 60
    k_path = high_symmetry_path(num_k=num_k)

    b1 = np.array([np.sqrt(3)/2, 1/2])
    b2 = np.array([0, 1])
    Γ = np.array([0.0, 0.0])
    M = 0.5 * b1
    K = (2 * b1 + b2) / 3

    # Start points of each segment
    seg_starts = [0, num_k, 2*num_k]
    expected_starts = [Γ, K, M]

    for idx, expected in zip(seg_starts, expected_starts):
        np.testing.assert_array_almost_equal(k_path[idx], expected, decimal=10)

    # The final appended Γ point (last point)
    np.testing.assert_array_almost_equal(k_path[-1], Γ, decimal=10)


def test_points_are_on_lines():
    # Check that points lie on straight lines between symmetry points by verifying linear interpolation
    num_k = 60
    k_path = high_symmetry_path(num_k=num_k)

    b1 = np.array([np.sqrt(3)/2, 1/2])
    b2 = np.array([0, 1])
    Γ = np.array([0.0, 0.0])
    M = 0.5 * b1
    K = (2 * b1 + b2) / 3

    segments = [(Γ, K), (K, M), (M, Γ)]

    for seg_idx, (start, end) in enumerate(segments):
        for i in range(num_k):
            idx = seg_idx*num_k + i
            t = i / num_k
            expected = (1 - t)*start + t*end
            np.testing.assert_array_almost_equal(k_path[idx], expected, decimal=10,
                                                 err_msg=f"Point {idx} not on line segment {seg_idx}")

if __name__ == "__main__":
    test_path_length()
    test_start_and_end_points()
    test_segment_endpoints()
    test_points_are_on_lines()
    print("All tests passed!")
