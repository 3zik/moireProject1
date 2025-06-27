# test_k_path_generation.py
import numpy as np
from k_path_generation import high_symmetry_path


def test_k_path_length_and_endpoints():
    num_k = 60
    k_path = high_symmetry_path(num_k=num_k)
    expected_points = 3 * num_k + 1
    assert k_path.shape == (expected_points, 2), "Unexpected number of k-points"
    np.testing.assert_array_almost_equal(k_path[0], np.array([0.0, 0.0]), decimal=8)
    np.testing.assert_array_almost_equal(k_path[-1], np.array([0.0, 0.0]), decimal=8)

def test_high_symmetry_points():
    k_path = high_symmetry_path(num_k=60)
    b1 = np.array([np.sqrt(3)/2, 1/2])
    b2 = np.array([0.0, 1.0])
    K = (2 * b1 + b2) / 3
    M = 0.5 * b1
    np.testing.assert_array_almost_equal(k_path[60], K, decimal=8)
    np.testing.assert_array_almost_equal(k_path[120], M, decimal=8)
    np.testing.assert_array_almost_equal(k_path[180], np.array([0.0, 0.0]), decimal=8)

def is_collinear(points, tol=1e-8):
    vec = points[-1] - points[0]
    vec_norm = vec / np.linalg.norm(vec)
    for p in points[1:-1]:
        vec_p = p - points[0]
        cross_mag = np.abs(vec_norm[0]*vec_p[1] - vec_norm[1]*vec_p[0])
        if cross_mag > tol:
            return False
    return True

def test_segments_are_collinear():
    k_path = high_symmetry_path(num_k=60)
    seg1 = k_path[0:61]
    seg2 = k_path[60:121]
    seg3 = k_path[120:181]
    assert is_collinear(seg1), "Γ->K segment points are not collinear"
    assert is_collinear(seg2), "K->M segment points are not collinear"
    assert is_collinear(seg3), "M->Γ segment points are not collinear"

def test_no_nans_or_duplicates():
    k_path = high_symmetry_path(num_k=60)
    assert not np.any(np.isnan(k_path)), "NaN detected in k-path"
    assert np.unique(k_path, axis=0).shape[0] == k_path.shape[0], "Duplicate points detected in k-path"

if __name__ == "__main__":
    test_k_path_length_and_endpoints()
    test_high_symmetry_points()
    test_segments_are_collinear()
    test_no_nans_or_duplicates()
    print("All k_path_generation tests passed!")
