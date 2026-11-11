import numpy as np
from contour_model.contact_sheets import endpoint_indices


def test_endpoints_use_instance_transitions_and_not_ring_seams():
    # Two same-class segments may have different IDs. Second ring is one circle.
    ids=np.array([1,1,2,2,1,3,3,3,3])
    np.testing.assert_array_equal(endpoint_indices(ids,[0,5,9]),[2,4])


def test_endpoint_at_actual_wrap_join():
    np.testing.assert_array_equal(endpoint_indices(np.array([1,1,2,2]),[0,4]),[0,2])
