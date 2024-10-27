from itertools import combinations


def test_compatability(column_labels):
    """Test that the compatibility relationships are symmetric."""
    for column_label_pair in combinations(column_labels, 2):
        if (column_label_pair[0] in column_label_pair[1].compatible_with) and (
            column_label_pair[1] not in column_label_pair[0].compatible_with
        ):
            assert False
