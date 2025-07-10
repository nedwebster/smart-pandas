from itertools import combinations


def test_compatability(column_tags):
    for column_tag_pair in combinations(column_tags, 2):
        if (column_tag_pair[0] in column_tag_pair[1].compatible_with) and (
            column_tag_pair[1] not in column_tag_pair[0].compatible_with
        ):
            raise ValueError(
                f"Tags {column_tag_pair[0].name} and {column_tag_pair[1].name} have asymmetric compatibility"
            )
