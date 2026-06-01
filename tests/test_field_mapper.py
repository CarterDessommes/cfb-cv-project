from field_mapper import CANVAS_SCALE, FIELD_WIDTH, field_to_canvas_point


def test_field_to_canvas_point_flips_y_across_x_axis():
    bottom_y = int(FIELD_WIDTH * CANVAS_SCALE) - 1

    assert field_to_canvas_point(25.0, 0.0) == (200, bottom_y)
    assert field_to_canvas_point(25.0, FIELD_WIDTH) == (200, 0)


def test_field_to_canvas_point_preserves_field_x_mapping():
    left = field_to_canvas_point(10.0, 20.0)
    right = field_to_canvas_point(20.0, 20.0)

    assert left[0] == 80
    assert right[0] == 160
    assert left[1] == right[1]


def test_field_to_canvas_point_can_skip_x_axis_flip():
    assert field_to_canvas_point(25.0, 0.0, flip_x_axis=False) == (200, 0)
