import unittest

from refcocog.grounding import iou, iou_reward, normalized_to_pixels, parse_box, source_bbox_to_pixels


class GroundingTests(unittest.TestCase):
    def test_parse_single_and_double_bracket_boxes(self):
        self.assertEqual(parse_box("[10, 20, 500, 600]"), [10.0, 20.0, 500.0, 600.0])
        self.assertEqual(parse_box("[[10, 20, 500, 600]]"), [10.0, 20.0, 500.0, 600.0])

    def test_rejects_invalid_boxes(self):
        self.assertIsNone(parse_box("not a box"))
        self.assertIsNone(parse_box("[0, 0, 1001, 1000]"))
        self.assertIsNone(parse_box("[10, 20, 5, 600]"))

    def test_iou_for_perfect_and_disjoint_boxes(self):
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)

    def test_normalized_box_converts_to_image_pixels(self):
        self.assertEqual(normalized_to_pixels([0, 100, 500, 1000], 640, 480), [0.0, 48.0, 320.0, 480.0])

    def test_reward_uses_pixel_iou_and_penalizes_invalid_output(self):
        self.assertEqual(iou_reward("[0, 0, 1000, 1000]", [0, 0, 640, 480], 640, 480), 1.0)
        self.assertEqual(iou_reward("no box", [0, 0, 640, 480], 640, 480), -0.1)

    def test_source_box_maps_to_cached_image_coordinates(self):
        self.assertEqual(source_bbox_to_pixels([0, 0, 640, 438], 640, 438, 639, 437),
                         [0.0, 0.0, 639.0, 437.0])


if __name__ == "__main__":
    unittest.main()
