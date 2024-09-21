import numpy as np

from pipelines.video_anonymization.app.run_pipeline_v2 import (
    draw_face_anonymization_mask_with_mask,
)


def test_draw_face_anonymization_mask_with_mask():
    img = np.ones((5, 5, 3)).astype(np.uint8)
    mask_cls1 = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    mask_cls2 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    )
    mask = np.stack([mask_cls1, mask_cls2], axis=2).astype(np.uint8)
    vis_img_expect_ch0 = np.array(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ]
    )

    vis_img = draw_face_anonymization_mask_with_mask(img, mask)
    np.testing.assert_array_equal(vis_img[:, :, 0], vis_img_expect_ch0)
