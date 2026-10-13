"""Warning-only checks for annotation geometry against decoded frames."""

import warnings


MAX_ANNOTATION_AREA_FRACTION = 0.10


def warn_if_extreme_annotation(
    frame_number,
    box_xyxy,
    frame_shape,
    *,
    source="annotations",
):
    """Warn on out-of-frame boxes or boxes over 10% frame area; never mutate them."""
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = box_xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    frame_area = frame_width * frame_height
    box_area_fraction = (
        (box_width * box_height) / frame_area
        if frame_area > 0
        else 0.0
    )

    reasons = []
    if box_width <= 0 or box_height <= 0:
        reasons.append("non-positive dimensions")
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        reasons.append("coordinates extend outside the frame")
    if (
        box_width > frame_width
        or box_height > frame_height
        or box_area_fraction > MAX_ANNOTATION_AREA_FRACTION
    ):
        reasons.append(
            f"extreme size ({box_width}x{box_height}, "
            f"{box_area_fraction:.1%} of frame area; "
            f"warning threshold is >{MAX_ANNOTATION_AREA_FRACTION:.0%})"
        )

    if reasons:
        warnings.warn(
            f"{source}: annotation row/frame {frame_number}, "
            f"XYXY box={tuple(box_xyxy)}: {'; '.join(reasons)}. "
            "The annotation is retained unchanged.",
            UserWarning,
            stacklevel=2,
        )
