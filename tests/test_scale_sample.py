from scripts.validate_scale_sample import CLIPS, PRIOR_SAMPLES, select_disjoint_frames


ANNOTATION_COUNTS = {"crosswalk": 378, "fourway": 1281, "night": 565}


def test_disjoint_selection_is_unique_spread_and_avoids_both_prior_samples():
    for clip in CLIPS:
        selected = select_disjoint_frames(
            ANNOTATION_COUNTS[clip],
            PRIOR_SAMPLES[clip],
        )

        assert len(selected) == 20
        assert len(set(selected)) == 20
        assert not set(selected) & PRIOR_SAMPLES[clip]
        assert selected == sorted(selected)
        assert selected[0] > 1
        assert selected[-1] < ANNOTATION_COUNTS[clip]
