# User-reviewed phase calibration

60 original grayscale channel crops from six local flow settings, with 159 user-adjusted axial droplet intervals. The user explicitly confirmed frames with no droplet annotations are negative examples (17 frames). Labels omit machine-specific paths and hardware metadata. Images preserve original pixel values. This is calibration/regression data, not a held-out accuracy benchmark. Former carrier-gap rectangles sometimes overlap the revised droplet edges and are not imported as contradictory negative labels.

The regression gate checks approximate one-to-one localization, no unmatched predictions, all negative frames, and median length error. It does not claim every endpoint is accurate or the earlier proposed 90%-within-10% acceptance target has been achieved.
