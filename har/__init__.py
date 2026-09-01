"""Human Activity Recognition on the UCI HAR dataset.

Modules:
    data            -- dataset download and loading
    train           -- model comparison under random vs subject-wise CV
    features        -- redundancy analysis and feature-selection curve
    two_stage       -- hierarchical static/dynamic classifier
    error_analysis  -- confusion structure and per-subject error rates
    deep            -- 1D CNN on raw inertial signals (requires torch)
"""

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# Activities 1-3 involve motion; 4-6 are static postures.
DYNAMIC = (1, 2, 3)
STATIC = (4, 5, 6)
