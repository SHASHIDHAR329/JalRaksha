def check_trust(
    breach_width_m,
    min_training_breach_width_m,
    max_training_breach_width_m,
    min_depth_m,
    max_depth_m,
    validation_error=None,
    max_allowed_error=0.30,
):
    """
    Rule-based trust engine.

    Checks:
    1. Whether breach width is inside the training range.
    2. Whether predicted depth is physically reasonable.
    3. Whether validation error is acceptable.

    Returns:
        confidence level and reasons.
    """

    reasons = []
    confidence = "HIGH"

    # ---------------------------------------------------------
    # 1. Check breach width against training range
    # ---------------------------------------------------------

    if breach_width_m < min_training_breach_width_m:
        confidence = "LOW"
        reasons.append(
            "Breach width is below the model training range."
        )

    elif breach_width_m > max_training_breach_width_m:
        confidence = "LOW"
        reasons.append(
            "Breach width is above the model training range."
        )


    # ---------------------------------------------------------
    # 2. Check for negative water depth
    # ---------------------------------------------------------

    if min_depth_m < 0:
        confidence = "LOW"
        reasons.append(
            "Prediction contains negative water depth."
        )


    # ---------------------------------------------------------
    # 3. Check unusually large flood depth
    # ---------------------------------------------------------

    if max_depth_m > 10:

        if confidence != "LOW":
            confidence = "MEDIUM"

        reasons.append(
            "Maximum predicted flood depth is unusually large."
        )


    # ---------------------------------------------------------
    # 4. Check validation error
    # ---------------------------------------------------------

    if validation_error is not None:

        if validation_error > max_allowed_error:
            confidence = "LOW"

            reasons.append(
                "Model validation error is above the allowed threshold."
            )


    # ---------------------------------------------------------
    # 5. No warnings
    # ---------------------------------------------------------

    if not reasons:
        reasons.append(
            "Scenario is within the configured limits and "
            "no warning was triggered."
        )


    return {
        "confidence": confidence,
        "reasons": reasons,
    }


def check_scenario_range(
    breach_width_m,
    min_breach_width_m,
    max_breach_width_m,
):
    """
    Check whether the breach width is inside
    the range represented in the training data.
    """

    if breach_width_m < min_breach_width_m:
        return {
            "status": "OUT_OF_RANGE",
            "message": "Breach width is below the training range.",
        }

    if breach_width_m > max_breach_width_m:
        return {
            "status": "OUT_OF_RANGE",
            "message": "Breach width is above the training range.",
        }

    return {
        "status": "IN_RANGE",
        "message": "Breach width is inside the training range.",
    }