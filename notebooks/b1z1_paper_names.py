"""Paper labels are presentation only; implementation task IDs never change."""

_METHODS = {
    "b1z1_pact": ("PILOT", r"\pilot"),
    "b1z1_ppo_pos": ("Pos", r"\posppo"),
    "b1z1_unifp_original": ("UniFP", r"\unifp"),
    "b1z1_pact_ab4_coupled_none": ("A4", r"\pilotnofilm"),
    "b1z1_pact_ab5_coupled_concat": ("A5", r"\pilotconcat"),
    "b1z1_pact_ab6_position_film": ("A1", r"\pilotposnopinn"),
    "b1z1_pact_ab7_position_full": ("A2", r"\pilotpospinn"),
    "b1z1_pact_ab8_coupled_film": ("A3", r"\pilotnopinn"),
    "b1z1_pact_ab9_representation": ("A6", r"\pilotencpinn"),
    "b1z1_pact_ab10_actor": ("A7", r"\pilotactpinn"),
}
DISPLAY_NAMES = {task: pair[0] for task, pair in _METHODS.items()}
LATEX_METHOD_NAMES = {task: pair[1] for task, pair in _METHODS.items()}
