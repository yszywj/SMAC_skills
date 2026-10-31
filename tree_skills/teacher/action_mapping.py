"""P0-2: authoritative env-unit <-> teacher-action-slot mapping.

Verified against LOCAL SMACv2 source (smacv2/env/starcraft2/starcraft2.py,
get_avail_agent_actions) and the official MARL-GPT repo (no reordering —
model slots are the native action integers, wrapper passes them through):

  native action layout (10v10 terran_gen, conic_fov=False):
    0        no-op            (only meaningful for dead agents)
    1        stop
    2..5     move N/S/E/W
    6+t      target t:
               ground units  -> attack ENEMY t   (t in 0..n_enemies-1)
               medivac       -> heal ALLY t, allies only, medivacs excluded,
                                only ground allies within heal range
  n_actions = 6 + max(n_agents, n_enemies) = 16 for 10v10.
  The project's keep-heal-order extension (action id 16, one past native
  n_actions) is NOT a teacher action and must never enter teacher slots,
  logits means or labels.

The audited checkpoint's action head is 20 wide; slots 16..19 are
cross-environment padding, masked, never used for SMAC.
"""

N_ACTIONS_MOVE = 4
OFFSET_TARGETS = 6  # stop(1) + move(4) come first; targets start at 6


class ActionMappingError(ValueError):
    pass


class SmacActionMap:
    """Explicit bidirectional map for one terran 10v10 episode layout."""

    def __init__(self, n_agents=10, n_enemies=10, n_medivac_slots=10,
                 teacher_action_size=20):
        self.n_agents = n_agents
        self.n_enemies = n_enemies
        self.native_n_actions = OFFSET_TARGETS + max(n_agents, n_enemies)
        if self.native_n_actions > teacher_action_size:
            raise ActionMappingError(
                f"env needs {self.native_n_actions} actions but the teacher "
                f"head is {teacher_action_size} wide; refusing to remap")
        self.teacher_action_size = teacher_action_size
        # heal slots: medivac targeting ally j -> action 6+j (ground allies
        # only; env mask additionally excludes other medivacs and out-of-range)
        self.heal_slot_of_ally = {j: OFFSET_TARGETS + j for j in range(n_agents)}
        # attack slots for ground units: enemy e -> action 6+e
        self.attack_slot_of_enemy = {e: OFFSET_TARGETS + e for e in range(n_enemies)}

    def heal_slot(self, ally_index):
        """Teacher slot for 'this medivac heals ally <ally_index>'."""
        if ally_index not in self.heal_slot_of_ally:
            raise ActionMappingError(f"ally index {ally_index} out of range")
        return self.heal_slot_of_ally[ally_index]

    def ally_of_heal_slot(self, slot):
        for ally, s in self.heal_slot_of_ally.items():
            if s == slot:
                return ally
        raise ActionMappingError(f"slot {slot} is not a heal slot")

    def is_native_slot(self, slot):
        return 0 <= slot < self.native_n_actions

    def is_padding_slot(self, slot):
        return self.native_n_actions <= slot < self.teacher_action_size

    def keep_action_id(self):
        """The project's keep extension id: one past native n_actions."""
        return self.native_n_actions

    def validate_teacher_slots(self, slots):
        """Reject padding/keep/out-of-head slots before any label math."""
        keep = self.keep_action_id()
        for slot in slots:
            if slot == keep:
                raise ActionMappingError(
                    f"slot {slot} is the project keep-heal-order extension, "
                    "not a teacher action")
            if not self.is_native_slot(slot):
                raise ActionMappingError(
                    f"slot {slot} outside native action range "
                    f"0..{self.native_n_actions - 1}")
        return True
