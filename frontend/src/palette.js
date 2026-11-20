// Mirror of theme.css tokens for SVG attributes (presentation attributes cannot read CSS variables).
export const SLOT = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300'];
export const SEV = { none: '#0ca30c', sev2: '#fab219', sev1: '#d03b3b' };
export const STATUS = { admissible: '#0ca30c', excluded: '#d03b3b', unavailable: '#8f8e85' };
export const SURFACE = '#1a1a19';
export const GRID = '#2b2a27';
export const INK = '#f0efec';
export const INK2 = '#c3c2b7';
export const INK3 = '#8f8e85';

// Arm and lane identity: each entity keeps its slot for the whole session (color follows the entity).
export const ARM_COLOR = {
  '1_unconstrained': SLOT[0],
  '2_finite_penalty': SLOT[1],
  '3_zero_budget_cmdp': SLOT[2],
  '4_runtime_only': SLOT[3],
  '5_constraint_aware_only': SLOT[4],
  '6_combined': SLOT[5],
  F1_no_update: SLOT[0],
  F2_stale_policy_with_new_filter: SLOT[1],
  F3_stale_policy_no_filter_after_switch: SLOT[2],
  F4_reoptimised_with_filter: SLOT[3],
  F5_cmdp_resolve_no_filter: SLOT[4],
};
export const LANE_COLOR = { icmdp: SLOT[0], cmdp: SLOT[1] };
