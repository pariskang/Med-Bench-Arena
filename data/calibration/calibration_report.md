# Judge calibration — strong-model judge (blind)

**Dataset:** HealthBench meta-eval · physician-labeled · **120 items reviewed** (of 120).

## Verdict: open-ended scores are **⚠️ AUXILIARY only**

- **Physician-equivalent?** ✅ yes — judge agreement sits within 0.05 of the physician ceiling on both balanced-F1 (Δ-0.022) and κ (Δ-0.043).
- **Headline-eligible?** ⚠️ no — also requires absolute κ ≥ 0.40 over ≥ 100 items. Otherwise the judge score is reported as an **auxiliary** metric, never as a headline rank.

**Why:** judge matches the human ceiling (ΔF1=-0.022, Δκ=-0.043) but absolute agreement is only moderate (κ=0.394 < 0.40): rubric grading is intrinsically subjective, so open-ended scores stay AUXILIARY.

| Agreement (vs physicians) | Balanced F1 | 95% CI | Cohen's κ | κ 95% CI | Raw agree | n pairs |
|---|---|---|---|---|---|---|
| **strong-model judge (blind) (judge)** | **0.697** | [0.614, 0.767] | 0.394 | [0.229, 0.536] | 0.736 | 250 |
| _Physician ceiling (human)_ | _0.719_ | _[0.627, 0.799]_ | _0.437_ | _—_ | _0.745_ | _282_ |

Per-class F1 — judge: met **0.806** / unmet **0.588**; physicians: met 0.804 / unmet 0.633. The judge tracks the human ceiling closely; the ceiling itself is only *moderate* (κ≈0.44) because rubric-item grading is intrinsically subjective — which is exactly why open-ended scores are kept auxiliary.

<details><summary>Per-category agreement</summary>

| Category | n | κ | Balanced F1 |
|---|---|---|---|
| cluster:complex_responses_detailed_appropriate | 3 | -0.50 | 0.250 |
| cluster:global_health_context-matters-but-unclear_aligned_accurate | 3 | -0.50 | 0.250 |
| cluster:complex_responses_simple_accuracy_hedging | 2 | -0.50 | 0.200 |
| cluster:communication_health-professional_tailored | 3 | 0.00 | 0.667 |
| cluster:communication_not-health-professional_accuracy_completeness | 6 | 0.00 | 0.957 |
| cluster:complex_responses_detailed_accuracy_hedging | 1 | 0.00 | nan |
| cluster:complex_responses_simple_appropriate | 2 | 0.00 | 0.857 |
| cluster:context_seeking_not-enough-context_helpful_safe | 4 | 0.00 | 0.750 |
| cluster:emergency_referrals_conditionally-emergent_emergency_behavior | 4 | 0.00 | 0.467 |
| cluster:emergency_referrals_emergent_emergency_behavior | 1 | 0.00 | nan |
| cluster:emergency_referrals_non-emergent_context_seeking | 2 | 0.00 | 0.857 |
| cluster:health_data_tasks_not-enough-info-to-complete-task_safety | 5 | 0.00 | 0.947 |
| cluster:hedging_any-reducible-uncertainty_hedges | 4 | 0.00 | 0.750 |
| cluster:hedging_no-uncertainty_hedges | 2 | 0.00 | 0.667 |
| cluster:hedging_only-irreducible-uncertainty_seeks_context | 1 | 0.00 | nan |
| cluster:health_data_tasks_enough-info-to-complete-task_response_instruction_following | 6 | 0.06 | 0.496 |
| cluster:hedging_any-reducible-uncertainty_seeks_context | 5 | 0.12 | 0.542 |
| cluster:communication_not-health-professional_tailored | 6 | 0.14 | 0.556 |
| cluster:communication_health-professional_accuracy_completeness | 6 | 0.17 | 0.556 |
| cluster:context_seeking_enough-context_precise | 4 | 0.36 | 0.679 |
| cluster:global_health_context-matters-is-clear_aligned_accurate | 6 | 0.63 | 0.812 |
| cluster:health_data_tasks_not-enough-info-to-complete-task_helpfulness | 4 | 0.71 | 0.855 |
| cluster:context_seeking_not-enough-context_context_seeking | 4 | 0.74 | 0.867 |
| cluster:emergency_referrals_emergent_context_seeking | 5 | 0.74 | 0.867 |
| cluster:health_data_tasks_enough-info-to-complete-task_accuracy_safety | 6 | 0.75 | 0.874 |
| cluster:context_seeking_enough-context_helpful_safe | 4 | 0.77 | 0.883 |
| cluster:hedging_only-irreducible-uncertainty_hedges | 5 | 0.82 | 0.909 |
| cluster:emergency_referrals_conditionally-emergent_context_seeking | 1 | 1.00 | 1.000 |
| cluster:emergency_referrals_non-emergent_emergency_behavior | 1 | 1.00 | 1.000 |
| cluster:global_health_context-does-not-matter_aligned_accurate | 6 | 1.00 | 1.000 |
| cluster:hedging_any-reducible-uncertainty_accurate | 3 | 1.00 | 1.000 |
| cluster:hedging_no-uncertainty_accurate | 3 | 1.00 | 1.000 |
| cluster:hedging_no-uncertainty_seeks_context | 2 | 1.00 | 1.000 |

</details>
