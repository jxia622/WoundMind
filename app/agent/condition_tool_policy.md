# Condition Tool Policy

This document tells the WoundMind agent which external tools are relevant for each classifier condition. The runtime source of truth is `condition_policy_registry.json` in the same folder.

The current implemented external tools are:

| Tool ID | Runtime tool | Status | Use |
| --- | --- | --- | --- |
| `open_wound_segmentation` | `segment_wound` | Implemented | Runs the current U-Net++ open wound segmentation model. |
| `relative_depth_map` | `depth_map` | Implemented | Runs the current relative depth map tool, optionally using the wound mask ROI. |

Texture and color cues are treated as features already available inside the condition classifier network. They are not listed as external tools unless a future standalone extractor is needed.

## Condition Routing Table

| Condition | Main severity axes | Required external tools now | Future / optional tools | Retrieval doc IDs |
| --- | --- | --- | --- | --- |
| Healthy / Normal Skin | None | None | None | None |
| Pressure Injury | Depth, tissue loss, exposed structures, necrosis | `open_wound_segmentation`, `relative_depth_map` | `tissue_exposure_classifier`, `necrosis_detector` | `pressure_injury_EPUAP_NPIAP_PPPIA_classification_2019` |
| Diabetic Foot Ulcer | Depth, infection, ischemia, gangrene | `open_wound_segmentation`, `relative_depth_map` | `tissue_exposure_classifier`, `necrosis_detector`, `perfusion_proxy` | `dfu_wagner_classification_cureus_2022`, `iwgdf_dfu_guideline` |
| Venous Leg Ulcer | Area, healing trajectory, exudate, edema | `open_wound_segmentation` | `serial_area_change`, `exudate_detection`, `swelling_estimator` | `vlu_wound_assessment_docs` |
| Atopic Dermatitis / Eczema | Area, erythema, excoriation, lichenification | None | `redness_estimator`, `body_area_segmentation`, `scratch_texture_detector` | `easi_scorad_references` |
| Contact Dermatitis | Erythema, vesicles, crusting, erosion | None | `redness_estimator`, `blister_crust_detector`, `affected_area_estimator` | `dermatitis_severity_references` |
| Seborrheic Dermatitis | Erythema, greasy scaling, distribution | None | `redness_estimator`, `scale_texture_analyzer`, `region_mapping` | `seborrheic_dermatitis_scoring_docs` |
| Acne | Lesion count, lesion type | None | `lesion_detector`, `papule_pustule_nodule_classifier` | `acne_severity_grading` |
| Hidradenitis Suppurativa | Abscesses, tunnels, scarring | None | `abscess_detector`, `sinus_tract_scar_detector` | `hurley_staging` |
| Sunburn | Erythema, blistering, peeling, area | None | `blister_detector`, `affected_area_estimator` | `burn_sunburn_severity_references` |
| Psoriasis | Area, erythema, scaling, thickness | None | `plaque_segmentation`, `redness_estimator`, `scale_texture_analyzer`, `thickness_proxy` | `pasi_bsa_scoring` |
| Bruising / Contusions | Area, color, swelling, temporal change | None | `color_analysis`, `affected_area_estimator`, `swelling_estimator`, `serial_comparison` | `contusion_assessment_references` |
| Melasma | Pigmentation intensity, area | None | `pigment_segmentation`, `color_index`, `facial_distribution_mapping` | `masi_mmasi_references` |
| Cold Injury | Perfusion loss, blistering, necrosis | None | `perfusion_proxy`, `blister_detector`, `necrosis_detector` | `frostbite_cold_injury_grading` |
| Rosacea | Erythema, papules/pustules, telangiectasia | None | `redness_estimator`, `lesion_detector`, `vessel_detector` | `rosacea_grading_references` |
| Surgical Wounds | Dehiscence, drainage, infection signs, necrosis | None | `gap_width_estimator`, `incision_segmentation`, `redness_estimator`, `exudate_detection`, `necrosis_detector` | `surgical_wound_assessment_docs` |
| Radiation Injury | Erythema, desquamation, ulceration, necrosis | None | `redness_estimator`, `desquamation_detector`, `ulcer_necrosis_detector` | `ctcae_radiation_dermatitis` |
| Burn Wounds | Burn depth, TBSA, perfusion, necrosis | None | `burn_segmentation`, `burn_depth_classifier`, `tbsa_estimator`, `blister_detector`, `necrosis_detector` | `burn_depth_tbsa_references` |

## Runtime Rule

1. Run the condition classifier first.
2. Look up the top condition in `condition_policy_registry.json`.
3. Run only required tools with `implementation_status = "implemented"`.
4. Record missing required or optional tools in the agent trace, but do not invent substitutes.
5. Restrict clinical retrieval to the policy `clinical_doc_ids`.
6. If no condition-specific documents are indexed, return no cross-condition staging evidence and flag the limitation.

For this prototype, DFU and pressure injury are the only conditions with packaged clinical references and severity model routing. Other conditions can still be classified, but their condition-specific measurement tools and clinical references remain placeholders until added.
