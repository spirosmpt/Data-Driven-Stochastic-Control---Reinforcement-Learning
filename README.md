# Data-Driven Conditional Expectations & Stochastic Control

**MATLAB** · University of Patras · Machine Learning coursework

Numerical and neural-network approximations of conditional expectations, together with value-function estimation and control experiments for a stochastic dynamical system.

## What this project covers

- conditional expectation
- numerical integration
- data-driven neural estimation
- custom loss functions
- Adam optimization
- value functions
- dynamic programming / stochastic control

## Repository layout

- `report/course-report-redacted.pdf` — student-ID-redacted copy of the submitted report with methodology, findings, plots and code appendix
- Source files:
  - `code/p1_data_driven_clipped_a1.m`
  - `code/p1_data_driven_clipped_c1.m`
  - `code/p1_data_driven_mean_a1.m`
  - `code/p1_data_driven_mean_a2.m`
  - `code/p1_numerical_clipped_value.m`
  - `code/p1_numerical_expected_value.m`
  - `code/p2_action1_data_driven_a1.m`
  - `code/p2_action1_data_driven_c1.m`
  - `code/p2_action2_data_driven_a1.m`
  - `code/p2_action2_data_driven_c1.m`
  - `code/p2_numerical_value_functions.m`
  - `code/p2_p3_generate_data.m`
  - `code/p3_data_driven_a1.m`
  - `code/p3_data_driven_c1.m`
  - `code/p3_numerical_control.m`

## Getting started

Start with the numerical / data-generation scripts and proceed to the relevant per-method experiments. Scripts generating `.mat` outputs must be run before those loading them. Check the appendix for the original run order.

### Data / software dependencies

- MATLAB functions including `normcdf` (Statistics and Machine Learning Toolbox)
- Generated or referenced `Set1_Pairs.mat`, `Set2_Pairs.mat` and intermediate result `.mat` files

The datasets and any large/generated artifacts have deliberately not been added to version control.

## Technical notes

The individual script names reflect the report headings for Problems 1–3 and hypotheses A1, A2, C1.

**Source provenance and execution status:** The author confirmed the appendix code was tested. Source was transcribed to separate `.m` files from the PDF; some long lines wrap in the PDF. Preserve the redacted copy of the report as the authoritative reference.

## Credits

Coursework authored by Spiros Bantis, University of Patras. The report remains the reference for experimental setup and observed results. Refer to the assignment for technical details.
