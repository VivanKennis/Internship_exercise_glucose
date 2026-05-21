# Systems Biology Project Template

This repository provides three implementation levels for mathematical modelling workflows:

- Basic: minimal starting scaffold
- Intermediate: standard end-to-end workflow
- Advanced: full workflow including uncertainty and identifiability analyses

Use this README to choose the right level. Then follow the README inside the chosen implementation folder for detailed usage.

## Which version should I choose?

| If you need... | Choose |
| -------------- | ------ |
| You are new to Python or modelling code and want the smallest possible starting point | `implementation_basic/` |
| You know basic Python (or can follow simple scripts) and want a ready-to-use standard workflow | `implementation_intermediate/` |
| You are comfortable with Python and optimization workflows, and need PI/PPL and uncertainty analysis | `implementation_advanced/` |

After choosing, the next step is to download/copy the respective sub-folder and start your project from the given base.

## Level overview

### Basic

Path: `implementation_basic/`

Best for:

- Python beginners who want to focus on model/data structure before full scripting
- New team members onboarding to the project structure
- Teaching or lightweight prototypes


### Intermediate

Path: `implementation_intermediate/`

Best for:

- Users with beginner-to-intermediate Python experience
- Standard modelling tasks used in most projects
- Parameter estimation workflows with reusable common utilities

Includes:

- `main.py` orchestration script
- `common/` reusable helper functions
- `methods/` modular task scripts (setup, cost, plotting, estimation)


### Advanced

Path: `implementation_advanced/`

Best for:

- Users comfortable reading and modifying modular Python code
- Projects requiring deeper analysis and uncertainty characterization
- Teams that need PI/PPL methods in addition to standard estimation

Includes:

- Everything in intermediate, plus:
- Parameter Identifiability (PI)
- Prediction Profile Likelihood (PPL)
- Model uncertainty utilities


## Recommendation by user profile

- If you are a Python novice: start with `implementation_basic/`, then move to `implementation_intermediate/`.
- If you are a modeller with basic Python skills: start with `implementation_intermediate/`.
- If you are an experienced modeller/developer: start with `implementation_advanced/` when PI/PPL or uncertainty analysis is required.
