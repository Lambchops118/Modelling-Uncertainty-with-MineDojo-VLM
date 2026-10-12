# Project Gathering Plan

This plan turns the experiment described in `Embodied AI Uncertainty Experiment.md` into a build-ready project. Its purpose is to collect and validate the decisions, software, assets, credentials, and experimental specifications needed before full implementation.

## 1. Freeze the minimum viable experiment

Write a one-page preregistration-style specification for the first experiment before building the agent.

Gather and decide:

- One target task: search for and retrieve a single target item from one of three visually distinct rooms.
- A precise definition of “retrieve” (for example, target item enters inventory).
- The initial uncertainty conditions. Recommended MVP set:
  - certain/correct location;
  - hedged/correct location;
  - unstated location;
  - explicit unknown location;
  - quantified location probability.
- Whether incorrect priors are excluded from the MVP and reserved for a follow-up experiment.
- The probability levels used in quantified conditions and the number of world instances required to realize each probability exactly.
- Maximum actions, maximum wall-clock time, retry policy, and terminal failure conditions.
- Primary hypotheses and planned statistical comparisons, especially unstated versus explicit unknown.

Deliverable: `docs/experiment_spec.md`, including hypotheses, independent and dependent variables, exclusions, and success criteria.

## 2. Validate the simulator before committing to it

MineDojo is an older stack with a Java/Minecraft backend, so environment validation is the first technical gate. The official material currently documents Python 3.9 or newer, JDK 8, `minedojo.make()`, RGB observations, action masks, custom starting conditions, world seeds, and drawing-based simulation customization.

Gather:

- Target execution platform: local macOS, Linux workstation, or Linux container/VM.
- Exact Python, JDK, MineDojo, operating-system, and graphics/headless-display versions.
- MineDojo source or package revision, plus licenses for MineDojo and Minecraft-related components.
- A reproducible environment definition (lockfile and preferably a container recipe).
- Evidence that a minimal environment can reset, return an RGB frame, accept actions, expose inventory for evaluation, and reproduce a seed.
- A list of any patches required to build or run MineDojo.

Gate: do not build the study on top of MineDojo until a smoke test passes repeatedly from a clean environment. If it cannot be made reproducible in a short technical spike, evaluate a maintained Minecraft interface while preserving the experiment-facing adapter API.

Deliverables: `environment/`, `scripts/smoke_test.py`, and `docs/environment_validation.md`.

## 3. Specify and acquire the VLM interface

Gather for each candidate model:

- Model/API name and pinned version or dated endpoint.
- Image input format, supported resolution, context limits, latency, rate limits, and cost.
- Sampling controls (temperature, seed support, maximum output tokens).
- Terms governing storage of screenshots and model outputs.
- Credentials supplied only through environment variables or a secret manager; never commit keys.
- A structured action-output format the model can reliably produce.

Choose one model for the MVP. Keep a provider-neutral adapter so later models can be compared without changing environment or evaluation code.

Deliverables: `docs/model_card.md`, `.env.example`, and an agent adapter contract.

## 4. Build the controlled world asset pack

Gather or create:

- A three-room layout with controlled dimensions, lighting, textures, doors, and chest placement.
- A target item available in the selected Minecraft version. If “key” is not a native item, define an unambiguous substitute or custom representation.
- At least three room variants that are visually distinguishable without leaking the target location.
- Fixed spawn position, camera orientation, field of view, weather, time, difficulty, and mob settings.
- A balanced table mapping world seeds/configuration IDs to target locations.
- A reset procedure that proves no state carries over between trials.
- Screenshots from the agent’s view for every room and object state.

Keep hidden state in the evaluator only. The agent-facing observation must contain RGB plus explicitly approved history; inventory/world state may be used for scoring but not placed in the prompt.

Deliverables: `assets/worlds/`, `configs/worlds/`, an asset manifest with hashes, and a world-validation script.

## 5. Define the action vocabulary and controller

Start with high-level, deterministic primitives so the experiment measures planning and information seeking rather than fine motor control.

Candidate primitives:

- turn to or navigate to Room A/B/C;
- inspect room;
- open chest;
- take target;
- return/stop;

For every primitive, gather or define preconditions, simulator action sequence, timeout, failure signal, retry behavior, and action cost. Decide whether one primitive or each low-level simulator step counts toward the primary action metric; log both.

Deliverables: `docs/action_spec.md`, controller implementation, and unit/integration tests for every primitive.

## 6. Create the prompt and uncertainty-condition registry

Gather exact, immutable prompt templates. Separate:

- constant system/agent instructions;
- task goal;
- uncertainty statement;
- action schema;
- observation/history serialization.

Each condition should alter only the uncertainty statement. Add automated checks that templates have the same task goal and differ only in approved fields. For quantified uncertainty, generate trial allocations whose empirical target frequencies exactly match the stated probabilities within each completed block.

Deliverables: `configs/conditions.yaml`, prompt snapshots, and prompt-difference tests.

## 7. Define logging, provenance, and evaluation

Gather a trial schema containing:

- run, block, trial, condition, and configuration IDs;
- environment and model versions;
- seeds and randomized trial order;
- prompts, RGB observation references, raw model responses, parsed actions, simulator actions, rewards, and hidden evaluator state;
- token usage, latency, errors, retries, termination reason, and final inventory;
- timestamps and code commit hash.

Predefine calculations for success rate, low-level action count, primitive count, normalized action efficiency, rooms searched, revisits/redundant actions, verification behavior, replanning, and invalid-action rate. Define the optimal-path denominator for every world configuration rather than deriving it after seeing results.

Deliverables: a versioned JSONL/Parquet schema, validator, metric definitions, and a small analysis notebook or script.

## 8. Run perception and control qualification tests

Before testing uncertainty, establish that failures are not dominated by vision or control.

Gather pilot data for:

- room-label recognition from controlled viewpoints;
- chest and target recognition;
- successful navigation/open/take execution;
- history use after visiting multiple rooms;
- robustness across the exact visual variants planned for the study.

Set pass thresholds in advance. Failed assets or primitives should be revised before experimental trials and then frozen.

Deliverable: `reports/qualification_report.md` with confusion counts and controller success rates.

## 9. Conduct a balanced pilot

Use a small, balanced set of configurations and all MVP uncertainty conditions.

Check:

- deterministic reset and absence of information leakage;
- parser and controller failure rates;
- ceiling/floor effects;
- whether the model distinguishes unstated from explicit unknown;
- whether quantified prompts cause sensible search ordering;
- cost and runtime per trial;
- completeness and replayability of logs.

Use the pilot only to repair protocol defects and estimate variance. Freeze the final protocol, analysis code, exclusions, sample size, and randomization before the main run.

Deliverables: pilot dataset, pilot report, frozen configuration bundle, and a main-run checklist.

## 10. Required project structure

The gathered pieces should eventually fit this separation:

```text
assets/                 # immutable world and visual assets
configs/                # worlds, conditions, models, trial manifests
environment/            # MineDojo adapter and reset/setup logic
agent/                  # VLM adapter, history, prompt assembly, parsing
controller/             # action primitives and execution results
evaluation/             # hidden-state scoring and metric calculations
runner/                 # randomized, resumable experiment orchestration
analysis/               # preregistered summaries and statistical tests
tests/                  # unit, integration, determinism, leakage tests
docs/                   # experiment, environment, action, and model specs
data/                   # ignored raw runs plus checked-in schemas/examples
```

## Immediate collection checklist

The next work session should collect these items in order:

1. Hardware/OS target and whether a Linux container or VM is available.
2. A pinned, smoke-tested MineDojo environment and minimal RGB/action recording.
3. The exact MVP task, success rule, action budget, and five prompt texts.
4. The selected VLM endpoint, access credentials, budget, and sampling settings.
5. One reproducible three-room world plus target-location manifest.
6. A tested high-level action vocabulary.
7. Trial and log schemas with seed/provenance fields.
8. Qualification thresholds and a balanced pilot manifest.

## Definition of “ready to implement the full experiment”

The project is ready when a clean machine can recreate the environment; one command can run a seeded trial; the VLM receives only permitted observations; the evaluator can determine success independently; all prompt differences are auditable; every action and observation is logged; and a trial can be replayed closely enough to diagnose any failure.

## Key risks to resolve early

- **MineDojo compatibility:** Java, graphics, Minecraft, and Python dependencies may be fragile.
- **Construct validity:** free-form low-level control can measure motor reliability more than uncertainty handling.
- **Prompt confounds:** wording length, tone, or implied advice can vary alongside uncertainty.
- **Probability calibration:** too few trials make stated probabilities impossible to realize faithfully.
- **Visual leakage:** room design or target placement may unintentionally reveal location.
- **Non-deterministic model behavior:** fixed parameters do not guarantee repeatability; repetitions and complete logs are required.
- **Metric ambiguity:** “number of actions” and “optimal path” must be defined before data collection.
- **Cost and rate limits:** image-heavy, multi-step episodes can be substantially more expensive and slower than text-only calls.

## Reference material consulted

- Project concept: `Embodied AI Uncertainty Experiment.md`
- MineDojo documentation: <https://docs.minedojo.org/>
- MineDojo first-agent/API guide: <https://docs.minedojo.org/sections/getting_started/sim.html>
- MineDojo task customization: <https://docs.minedojo.org/sections/customization/task.html>
- MineDojo repository: <https://github.com/MineDojo/MineDojo>
