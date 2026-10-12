# GPT-6.1 Sol MineDojo Agent

The initial agent sends the current MineDojo RGB observation and a short action
history to GPT-6.1 Sol through the OpenAI Responses API. A strict Pydantic schema
limits the model to one bounded high-level action. Local controller code then
translates that action into MineDojo's eight-value action vector.

## Installation

Activate the working MineDojo Python 3.9 environment, then install the pinned
agent dependency:

```bash
conda activate minedojo
python -m pip install -r requirements-agent.txt
```

Set the API key in the current shell. Do not put the key in source control:

```bash
export OPENAI_API_KEY="your-key-here"
```

The account must have API billing and access to `gpt-6.1-sol`. A ChatGPT
subscription is separate from API billing.

## Run

From the repository root:

```bash
python scripts/run_vlm_agent.py
```

The current default loads the local Minecraft 1.11.2 world `Enrichment Center 1`,
spawns at `(365, 57, -477)` facing east, and asks the agent to retrieve a
diamond from one of three chests. MineDojo's action wrapper cannot click chest
GUI slots, so the agent is instructed to break a chest and walk over its dropped
contents. Success is verified from the MineDojo inventory observation rather
than from the model's visual judgment.

The runner maintains persistent structured memory for the red, blue, and yellow
rooms. A room remains visited and its chest remains checked after the relevant
actions leave the rolling eight-action history. Each model response classifies
the current area and states whether the crosshair is visibly on a chest. After
a corresponding `BREAK_BLOCK`, the controller marks that colored room's chest
as checked and directs the model toward unchecked rooms. Each run also records
an explicit termination reason: `success`, `model_stop`, `environment_done`, or
`decision_budget_exhausted`.

Chest ownership is tracked separately from the agent's current area. This lets
the controller mark a blue-room chest as checked even when the agent breaks it
while standing in the corridor. The objective status also identifies the item
in hotbar slot zero, which appears in the lower-right hand view, so a collected
chest item is not mistaken for a chest block in the world.

A break attempt no longer marks a room checked immediately. The following
visual observation must confirm that the chest block disappeared. Chest breaks
are forced to 100 controller ticks, and if the model requests a break while it
is still outside the target room, the controller approaches instead and logs
both the requested and executed actions.

On this Mac the runner automatically selects the installed Temurin Java 8 and
sets `CONDA_PREFIX` from the active Python environment before MineDojo launches.
Run the script with the `minedojo` environment activated; do not wrap it in
`conda run`, because MineDojo manages child processes and temporary directories.

To use another saved world or starting pose:

```bash
python scripts/run_vlm_agent.py \
  --world-path "/absolute/path/to/world-folder" \
  --spawn-x 365 --spawn-y 57 --spawn-z -477 \
  --spawn-yaw -90 --spawn-pitch 0
```

The default run stops after at most 60 model decisions. To change the task text:

```bash
python scripts/run_vlm_agent.py \
  --task "Turn until you see a sheep, approach it, then stop." \
  --max-decisions 12
```

Every request is stateless at the API level (`store=False`). The runner supplies
the last eight action summaries itself. Each run writes its configuration,
frames, decisions, response IDs, token usage, latency, reward, and termination
state under `runs/`. API keys are never written to the run log.

The controller paces action macros at 20 Hz by default and sends an explicit
neutral action after every macro to release stateful Minecraft keys. Camera
movement uses 5-degree increments. Movement macros can cross a clear room or
corridor in fewer decisions, and `SPRINT_FORWARD` is available for long,
unobstructed paths. `BREAK_BLOCK` holds attack continuously for up to 100
controller steps; `ATTACK` is one strike. The Minecraft client will still pause
while the remote model deliberates
because this initial architecture is synchronous. Change visible action pacing
with `--action-step-delay`, for example `--action-step-delay 0.1` for slower
demonstration playback.

## Current scope

This is an integration demonstration, not yet an experimental protocol. It uses
the local three-room chamber and the moving `gpt-6.1-sol` model ID. Before
collecting research data, add a frozen prompt registry, trial manifest, exact
model snapshot if one is made available, retry/error policy, coordinate-based
room boundaries, and preregistered success metrics.
