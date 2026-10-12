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
movement uses 5-degree increments. `BREAK_BLOCK` is a separate primitive that
holds attack continuously for up to 40 controller steps; `ATTACK` is one
strike. The Minecraft client will still pause while the remote model deliberates
because this initial architecture is synchronous. Change visible action pacing
with `--action-step-delay`, for example `--action-step-delay 0.1` for slower
demonstration playback.

## Current scope

This is an integration demonstration, not yet an experimental protocol. It uses
MineDojo's existing wool-harvest environment, the moving `gpt-6.1-sol` model ID,
and a general navigation prompt. Before collecting research data, add a fixed
world, frozen prompt registry, trial manifest, exact model snapshot if one is
made available, retry/error policy, and preregistered success metrics.
