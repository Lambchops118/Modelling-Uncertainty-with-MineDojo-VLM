# Embodied AI Uncertainty Experiment

## Overview

This project investigates how different representations of uncertainty affect the performance of vision-language models (VLMs) acting as embodied AI agents.

The agent will operate inside a controlled Minecraft environment using **MineDojo**. Minecraft serves only as the simulated environment; the research focus is on how uncertainty affects an agent's perception, planning, information-seeking behavior, and task performance.

## Research Question

**How does the representation of uncertainty affect the performance and efficiency of VLM-based embodied agents?**

The same underlying task and environment will be presented using different descriptions of the agent's knowledge.

Example task:

> Retrieve the key.

Experimental conditions may include:

| Condition | Example |
|---|---|
| Certain | "Get the key. The key is in Room A." |
| Hedged / Semi-Certain | "Get the key. The key is probably in Room A." |
| Unstated | "Get the key." |
| Explicit Unknown | "Get the key. You do not know where the key is." |
| Quantified | "Get the key. There is a 70% chance the key is in Room A." |

The **Unstated vs. Explicit Unknown** comparison is particularly important because both provide the agent with the same location information (none), while differing in whether the agent is explicitly informed of its uncertainty.

## Experimental Environment

The initial implementation will use **MineDojo** as the interface to Minecraft.

MineDojo provides:

- Minecraft simulation
- First-person RGB observations
- Programmatic environment control
- Agent action interfaces
- Inventory and environment state
- Task/reward infrastructure
- Reproducible world configurations

The VLM will primarily receive **visual observations rather than privileged Minecraft world-state information**.

Hidden simulator state may be used by the experimental framework for environment setup, success evaluation, and logging, but will not be exposed to the agent unless explicitly required by an experimental condition.

## Agent Architecture

The planned interaction loop is:

```text
Task + Uncertainty Condition
            |
            v
      RGB Observation
            |
            v
           VLM
            |
            v
      Action Selection
            |
            v
        MineDojo
            |
            v
       Minecraft
            |
            v
     New Observation
            |
            +------> VLM
```

The VLM will maintain an action/observation history so that it can reason about previously explored locations and previous attempts.

Low-level motor control may eventually be abstracted through reusable action primitives so that experiments primarily measure planning and decision making rather than precise Minecraft controls.

## Initial Experiment

The first environment will intentionally be simple and highly controlled.

Example:

```text
             Start
               |
        +------+------+
        |      |      |
      Room A Room B Room C
        |      |      |
      Chest  Chest  Chest
```

One room contains a target object such as a key.

For each environment configuration, the same VLM will attempt the same task under each uncertainty condition.

Environment state, starting position, visual settings, model parameters, available actions, and agent instructions will remain constant except for the experimental variable being tested.

## Evaluation Metrics

Primary metrics:

- **Task success rate**
- **Number of actions required for successful completion**
- **Normalized action efficiency**

Additional behavioral metrics may include:

- Number of locations searched
- Information-seeking or verification actions
- Incorrect/redundant actions
- Replanning behavior
- Recovery from incorrect prior information

Complete action trajectories and observations will be logged for later analysis.

## Quantified Uncertainty

Quantified uncertainty will be tested using explicit probabilities, for example:

> "There is a 75% chance the key is in Room A."

The actual environment distribution will be calibrated to the stated probabilities across repeated trials.

This allows the experiment to investigate whether agent behavior changes appropriately as stated confidence changes.

## Experimental Controls

To isolate uncertainty representation:

- Identical environments will be reused across conditions.
- Each trial will begin with a fresh model context.
- Model parameters will remain fixed.
- Minecraft visual settings will remain fixed.
- Starting positions and orientations will be controlled.
- Hidden world state will not be exposed to the VLM.
- Trial ordering should be randomized.
- Environment seeds/configurations will be recorded for reproducibility.

A preliminary visual-perception evaluation may be conducted to verify that participating VLMs can reliably recognize important objects, rooms, and environmental features.

## Future Expansion

The initial key-search experiment is intended as a controlled proof of concept rather than the final scope of the research.

The same framework should support increasingly open-ended tasks such as:

> "Collect birch wood."

These scenarios can manipulate environmental uncertainty independently from communicated uncertainty—for example, comparing environments where birch trees are immediately visible with environments where the agent must explore to determine whether birch trees are available.

Future tasks may involve:

- Exploration
- Resource collection
- Object search
- Tool use
- Crafting
- Multi-stage objectives
- Environmental constraints
- Incorrect prior information

The experiment framework should therefore maintain a separation between the VLM agent, uncertainty-generation system, evaluation framework, and MineDojo environment so that additional tasks, models, and potentially other simulators can be added later.

## Intended Contribution

The goal is not to develop a Minecraft-playing agent.

Minecraft and MineDojo provide a controllable embodied environment in which uncertainty can be systematically manipulated.

The intended contribution is an experimental framework and empirical analysis of **how explicit, implicit, linguistic, and quantitative representations of uncertainty influence the behavior and performance of embodied VLM agents**.