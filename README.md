# RiskReasoner

The RiskReasoner repo aims to build a LLM-based reasoning system enhanced with system 2 reasoning ability and more explanability for financial domain tasks, especially for Credit and Risk Assessment


## In an ideal world
There are quite some resources I browsered to formulate my current ideas though they are still unripe. But idealy, the RiskReaoner should act like open o1 and specialized for the finance domain to achieve two main goals:

1. **System 2 Reasoning**: The RiskReasoner should be able to reason about the underlying causes of a problem and not direct explanations to increase its reasoning ability / classification metrics like ks score in my case.

2. **Explainability**: The RiskReasoner should be able to provide explanations with its step-by-step thinking for its predictions. This is especially important for the finance domain since traditional ML methods fail to justify their predictions (instead they work more like a black box).

The resources I recommand to have a better understanding of the ways to achieve these goals:

 - [Introducing OpenAI o1](https://openai.com/o1/)
 - [Training Verifiers to Solve Math Word Problems](https://arxiv.org/pdf/2110.14168)
 - [Let’s Verify Step by Step](https://arxiv.org/pdf/2305.20050)
 - [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/pdf/2406.06592)
 - [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/pdf/2408.03314)


## Quick Tour into repo

1. ``preprocessor``: contains scripts for preprocessing data for the RiskReasoner

    - ``prior.py``: process data needed for establishing baselines (e.g., expert systems, LLMs of direct inference)
    - ``posterior.py``: add ML predictions to prompts for LLMs as reference (therefore "posterior" in the name)

2. ``inference``: contains scripts for inference with the RiskReasoner

    - ``base.py``: base class for all inference methods (generators, not expert systems / ML systems)

    - ``sys1``: contains reasoning methods that require no thinking
      - ``expert_systems.py``: baseline using traditional ML methods
      - ``generator.py``: LLM-based inference, classification results are generated directly
      - ``generator_cot.py``: LLM-based inference, classification results are generated after a CoT process. This is still considered system 1 reasoning since the responses are generated in a autoregressive manner without pausing.

    - ``sys2``: contains reasoning methods that require thinking (or pauses)
      - ``generator_cot_N.py``: LLM-based inference, the model generates N different responses for each question, and the responses are to be compared/verified by a verifier to find the best one. The generation processing itself is not strictly system 2 reasoning, but the compound system will be to some extent since the system appears to pause to think (e.g., during verification.)
      - ``verifier.py``: LLM-based verifier, for results generated from ``generator_cot_N.py``. The model required for this is trained as introduced by [this work](https://arxiv.org/pdf/2305.20050), data is synthesized as the [OmegaPRM paper](https://arxiv.org/pdf/2406.06592)

3. ``training``: contains training scripts for the verifier (a process-supervised reward model, or PRM), and the [ReFT method](https://arxiv.org/pdf/2401.08967) for finetuning the generator (not completely identical yet)

    - ``omegaPRM``: contains the scripts for training the verifier, the scripts are from [openr framework](https://github.com/openreasoner/openr/tree/main/data/omegaPRM_v2) with some modifications

    - ``sft``: contains the scripts for finetuning the generator using SFT, also acts as the warm-up stage in the ReFT method

    - ``rl``: right now, only contains the scripts for finetuning the generator using PPO, which is the seconda stage of the ReFT method.

      **Note** I have not finished the ReFT method as it should be yet, and the SFT and PPO finetunings are seperate for now.

## In the future

Some High Level TODOs:
 - [ ] Finish ReFT method and test its performance with the verifier.
 - [ ] Use the OmegaPRM to start PPO training again and evalaute the performance.
 - [ ] Explore how to widen the solution space of the generator using methods similar to [this one](https://arxiv.org/pdf/2309.17179)
 - [ ] Some other ideas waiting to be explored...