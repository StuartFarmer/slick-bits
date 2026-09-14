# VLM black-box optimization

`VLMBlackBox(task, provider, evaluate, constraints="", required_tokens=()).run(initial, rounds=10, pool_size=3, candidates=1)` implements the good/bad/better search from [Language Models as Black-Box Optimizers for Vision-Language Models](https://arxiv.org/abs/2309.05950).

Official source inspected: [classification loop](https://github.com/shihongl1998/LLM-as-a-blackbox-optimizer/blob/main/auto_prompt_image_classification.py) and [prompt pool](https://github.com/shihongl1998/LLM-as-a-blackbox-optimizer/blob/main/prompt_pool.py). Every round presents the best and worst templates without their numeric scores, generates replacements, and adds unseen candidates to the archive. Fitness is finite and maximized; ties retain insertion order. The pools can overlap when the archive is small, as upstream. Duplicate text is scored once.

The caller's async `evaluate(text)` owns the frozen model and optimization split. Task descriptions replace dataset names; `constraints` can retain the source's 15-word target and `required_tokens` checks placeholders. JSON replaces line-prefixed parsing. Classification, image generation and inversion use the same text/fitness boundary; this port does not bundle CLIP, diffusion pipelines or source benchmark reporting. Provider/evaluator errors propagate without upstream's 15 transport retries.

Configure `slick.prompts.TEMPLATE_ROOT = Path(vlm_blackbox.__file__).parent / "prompts"` once before calls. Returns best text, score, archive and distinct evaluation count. Check: `rtk proxy optimizer/.venv/bin/python -B -m unittest tests.test_vlm_blackbox`.
