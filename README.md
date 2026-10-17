# ICL Reasoning Mini

**Does semantic demonstration retrieval improve few-shot logical reasoning?**

A small, controlled study on a PrOntoQA-derived binary logic task, comparing zero-shot, random answer-only examples, random chain-of-thought examples, and semantic chain-of-thought examples.

**Public research showcase — not an open-source release.** No open-source license is granted for the original project code. Third-party materials retain their own terms. See [rights statement](RIGHTS.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Main result

| Condition | Demonstrations and output | Correct / 60 | Accuracy |
|---|---|---:|---:|
| C0 | Zero-shot; final answer only | 48 | 80.00% |
| C1 | Random 2-shot; answer-only examples/output | 53 | 88.33% |
| C2 | Same random examples/order; reference derivations and requested output derivation | 60 | 100.00% |
| C3 | Semantic 2-shot; same derivation format as C2 | 60 | 100.00% |

The primary comparison is **C3 versus C2**: zero observed accuracy gain, with both reaching the ceiling. This does not establish statistical equivalence or prove retrieval universally ineffective. The project proposes no new retrieval algorithm and is not a full reproduction of the referenced papers.

![Four-condition accuracy](figures/four_condition_accuracy.png)

## Fixed design

- 600 generated candidates; 120 demo, 20 dev, 60 test, 400 unused. Splits are separated by question ID and logical world.
- Formal test: balanced True/False labels, 20 questions at each depth 2/3/4; four conditions give 240 responses. Development results are not pooled into this score.
- False means the explicit negation is derivable, not merely that the query is unproven.
- Every few-shot prompt uses one True-label and one False-label demo, True first. C1/C2 share random demonstrations and order, seed `20261009`.
- Semantic retrieval: fixed MiniLM-L6-v2, normalized 384-dimensional vectors, label-balanced cosine top1 selection, excluding the target ID/world.
- Target answers, depths and proofs are not used in target encoding, selection or generation. Selected demo labels and reference derivations have their specified roles.
- Answering API alias: `deepseek-v4-pro`; thinking disabled, temperature 0, max output tokens 1024. The alias does not independently identify immutable serving weights. No training or fine-tuning.
- Formal saved responses all ended normally on their first attempt; incorrect content was not retried.

## What is published

```text
scripts/          Experiment implementation and selected core tests
configs/          Inference and retrieval configurations
results/          Aggregate experimental statistics
figures/          Two result figures in PNG and SVG
environments/     Candidate environment configuration
tools/            Validation tools for this public package
docs/             Supporting documentation and release records
```

This repository provides selected code, configurations, aggregate results and figures for research demonstration. It does not include all datasets, original model responses and intermediate artifacts required to reproduce the experiments. **The published materials alone are therefore insufficient for complete end-to-end reproduction.**

## Safe checks for this public package

Use Python 3.12 on a Unix-like system. The selected core tests need only the standard library; this has been verified on macOS ARM64, not all platforms.

```bash
python -B tools/run_public_checks.py
```

This checks the public file manifest and aggregate tables, then runs an explicitly selected set of synthetic tests for parsing, client guards, logs, retry/resume, budget simulation, simulated execution and logical validation. Outbound socket connections are blocked. Tests use temporary files and fake responses; no real API or model is invoked.

The count for this subset is reported in [release validation](docs/RELEASE_VALIDATION.json). It must not be conflated with the 318 historical private-workspace tests or the 14 later portability tests.

The [environment candidate](environments/environment.analysis.osx-arm64.candidate.yml) captures the broader analysis setup. A fresh environment on the same macOS ARM64 host passed installation/runtime checks and the original 318 tests in an isolated full workspace copy. This does **not** make the reduced package an end-to-end or cross-machine reproduction, nor is the YAML an artifact-hash lock.

## Interpretation and limitations

- C1 fixed 8 C0 errors but introduced 3; the net improvement was 5 questions. C2 fixed all 7 C1 errors.
- C2 versus C1 changes both demonstration reference chains and the request to produce a derivation; it is not a single-factor causal estimate of CoT.
- C3 used 44,767 input tokens versus C2's 43,084 (+3.91%); token counts are not a bill or an end-to-end efficiency measurement.
- One answering model, synthetic task subset, fixed 2-shot, one seed and one generation run; no confidence intervals or significance tests are reported.
- Data review used one human blind answer plus eleven assistant-assisted nonblind checks, not twelve independent human blind reviews.
- Later case analysis was posthoc, nonblind and purposively selected. Twenty inspected displayed derivations do not validate all chains or establish the model's internal reasoning mechanism.
- No extra test tuning or best-of-rerun selection was performed after observing formal results.

![C2 and C3 by reasoning depth](figures/c2_c3_accuracy_by_depth.png)

## References

1. Long, Q., Wu, Y., Wang, W., & Pan, S. J. (2024). *Does In-Context Learning Really Learn? Rethinking How Large Language Models Respond and Solve Tasks via In-Context Learning*. COLM. <https://arxiv.org/abs/2404.07546>
2. Saparov, A., & He, H. (2023). *Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought*. ICLR. <https://arxiv.org/abs/2210.01240>


