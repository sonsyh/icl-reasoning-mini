# Third-party sources and release scope

Original project materials are not offered under an open-source license; see [RIGHTS.md](RIGHTS.md). No third-party licenses are replaced by that statement.

## PrOntoQA

- Upstream: <https://github.com/asaparov/prontoqa>
- Fixed commit: `066f73fc70bfaa92ed2476fd3c41b4951ad6e63a` (source branch `v1`).
- Upstream source license: Apache-2.0; preserved in the private source checkout.
- The public package does not bundle the upstream checkout, its model_outputs.zip, or the generated question datasets.
- The project data pipeline and validator are original project implementation referring to the upstream format and source. Cite Saparov and He (ICLR 2023) as described in README.

## Sentence Transformers / all-MiniLM-L6-v2

- Upstream: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- Fixed commit: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Fixed model-card metadata declares `apache-2.0`; the queried file list did not contain a standalone license file.
- No model weights, tokenizer files or model card text are bundled in this package. A dependency/version reference is not redistribution of the underlying model.

## DeepSeek-V4-Pro tokenizer and encoding resources

- Upstream: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro>
- Fixed commit: `b5968e9190ef611bbf34a7229255be88a0e937c1`.
- Root LICENSE at the fixed commit is MIT, Copyright (c) 2023 DeepSeek.
- Queried LICENSE: 1,084 bytes; SHA-256 `f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040`.
- No upstream tokenizer/encoding files or model weights are bundled. This software license is not a statement about remote API service terms or permission to redistribute raw API logs, which are excluded.

## Other dependencies and publications

Python/Conda packages are referenced by version but not bundled; each retains its own license. Referenced papers are cited by links, not distributed as PDFs. This file records the current reduced-release scope, not a blanket permission to redistribute excluded resources.
