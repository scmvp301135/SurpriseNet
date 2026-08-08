# Spec: Research code support boundary

## Objective

Resolve the repository's long-running setup questions without presenting an incomplete 2021 research snapshot as a supported training or inference release. A reader should be able to tell what is available, what was renamed, what is missing, and which commands are verified today.

## Tech stack

- Markdown for the public support contract.
- Node.js standard-library tests for documentation integrity.
- Git history as provenance for renamed legacy files; no legacy source is duplicated into the current tree.

## Commands

```bash
node --test tests/repository-docs.test.mjs tests/web-demo.test.mjs
python3 -m http.server 8000 --directory docs
```

There is intentionally no supported research training or arbitrary-melody inference command.

## Project structure

```text
README.md                         Project overview and availability matrix
RESEARCH_CODE.md                  Detailed archival status and historical pointers
tests/repository-docs.test.mjs    Documentation contract tests
docs/                             Static, precomputed listening demo
specs/research-code-support.md    Durable support-boundary specification
```

## Code style

Availability claims use explicit states rather than implied support:

```markdown
| Capability | Status |
| --- | --- |
| Pretrained checkpoint | Not included |
```

## Testing strategy

- Verify every local Markdown link in the maintained documentation resolves to a checked-in path.
- Verify documented `python … .py` commands reference checked-in files.
- Verify the availability contract explicitly covers preprocessing, training, checkpoints, and arbitrary-melody inference.
- Run the documentation checks together with the existing web-demo tests in CI.

## Boundaries

- Always: distinguish the precomputed web demo from model inference; link historical claims to Git history; describe unsupported paths explicitly.
- Ask first: restore or redesign preprocessing, training, checkpoint distribution, or user-melody inference.
- Never: add a fake compatibility wrapper, publish an unverified requirements lock, claim the research pipeline runs on modern Python/Mac, or close an unavailable feature as completed.

## Success criteria

- The old `surprisenet_train.py` and `surprisenet_inference.py` names are explained with their rename commit and legacy snapshots.
- Readers are told that no verified preprocessing recipe, pretrained checkpoint, supported training CLI, or arbitrary-melody inference adapter is included.
- The environment choice is unambiguous: `environment.yml` is archival, not a modern reproducibility guarantee.
- Documentation tests fail if local links, executable file references, or the availability contract regress.
- Issues #1, #4, and #6 receive a precise not-planned resolution; issue #3 receives a completed documentation resolution.

## Open questions

- A future reproducibility project would need a redistributable fixture, preprocessing schema, tested environment lock, published checkpoint, and end-to-end CPU test. That work is outside this documentation repair.
- Before deleting the legacy `website` branch, a repository administrator must switch Pages to GitHub Actions and verify a successful deployment.
