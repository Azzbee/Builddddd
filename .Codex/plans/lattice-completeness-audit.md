# Lattice completeness audit

## Goal

Find and permanently fix the remaining production defects after the first full hardening pass.

## Work

1. Audit security boundaries, external downloads, API contracts, UI failure paths, persistence, deployment, dependency health, and documented claims.
2. Fix every confirmed issue in a focused change with regression tests.
3. Update existing documentation when behavior or operating procedures change.
4. Run the slop scan and an adversarial review of the resulting diff.
5. Run type checks, lint, formatting checks, all tests, coverage, builds, evaluation, dependency audits, and deployment configuration checks.
6. Commit each logical change, push the branch, and confirm remote CI.

## Constraints

- Preserve the user's existing unstaged `web/tsconfig.json` change.
- Do not weaken existing security, coverage, or evaluation gates.
- Do not claim completion unless every available verification gate passes.
