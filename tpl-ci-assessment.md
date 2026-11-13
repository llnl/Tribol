# TPL CI Branch

## Must Address Before Merging

### 1. TPL runs can use stale results

`FULL_BUILD_ROOT` is constant per job, and `--short-path` removes the timestamp:

- `.gitlab-ci.yml:13`
- `.gitlab-ci.yml:55`

More seriously, newly generated host configs are copied only when the destination does not already exist:

- `scripts/llnl/common_build_functions.py:486`

A nightly run can build new TPLs and then test with the previous run's host config. The workflow should:

- Use a pipeline- or job-specific install prefix.
- Retain a separate shared mirror for downloaded source archives.
- Always overwrite the copied host config.
- Serialize overlapping jobs if any install state remains shared.

### 2. The documented devtools workflow is broken

`scripts/spack/devtools.json:2` requests `tribol_devtools`, while the new package is named `triboldevtools`.
Additionally, `scripts/spack/devtools.json:9` references the deleted `scripts/spack/packages` directory.

The README also lists removed or outdated paths:

- `README.md:25` lists the deleted BlueOS configuration.
- `README.md:48` points to the old Tribol package location.

The devtools configuration and documentation should be updated as part of the package-repository migration.

### 3. Regression coverage was removed rather than repaired

`mortar_lm_patch_test` still builds, but it is no longer registered with CTest:

- `src/examples/CMakeLists.txt:40`

The commit describes this as a temporary disable and references
[issue #281](https://github.com/llnl/Tribol/issues/281). The test should be repaired before merge, or the loss of
coverage should receive explicit project approval and be documented as a known limitation.

### 4. Spack variant relationships are inconsistent

CUDA, HIP, and OpenMP compilation require RAJA, but the package permits combinations such as `+cuda~raja` and
`+openmp~raja`. Dependencies can then be pulled in implicitly despite the disabled Tribol variant:

- `scripts/spack/spack_repo/tribol/packages/tribol/package.py:109`
- `scripts/spack/spack_repo/tribol/packages/tribol/package.py:126`

The package should use Spack `requires()` or `conflicts()` directives to encode valid combinations explicitly.

ROCm also forces `mfem+raja+umpire` regardless of Tribol's RAJA and Umpire variants:

- `scripts/spack/spack_repo/tribol/packages/tribol/package.py:147`

This prevents lean MFEM ROCm configurations and makes Tribol's variants misleading. The unresolved review questions
about MFEM Enzyme, OpenMP, RAJA, and Umpire coupling should be settled before merging the package migration.

### 5. Spack configuration contains malformed keys

`build_stage::` becomes a literal YAML key named `build_stage:` and is ignored by Spack:

- `scripts/spack/configs/defaults.yaml:9`

`compilers::` has the same issue:

- `scripts/spack/configs/linux_ubuntu_24/spack.yaml:2`

The `compilers::` typo predates this branch, but this configuration is being substantially regenerated here and should
be corrected as part of the migration.

### 6. The new TPL jobs need fresh-prefix validation

Current PR checks run the `*-src` jobs, not the new `*-tpl` jobs. A fresh-prefix Dane, Matrix, and Tuolumne TPL
pipeline should pass before merge. Reusing the current fixed prefixes would not provide sufficient evidence because of
the stale-host-config issue.

## What the Branch Does Well

- Moves Tribol's packages into a proper in-repository Spack v2 repository.
- Removes the RADIUSS configuration submodule dependency.
- Pins both Spack and `spack-packages` commits in `.uberenv_config.json`, improving reproducibility.
- Separates source CI from scheduled TPL CI.
- Adds real Matrix CUDA and Tuolumne HIP source coverage.
- Updates dependency versions, BLT, Docker images, host configs, and static-library requirements consistently.
- Pins GitHub Actions by immutable commit SHA.
- Adds a regression test ensuring `TRIBOL_VERSION_FULL` is nonempty.
- Passes all currently enabled GitHub and GitLab source checks, including Matrix CUDA and Tuolumne HIP.

## Recommended Improvements

### Publish TPL build results

The TPL build output is outside `CI_PROJECT_DIR`, and `.tpl_build_script` has no artifact or JUnit handling:

- `.gitlab-ci.yml:50`

Copy the relevant logs and JUnit output into the project directory and publish them with `when: always` so failures can
be diagnosed after runner cleanup.

### Reuse downloads, not installations

Use the shared mirror helpers rather than creating a separate mirror under every job's build root. A shared source
mirror reduces repeated downloads without introducing the correctness risks of shared Spack installations and host
configs.

### Correct the Tuolumne setup

The TPL-specific `before_script` replaces the inherited CMake module load, and its pip version comparison is
lexicographic:

- `.gitlab/build_tuolumne.yml:28`

Preserve the parent setup and compare pip versions with a version-aware mechanism.

### Remove dead cross-repository code

`config-build.py` references a helper that does not exist in this repository:

- `config-build.py:39`

The fallback hides the problem, so the branch should either add the intended helper or remove this lookup.

### Clean the diff

`git diff --check origin/develop...HEAD` reports:

- Trailing whitespace at `scripts/llnl/common_build_functions.py:620`.
- Extra blank lines at the end of eight generated host configs.

### Clean up the commit history

The branch contains temporary, revert, empty, and unrelated example/logger commits. Squashing or reorganizing the
history would make the package migration and CI design easier to review and maintain.

## Validation Performed

- Built the CPU configuration locally using 14 parallel jobs.
- Passed all 65 local CPU tests using 14 parallel jobs.
- Compiled the Python files with `python3 -m compileall`.
- Confirmed that all currently enabled GitHub and GitLab source checks pass.
- Used the exact pinned Spack version and `spack-packages` commit.
- Successfully concretized the ROCm configuration and confirmed the forced MFEM RAJA/Umpire variants.
- Verified that the local branch and GitHub PR head both point to `6af198e2`.
- Verified that local `origin/develop` matches the current GitHub `develop` branch.

The full scheduled TPL builds were not run from fresh prefixes during this assessment. Those runs remain necessary after
the install-prefix and host-config handling are corrected.
