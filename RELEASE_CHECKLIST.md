# Release Checklist

The steps for releasing version `X.Y.Z`. Tags use the bare version (`1.1.3`, `2.0.0`), with no `v`
prefix, to match `tag_prefix = ''` in `setup.cfg`.

## 1. Version and release documents

- [ ] Set `version="X.Y.Z"` in `setup.py`. The version is hard-coded there, while versioneer
      supplies `implicit_solvent_ddm.__version__` from the git tag, so the two must agree.
- [ ] Check `python_requires` in `setup.py` against the Python version pinned in
      `devtools/conda-envs/test_env.yaml`.
- [ ] `CHANGELOG.md`: rename the pending section to `## [X.Y.Z] - YYYY-MM-DD` and sort its entries
      into Breaking, Added, Fixed, Removed and Experimental. List each change once, list under
      Fixed only fixes to behaviour that a previous release had, and cite only paths that are
      tracked in the repository.
- [ ] Write `RELEASE_NOTES_vX.Y.Z.md`: summary, highlights, breaking changes with migration notes
      (what an old config or analysis does now, and what to change), install, testing.
- [ ] Update `docs/` for new or changed config keys.
- [ ] Every example config loads with the release code as committed. Run the check from the root
      of a clean checkout of the release branch (for example
      `git worktree add ../isddm-check release/X.Y.Z`), not in a working tree with uncommitted
      edits, which could make it pass:

      ```bash
      python -c "import glob, yaml; from implicit_solvent_ddm.config import Config; [Config.from_config(yaml.safe_load(open(f))) for f in sorted(glob.glob('script_examples/config_files/*.yaml'))]"
      ```

      Update the configs and the `docs/examples/` walkthroughs when a key is renamed or removed.
- [ ] Every file the release documents cite, or whose change they describe, is committed with them:
      `git ls-files --error-unmatch <path>` succeeds for each, and `git status --short` shows no
      `??` or ` M` for any of them. If one is left out, drop the claims that depend on it. For
      2.0.0 these are `RELEASE_NOTES_v2.0.0.md`, the example configs
      `script_examples/config_files/in_solvent_paper.yaml`, `remd_config.yaml` and
      `user_defined_endstate.yaml`, and the walkthroughs `docs/examples/basic_md_example.rst`,
      `remd_example.rst` and `user_defined_example.rst`.
- [ ] Stage files by name, not with `git add -A`, so editor backups (`*.bak`) and run output stay
      out of the commit.

## 2. Tests

Run every tier locally, in the env CI uses. Build it from the lock file, which needs no solve:

```bash
micromamba create -n isddm_env -f devtools/conda-envs/test_env.conda-lock.yml
# or: conda-lock install -n isddm_env devtools/conda-envs/test_env.conda-lock.yml
micromamba activate isddm_env
pip install -e .

pytest -m "not workflow and not hpc"   # unit tier, the CI unit job's command (about 3 s)
pytest -m workflow                     # CB7 Toil + sander workflow tier (about 65 s)
```

Plain `pytest` deselects only `workflow` (`addopts` in `setup.cfg`), so it runs the unit tier plus
the two `hpc` tests, which skip where their files or binaries are missing.

- [ ] If `devtools/conda-envs/test_env.yaml` changed, regenerate the lock file and run both tiers
      in an env built from the new one:
      `conda-lock -f devtools/conda-envs/test_env.yaml -p linux-64 --lockfile devtools/conda-envs/test_env.conda-lock.yml`.
- [ ] Unit tier passes.
- [ ] Workflow tier passes.
- [ ] On a cluster node with an MPI build of AMBER loaded (`mpirun` and `sander.MPI` on `PATH`) and
      the MCL-1 benchmark files present, `pytest -m hpc` passes with neither test skipped. The
      MCL-1 files are not in the repository, and GitHub CI never runs this tier.

## 3. Docs

```bash
pip install -r docs/requirements.txt
make -C docs clean html
```

- [ ] The build finishes, and any new warnings have been read.
- [ ] The new and changed `docs/config.rst` sections and the API pages of new modules render
      (`docs/_build/html`).
- [ ] Markdown pages under `docs/` (for 2.0.0, `docs/in_solvent_cycle.md` and
      `docs/post_production_als.md`) read correctly on GitHub. Sphinx builds only `.rst`
      (`source_suffix = '.rst'` in `docs/conf.py`), so they are not in `_build/html`.

## 4. Integration branch

- [ ] Merge every branch that goes into the release, in merge order and with merge commits, into
      one local integration branch made from `main` (for example `integration/vX.Y.Z`) before
      opening any PR.
- [ ] Resolve conflicts on the source branches, not only on the integration branch, so that each
      PR merges into `main` without conflicts. When the CI branch and another branch both change a
      file under `.github/workflows/`, keep the CI branch's version: it defines the jobs section 5
      waits for.
- [ ] On the integration branch, run sections 2 and 3 again.
- [ ] Check that the current tip of every source branch is merged on the integration branch
      (`git merge-base --is-ancestor <branch> integration/vX.Y.Z` succeeds for each), so the PRs
      carry exactly the commits that were tested. If a branch moved, merge it again and rerun
      sections 2 and 3.
- [ ] Only then open the PRs to `main`, in merge order, with the CI branch first.

For 2.0.0, the integration branch is `integration/v2.0.0`, and the merge order is `ci/fast-tests`,
`land/v2-engine`, `land/v2-paper-protocol`, `release/2.0.0`. The feature commits e7fb60e and
40e45bb changed `.github/workflows/CI.yaml` to a single `test` job whose pytest command names
`test_pilot_mdin.py` and `test_adaptive_insertion.py`, which 7a509c9 deleted. `land/v2-engine` ends
with a commit that restores `main`'s `CI.yaml` (463e16d), so its merge keeps the `ci/fast-tests`
version. `land/v2-paper-protocol` and `release/2.0.0` start after those commits and do not change
`CI.yaml`.

## 5. Merge

- [ ] Wait for both CI jobs (`unit`, `workflow`) to pass on each PR. These jobs are defined in the
      CI branch's `CI.yaml` (for 2.0.0, `ci/fast-tests`). A pull request runs the workflow file of
      its merge with `main`, so the later PRs run them once the CI branch is merged. A PR that
      changes only `docs/`, `scripts/`, Markdown files or the docs workflow does not trigger CI.
- [ ] Merge each PR with a merge commit, not squash or rebase, so the commit SHAs cited in
      `CHANGELOG.md` and the release notes stay valid.
- [ ] After the last merge, check that `main` matches the verified integration branch:
      `git diff integration/vX.Y.Z origin/main` shows nothing unexpected.

## 6. Tags

Push tags by name. Never use `git push --tags`: it would also publish local tags that are not
meant for the remote.

- [ ] Push any older release tag that exists only locally, oldest first. For 2.0.0 that is
      `1.1.3` (on d9b5138, already on `main`):

      ```bash
      git ls-remote --tags origin               # tags already on GitHub
      git push origin 1.1.3
      ```

- [ ] Decide what happens to local pre-release tags. For 2.0.0, `2.0.0rc1` (on 7454d90, the code
      the refine campaign ran) is local only: push it by name on purpose, or keep it local.
- [ ] Tag the merge commit of the last release PR on `main`, not a commit on a feature branch:

      ```bash
      git checkout main && git pull
      git log -1 --format='%h %s'               # the merge commit of the last release PR
      git tag -a X.Y.Z -m "Release X.Y.Z"
      git push origin X.Y.Z
      ```

- [ ] In a clean checkout of the tag,
      `python -c "import implicit_solvent_ddm; print(implicit_solvent_ddm.__version__)"` prints
      `X.Y.Z`.
- [ ] `git archive X.Y.Z` writes the version into `implicit_solvent_ddm/_version.py`
      (`export-subst` in `.gitattributes`).

## 7. GitHub releases

Publish older releases first, so the newest one is published last and GitHub marks it Latest. For
2.0.0, publish 1.1.3 before 2.0.0:

```bash
gh release list                                       # what is already published
gh release create 1.1.3 --title "Implicit Solvent DDM 1.1.3" --notes-file RELEASE_NOTES_v1.1.3.md --latest=false
gh release create X.Y.Z --title "Implicit Solvent DDM X.Y.Z" --notes-file RELEASE_NOTES_vX.Y.Z.md
```

- [ ] Every backlog tag pushed in section 6 has a release, published before X.Y.Z. If one is
      published after X.Y.Z, pass `--latest=false` (or untick "Set as the latest release" on the
      web form), so it does not take Latest from X.Y.Z.
- [ ] Check that the releases page shows X.Y.Z as Latest.

## 8. After the release

- [ ] Add an empty `## [Unreleased]` section at the top of `CHANGELOG.md`.
- [ ] Delete merged release, landing, integration and feature branches that are no longer needed.
