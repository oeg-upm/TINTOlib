# Release format and process

This convention follows the published **TINTOlib 1.3.2** release. Write release
notes in English and describe changes relative to the previous release tag.

## Naming

- Git tag: `vX.Y.Z`.
- GitHub release title: `TINTOlib X.Y.Z`.
- Changelog entry and release body start with `## X.Y.Z - YYYY-MM-DD`.
- Use the actual release date. Keep the GitHub body identical to the matching
  `CHANGELOG.md` entry.

## Structure

Start with a short paragraph explaining the user-visible result. Use the
following sections in this order; omit sections that do not apply:

1. `Fixed`: bug fixes, with the affected method and resulting behavior.
2. `Added`: new functionality, when present.
3. `Compatibility and dependencies`: deprecations, migration requirements,
   dependency changes, and intentional changes to generated images.
4. `Validation`: checks actually executed, their scope, and material limits.
5. `Installation`: an exact-version installation command once available on PyPI.

Do not claim tests passed merely because they were added. Distinguish tests
already in the repository from temporary diagnostics. If comparison images
change intentionally, describe the reason instead of claiming exact matches.
Do not list internal task IDs or review history unless they explain the change.

## Template

````markdown
## X.Y.Z - YYYY-MM-DD

TINTOlib X.Y.Z fixes ...

### Fixed

- Fix ... in `Method`, so that ...

### Compatibility and dependencies

- Describe any required migration or intentional behavior change.

### Validation

- State the commands/checks actually run and their outcomes.
- State relevant coverage limits.

### Installation

```bash
pip install TINTOlib==X.Y.Z
```
````

If the GitHub release precedes PyPI publication, label the installation command
as pending availability on PyPI. A GitHub release does not publish the Python
package to PyPI.

## Preparing a release

1. Inspect the previous release and `git diff vPREVIOUS..HEAD`. Confirm the
   intended commit, a clean worktree, and the applicable repository rules.
2. Synchronize `pyproject.toml`, `CITATION.cff`, and `codemeta.json`: version,
   release dates, tag links, PyPI links, archive URL, and release-notes URL.
   Preserve metadata about earlier articles and the project's creation date.
3. Add the matching `CHANGELOG.md` entry using the structure above.
4. Run relevant tests and inspect any intentional image changes. Run
   `git diff --check`.
5. Build wheel and source distribution in a fresh output directory, run
   `python -m twine check`, inspect their contents, and smoke-test the wheel
   from outside the source tree. Use the project's build backend in
   `pyproject.toml`.
6. Commit the reviewed release metadata using a title such as
   `chore(release): prepare X.Y.Z`; publish the commit through the repository's
   approved workflow. Do not overwrite existing tags or release artifacts.
7. Create an annotated tag `vX.Y.Z` on that exact commit, then create the GitHub
   release titled `TINTOlib X.Y.Z` using the changelog entry as its body.
8. Publish the built artifacts to PyPI only when that action is authorized and
   credentials are available. Verify the published version in a fresh
   environment. GitHub and PyPI publication are separate steps.

Published format reference:
<https://github.com/oeg-upm/TINTOlib/releases/tag/v1.3.2>.
