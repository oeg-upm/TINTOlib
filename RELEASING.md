# Release format and process

This document is a maintainer guide, not the release history. Record changes
for users in `CHANGELOG.md` and publish release announcements on GitHub.
`RELEASING.md` is a conventional filename, not a Python packaging requirement.

Write release notes in English and describe changes relative to the previous
release tag. For new changelog entries, follow
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Preserve historical
entries as published; older releases used project-specific headings.

## Naming

- Git tag: `vX.Y.Z`.
- GitHub release title: `TINTOlib X.Y.Z`.
- Changelog entry and release body start with `## X.Y.Z - YYYY-MM-DD`.
- Use the actual release date. Base the GitHub body on the matching changelog
  entry; it may also include installation information.

## Structure

Use the following Keep a Changelog categories; omit empty categories:

1. `Added`: new functionality.
2. `Changed`: changes to existing functionality, including dependency changes.
3. `Deprecated`: functionality scheduled for removal.
4. `Removed`: removed functionality.
5. `Fixed`: bug fixes, with the affected method and resulting behavior.
6. `Security`: vulnerability fixes.

Keep an `Unreleased` section at the top when accumulating changes for the next
version. Explain migration requirements with the relevant change. An optional
`Installation` section can follow the changelog content in GitHub announcements;
it is a project convention, not a Keep a Changelog category.

Public notes should explain what changed and how it affects users. Keep routine
test results, temporary diagnostics, and lists of checks not executed in the PR
description or an internal validation record, rather than a public `Validation`
section. Document known issues and limitations when they affect users, along
with any workaround. If generated images change intentionally, describe the
reason and impact. Omit internal task IDs and review history unless needed to
explain the change.

## Template

````markdown
## X.Y.Z - YYYY-MM-DD

TINTOlib X.Y.Z fixes ...

### Changed

- Describe any dependency change or intentional behavior change.

### Fixed

- Fix ... in `Method`, so that ...
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
   `git diff --check`. Record commands, results, coverage limits and diagnostic
   provenance in the PR or internal validation record. Do not claim a check
   passed unless it was executed successfully.
5. Build wheel and source distribution in a fresh output directory, run
   `python -m twine check`, inspect their contents, and smoke-test the wheel
   from outside the source tree. Use the project's build backend in
   `pyproject.toml`.
6. Commit the reviewed release metadata using a title such as
   `chore(release): prepare X.Y.Z`; publish the commit through the repository's
   approved workflow. Do not overwrite existing tags or release artifacts.
7. Create an annotated tag `vX.Y.Z` on that exact commit, then create the GitHub
   release titled `TINTOlib X.Y.Z` using the user-facing changelog entry, with
   optional installation details.
8. Publish the built artifacts to PyPI only when that action is authorized and
   credentials are available. Verify the published version in a fresh
   environment. GitHub and PyPI publication are separate steps.

## References

- [Keep a Changelog](https://keepachangelog.com/en/1.1.0/): user-facing change history.
- [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/): building and publishing distributions.
- [pip release process](https://pip.pypa.io/en/stable/development/release-process/): an example of separate maintainer documentation.
- [Pipenv repository](https://github.com/pypa/pipenv): an example using `RELEASING.md`.
