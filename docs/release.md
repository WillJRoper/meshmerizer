# Release and Deployment

This page describes the release model used by the repository.

## Recommended release flow

1. make sure `main` is in a releasable state,
2. ensure CI is green,
3. create and push a version tag like `v0.1.0`,
4. let GitHub Actions build and publish the release,
5. verify the package on PyPI and the docs on GitHub Pages.

The canonical deployment path is now GitHub Actions plus PyPI trusted
publishing, following the same general pattern used in `synthesizer`.

## GitHub workflows

The repository now uses three main workflows:

- `test.yml`
  - runs Ruff and pytest on pull requests and pushes to `main`
- `docs.yml`
  - builds MkDocs and deploys the site to GitHub Pages
- `deploy.yml`
  - builds the release sdist on version tags and publishes it to PyPI

## Tag-driven release

Create a release tag and push it:

```bash
git tag v0.1.0
git push origin v0.1.0
```

That tag triggers:

- package build and PyPI publish,
- documentation build and deployment.

## PyPI trusted publishing

The deploy workflow is designed to use GitHub Actions OIDC trusted publishing
rather than storing a PyPI username and password in repository secrets.

To enable this, configure the PyPI project to trust this GitHub repository and
the `pypi-deployment` environment.

## Local validation

If you want to validate a release locally before tagging, the key steps are:

```bash
pytest
python -m build --sdist
```

You can also build the docs locally with:

```bash
mkdocs build --clean --strict
```

## Notes

For public releases, prefer:

- tag-driven GitHub Actions publishing,
- trusted publishing instead of password-based uploads,
- verifying install and CLI/API smoke tests after release.
