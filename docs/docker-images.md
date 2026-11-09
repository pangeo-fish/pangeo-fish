# Automated notebook images

Pushing or merging to `main` requests a notebook image for that exact commit.
Publishing a GitHub Release (including a prerelease) requests an image for the
release's tagged commit. Publishing a release is separate from updating the
branch named `release`.

The source repository sends a `pangeo-fish-updated` repository dispatch to
[`pangeo-fish/pangeo-fish-docker`](https://github.com/pangeo-fish/pangeo-fish-docker).
That repository builds the image, checks package imports and Jupyter kernel
registration, and publishes it to GHCR:

| Source event             | Image tags                                                          |
| ------------------------ | ------------------------------------------------------------------- |
| Push to `main`           | `<notebook-date>-<commit>` and `latest`                             |
| GitHub Release published | `<notebook-date>-<commit>` and the exact release tag, e.g. `v1.2.3` |

All image names start with `ghcr.io/pangeo-fish/pangeo-fish-notebook:`.
Release builds do not update `latest`, which tracks `main`. A release tag must
be a valid Docker tag: up to 128 ASCII letters, digits, underscores, periods,
or hyphens, starting with a letter, digit, or underscore. `latest` is reserved.
An existing release image tag cannot be reassigned to a different image digest.

## One-time setup

1. Merge the release-tag support in `pangeo-fish/pangeo-fish-docker` first.
2. Create a fine-grained personal access token (or use a GitHub App token) with
   access to **only** `pangeo-fish/pangeo-fish-docker` and repository permission
   **Contents: Read and write**. Follow any organization approval requirements.
3. In **pangeo-fish/pangeo-fish**, add the token as the Actions repository secret
   `DOCKER_IMAGE_DISPATCH_TOKEN` (Settings → Secrets and variables → Actions).
   Renew the secret when its token expires.
4. Merge the source notification workflow. Its `main` push requests the first
   automatic image. Subsequent `main` pushes and published releases request
   images without a manual build step.

The source repository's automatic `GITHUB_TOKEN` cannot dispatch events to a
different repository. This dedicated token is used only for notification;
GHCR publication uses the Docker repository's own `GITHUB_TOKEN`.

The daily check in the Docker repository remains as a fallback for `main`
and for new Pangeo Notebook base images. It does not recover missed release
notifications. After fixing a failed notification, use **Re-run failed jobs**
on the source workflow; its original commit and release tag are retained.

## Check a publication

The source workflow confirms that the build request was accepted, not that the
image is ready. Follow **Build and publish image** in the Docker repository's
[Actions](https://github.com/pangeo-fish/pangeo-fish-docker/actions) for the build,
smoke test, and publication result. If that workflow fails, rerun it there.

For repeatable notebooks, choose a release tag or a `<notebook-date>-<commit>`
tag instead of `latest` in JupyterHub's Custom image field, then select the
**Python (pangeo-fish)** kernel.
