# Contribution Guidelines

Thank you for your interest in contributing to **pangeo-fish**!

## How to contribute?

- Fork the repository.
- Create a feature branch: `git checkout -b feature/my-feature`.
- Commit your changes clearly and concisely.
- Submit a Pull Request against the `main` branch.

For coordinated integration testing, target `DEV` first. Unit tests and the
pinned/latest Pangeo Notebook compatibility checks run on pushes to `DEV` and
pull requests targeting it. Compare scientific results against a recorded
`main` commit using identical inputs and parameters before promoting `DEV` to
`main`. Run the base and complete demos manually on the selected branch when
Copernicus Marine credentials are configured. Docker image publication remains
triggered by `main` pushes and published GitHub Releases.

## Coding Standards

- Follow existing coding conventions.
- Provide tests and documentation for new features.

## Issue Reporting

Report issues clearly with steps to reproduce the problem.

## License of contributions

By submitting a contribution to this repository, you agree that your contribution will be licensed under the same license as **pangeo-fish**. At present, that license is the GNU General Public License v3.0 only (`GPL-3.0-only`).

You confirm that you have the right to submit the contribution under those terms.

## Code of Conduct

Be kind and respectful.

Thank you for helping improve **pangeo-fish**!
