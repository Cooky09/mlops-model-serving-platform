Contributing Guide

Development Workflow

All project changes should follow a consistent GitHub-based workflow.

Issue → Branch → Commit → Pull Request → CI → Merge

1. Create or identify an issue

Every meaningful change should be associated with a GitHub Issue.

Use the available issue templates for:

* Bug reports
* Feature requests

Describe the problem, proposed change, and expected outcome clearly.

2. Create a feature branch

Create a dedicated branch from main.

Branch naming convention:

feat/<short-description>
fix/<short-description>
chore/<short-description>
docs/<short-description>

Examples:

feat/model-drift-detection
fix/prediction-validation
chore/update-ci
docs/development-workflow

Do not make development changes directly on main.

3. Make focused commits

Keep commits small and focused on one logical change.

Use descriptive commit messages, for example:

feat: add model drift detection
fix: handle invalid prediction input
chore: improve CI checks
docs: update development workflow

4. Open a Pull Request

Push the branch to GitHub and open a Pull Request against main.

Every Pull Request should:

* Explain what changed
* Reference the related issue
* Describe testing performed
* Identify any MLOps impact
* Pass the repository CI checks

5. CI validation

Pull Requests are validated automatically by GitHub Actions.

CI should verify at minimum:

* Code quality / linting
* Automated tests

Additional checks may be added as the project evolves.

A Pull Request should not be merged while required CI checks are failing.

6. Review and merge

Changes should be reviewed through the Pull Request before being merged into main.

Once required checks pass and the change is ready, merge the Pull Request.

7. Keep the repository history clean

Avoid committing generated files, credentials, secrets, local environment files, or unrelated changes.

Keep changes focused and maintain documentation when project behavior or development procedures change.

Engineering Principles

This project aims to follow production-oriented engineering practices:

* Reproducible development
* Automated testing
* Continuous integration
* Explicit code review
* Small, traceable changes
* Clear documentation
* Secure handling of credentials and configuration
* Separation of application, ML, infrastructure, and deployment concerns

The workflow will evolve as the MLOps platform becomes more sophisticated.
