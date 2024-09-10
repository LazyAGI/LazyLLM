## Code Contribution Guide
### Contributing Code

As a rapidly developing open-source project, we warmly welcome all forms of contributions, whether it is documentation improvement, adding new test cases, adding new tools, enhancing or adding new features, or improving the infrastructure.

#### Bug Fixes
The steps to fix code implementation errors are as follows:

1. If the submitted code changes are significant, it is recommended to first submit an issue. Properly describe the issue's phenomenon, cause, and reproduction method, and confirm the fix plan after discussion.
2. Fix the bug and supplement with corresponding unit tests, then submit a ``pull request``.

#### New Features or Components

1. If the new feature or module involves significant code changes, it is recommended to first submit an issue to confirm the necessity of the feature.
2. Implement the new feature and submit a ``pull request``.

#### Documentation Supplement

Fixes to documentation can be submitted directly via a ``pull request``. The steps for adding documentation or translating documentation into other languages are as follows:

1. Submit an issue to confirm the necessity of adding the documentation.
2. Add the documentation and submit a ``pull request``.

### Contribution Process

If you're not familiar with pull requests, don't worry. The following content will guide you step-by-step on how to create a ``pull request`` from scratch. For a deeper understanding of the ``pull request`` development model, you can refer to [GitHub's tutorial](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

#### Fork the Repository

When you submit a pull request for the first time, start by forking the original repository. Click the Fork button at the top right of the GitHub page. The forked repository will appear under your GitHub personal profile.

Clone the Repository Locally

``git clone git@github.com:{username}/LazyLLM.git``

Add the Original Repository as an Upstream Repository

``git remote add upstream git@github.com:LazyAGI/LazyLLM.git``

Check if the remote has been added successfully by typing ``git remote -v`` in the terminal.

```
origin	git@github.com:{username}/LazyLLM.git (fetch)
origin	git@github.com:{username}/LazyLLM.git (push)
upstream	git@github.com:LazyAGI/LazyLLM.git (fetch)
upstream	git@github.com:LazyAGI/LazyLLM.git (push)
```

Here's a quick introduction to origin and upstream. When we use git clone to clone code, a remote called origin is created by default, which points to the repository we cloned. Upstream is something we add ourselves, and it points to the original repository. Of course, if you don't like the name upstream, you can change it to something else, like LazyLLM. We usually push code to origin (i.e., the forked remote repository), and then submit a pull request to upstream. If the code we submit conflicts with the latest code, we pull the latest code from upstream, resolve the conflict with our local branch, and then push to origin.

#### Create a Development Branch

You need to create a development branch based on main. The recommended naming convention for branches is ``username/pr_name``.

``git checkout -b username/pr_name``

During development, if the local main branch is behind the upstream main branch, you need to first pull the code from upstream to synchronize, and then execute the command above.

``git pull upstream main``

#### Style and Testing
The submitted code needs to pass unit tests to ensure correctness and maintain proper code style. You can execute the following commands:

```
pytest -vxs tests
python -m flake8
```

#### Push Code to Remote
Push your code to the remote repository. If it's the first time you're pushing, you can add the ``-u`` parameter after ``git push`` to associate the remote branch.

```
git push -u origin {branch_name}
```
Push your code to the remote repository. If it's the first time you're pushing, you can add the -u parameter after ``git push`` to associate the remote branch.

#### Submit a Pull Request (PR)

- Create a pull request on the GitHub Pull request page
- Modify the PR description according to the guide, so that other developers can better understand your changes.

!!! Note

      - The PR description should include the reason for the changes, the content of the changes, the impact of the changes, and link to related issues.
      - Once all reviewers agree to merge the PR, we will merge it into the main branch as soon as possible.

#### Resolve Conflicts

Over time, our codebase will continuously update. If your PR conflicts with the main branch, you need to resolve the conflicts. There are two ways to do this:

```
git fetch --all --prune
git rebase upstream/main
```
or
```
git fetch --all --prune
git merge upstream/main
```
If you are very skilled at handling conflicts, you can use the rebase method to resolve conflicts, as this keeps your commit log clean. If you are not familiar with using rebase, you can use the merge method to resolve conflicts.

### Pull Request Guidelines

1. One ``pull request`` per Short-Term Branch

2. Granularity Should Be Small,Each pull request should focus on a single task, avoiding overly large pull requests.

      - Bad：Adding all operators needed for multiple models in one PR.
      - Acceptable：Implementing one or several related operators in one PR.
      - Good：Fixing a bug that occurs when the input of an operator is empty.

3. Provide Clear and Meaningful Commit Messages

4. Provide Clear and Meaningful ``pull pequest`` Descriptions

      - Provide Clear and Meaningful Pull Request Descriptions: ``[Prefix] Short description of the pull request (Suffix)``
      - prefix: New feature  ``[Feature]``, Bug fix ``[Fix]``, Documentation-related ``[Docs]``,Work in progress ``[WIP]`` (not ready for review)
      - The description should introduce the main changes, results, and impacts on other parts of the project, following the ``pull request`` template
      - Link to related ``issues`` and other ``pull requests``

5. If introducing third-party libraries or borrowing code from third-party libraries, ensure their licenses are compatible with LazyLLM.Add a note to the borrowed code:  ``This code is inspired from http://``
