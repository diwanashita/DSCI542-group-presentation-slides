# Contributing to Group 04 Project

## Clone the Repository
To clone the repository into your local machine, you can run the following command:

```
git clone https://github.com/UBC-MDS/group04.git
```

## Creating a Branch
Once your local environment is up-to-date, you can create a new git branch which will contain your contribution (always create a new branch instead of making changes to the main branch):
```
git switch -c <your-branch-name>
```

With this branch checked-out, make the desired changes to the package.
When you are happy with your changes, you can commit them to your branch by running

## Creating a Pull Request (PR)
```
git add <modified-file>
git commit -m "Some descriptive message about your change"
git push origin <your-branch-name>
```

You will then need to submit a pull request (PR) on GitHub asking to merge your example branch into the main branch. For details on creating a PR see GitHub documentation Creating a pull request. You can add more details about your example in the PR such as motivation for the example or why you thought it would be a good addition. You will get feedback in the PR discussion if anything needs to be changed. To make changes continue to push commits made in your local example branch to origin and they will be automatically shown in the PR.