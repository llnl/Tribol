.. ##
.. ## Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory
.. ##

******************
Tribol Git Aliases
******************

This section provides an overview of the various git aliases that are available
upon running the SetupForDevelopment script.

.. list-table:: Listing of Git Aliases
   :widths: 25 25 50
   :header-rows: 1

   * - Git Alias
     - Description
     - Usage Example
   * - **git hall-of-fame**
     - Lists all contributors ranked by the number of commits
     - > git hall-of-fame
   * - **git incoming**
     - Lists all commits that are on origin/master and are not on the current branch
     - > git incoming
   * - **git outgoing**
     - Lists all commits on the current branch that are not on master    
     - > git outgoing  
   * - **git rmtag <tag>**
     - Removes a tag both on the local clone and on origin
     - > git rmtag 3.9.260
   * - **git make-patch <from-tag> <patch_file>**
     - Creates patch file consisting of all the commits from the given tag to the HEAD of the current branch.
     - > git make-patch 3.9.260 ~/fix.patch 
   * - **git apply-patch <patch_file>**
     - Applies the changes from the given patch
     - > git apply-patch fix.patch    
   * - **git changelog <from> <to>**
     - Generates and prints a changelog to STDOUT consisting of the changes in between two different commits
       specified with the <from> and <to> arguments. The <from> and <to> can be any git ref object, e.g, a
       commit SHA1 or a tag.
     - > git changelog 3.9.1 3.9.100
   * - **git unstage <file>**
     - Unstages the changes on the given file.
     - > git unstage src/main.cpp  
   * - **git history**
     - Displays the history on the current branch
     - > git history
   * - **git show-config**
     - Lists the contents of the git-configuration, i.e., ./git/config for this project     
     - > git show-config
   * - **git show-aliases**
     - Displays the current set of git aliases
     - > git show-aliases    
   * - **git prepush**
     - Provides a snapshot of the changes to be pushed
     - > git prepush
   * - **git undo**
     - Removes the last commit on the current branch, but leaves the
       the changes unstaged in the checkout space
     - > git undo     
   * - **git squash-merge <branch>**
     - Merges the given branch by squashing all the commits on that branch in to a single commit at 
       the HEAD of the current branch
     - > git squash-merge feature/stuff  
   * - **git ff-merge <branch>**
     - Merges the given branch by doing a fast-forward merge 
     - > git ff-merge feature/stuff  
