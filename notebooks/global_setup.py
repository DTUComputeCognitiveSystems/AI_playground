# Set working direction to project root
import git
import os
git_root = git.Repo('.', search_parent_directories=True).git.rev_parse("--show-toplevel") # '.' causes issue on windows/osx?
os.chdir(git_root)