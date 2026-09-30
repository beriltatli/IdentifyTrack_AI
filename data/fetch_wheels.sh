#!/bin/zsh
# Resumable downloads: curl -C - continues after every connection reset.
cd "$(dirname "$0")/wheels"
get() { # url file
  until curl -sfL -C - --retry 100 --retry-all-errors --retry-delay 2 -m 0 -o "$2" "$1"; do echo "retry $2"; sleep 2; done
  echo "done $2 $(stat -f%z "$2") bytes"
}
get "https://files.pythonhosted.org/packages/43/19/23a1aed488423a5055727256b25406e4b93bd2bcf1352bef582b9951c10c/torch-2.14.1-cp314-cp314-macosx_14_0_arm64.whl" torch-2.14.1-cp314-cp314-macosx_14_0_arm64.whl
echo ALLDONE
