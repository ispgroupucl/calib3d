#!/bin/bash

# build documentation
pdoc calib3d -o docs/ -c latex_math=True --force --html
git add -u docs/
git commit -m "Update documentation"

# Extract version number from setup.py
RELEASE_NUM=`grep version setup.py | cut -d\" -f2 | cut -d\' -f2`
echo "RELEASE_NUM=$RELEASE_NUM"

# Push to PyPi
python setup.py sdist
twine upload dist/calib3d-$RELEASE_NUM.tar.gz

# Tag in Git and push to remote
git tag -f $RELEASE_NUM -m "Tagging release $RELEASE_NUM"
git push -f --tags

