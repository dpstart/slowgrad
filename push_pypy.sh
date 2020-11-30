#!/bin/bash -e
rm -rf dist
pipenv run python setup.py sdist bdist_wheel
twine upload dist/*