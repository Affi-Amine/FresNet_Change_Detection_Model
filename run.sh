#!/usr/bin/env bash

set -eu

notebook_path=$bin/$1

# copy the notebook to the execution directory so that it can be updated by quarto
cp $notebook_path main.ipynb


### fetch parameters
# Always provide the working directory
params="-P SRCDIR:$(dirname $notebook_path)"
# Then add remaining parameters
# Start from the second argument: $1 is the notebook path

# Run and render the notebook
quarto render main.ipynb -o output.html --execute $params
