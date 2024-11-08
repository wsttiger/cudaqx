#!/bin/bash

# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

CUDAQX_INSTALL_PREFIX=${CUDAQX_INSTALL_PREFIX:-"$HOME/.cudaqx"}
DOCS_INSTALL_PREFIX=${DOCS_INSTALL_PREFIX:-"$CUDAQX_INSTALL_PREFIX/docs"}
export PYTHONPATH="$CUDAQX_INSTALL_PREFIX:${PYTHONPATH}"
export CUDAQX_DOCS_GEN_IMPORT_CUDAQ=ON 

# Process command line arguments
force_update=""

__optind__=$OPTIND
OPTIND=1
while getopts ":u:" opt; do
  case $opt in
    u) force_update="$OPTARG"
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
  esac
done
OPTIND=$__optind__

# Need to know the top-level of the repo
working_dir=`pwd`
repo_root=$(git rev-parse --show-toplevel)
docs_exit_code=0 # updated in each step

# Make sure these are full path so that it doesn't matter where we use them
docs_build_output="$repo_root/build/docs"
sphinx_output_dir="$docs_build_output/sphinx"
doxygen_output_dir="$docs_build_output/doxygen"
dialect_output_dir="$docs_build_output/Dialects"
rm -rf "$docs_build_output"
doxygen_exe=doxygen

# Generate API documentation using Doxygen
echo "Generating XML documentation using Doxygen..."
mkdir -p "${doxygen_output_dir}"
sed 's@${DOXYGEN_OUTPUT_PREFIX}@'"${doxygen_output_dir}"'@' "$repo_root/docs/Doxyfile.in" | \
sed 's@${CUDAQX_REPO_ROOT}@'"${repo_root}"'@' > "${doxygen_output_dir}/Doxyfile"
"$doxygen_exe" "${doxygen_output_dir}/Doxyfile" 2> "$logs_dir/doxygen_error.txt" 1> "$logs_dir/doxygen_output.txt"
doxygen_exit_code=$?
if [ ! "$doxygen_exit_code" -eq "0" ]; then
    cat "$logs_dir/doxygen_output.txt" "$logs_dir/doxygen_error.txt"
    echo "Failed to generate documentation using doxygen."
    echo "Doxygen exit code: $doxygen_exit_code"
    docs_exit_code=11
fi

echo "Building CUDA-QX documentation using Sphinx..."
cd "$repo_root/docs"
# The docs build so far is fast such that we do not care about the cached outputs.
# Revisit this when caching becomes necessary.

rm -rf sphinx/_doxygen/
rm -rf sphinx/_mdgen/
cp -r "$doxygen_output_dir" sphinx/_doxygen/
# cp -r "$dialect_output_dir" sphinx/_mdgen/ # uncomment once we use the content from those files

rm -rf "$sphinx_output_dir"
sphinx-build -v -n -W --keep-going -b html sphinx "$sphinx_output_dir" -j auto #2> "$logs_dir/sphinx_error.txt" 1> "$logs_dir/sphinx_output.txt"
sphinx_exit_code=$?
if [ ! "$sphinx_exit_code" -eq "0" ]; then
    echo "Failed to generate documentation using sphinx-build."
    echo "Sphinx exit code: $sphinx_exit_code"
    echo "======== logs ========"
    cat "$logs_dir/sphinx_output.txt" "$logs_dir/sphinx_error.txt"
    echo "======================"
    docs_exit_code=12
fi

rm -rf sphinx/_doxygen/
rm -rf sphinx/_mdgen/

mkdir -p "$DOCS_INSTALL_PREFIX"
if [ "$docs_exit_code" -eq "0" ]; then
    cp -r "$sphinx_output_dir"/* "$DOCS_INSTALL_PREFIX"
    touch "$DOCS_INSTALL_PREFIX/.nojekyll"
    echo "Documentation was generated in $DOCS_INSTALL_PREFIX."
    echo "To browse it, open this url in a browser: file://$DOCS_INSTALL_PREFIX/index.html"
else
    echo "Documentation generation failed with exit code $docs_exit_code."
    echo "Check the logs in $logs_dir, and the documentation build output in $docs_build_output."
fi

cd "$working_dir" && (return 0 2>/dev/null) && return $docs_exit_code || exit $docs_exit_code
