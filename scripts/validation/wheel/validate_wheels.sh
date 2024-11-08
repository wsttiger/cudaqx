#!/bin/bash

# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -e

CURRENT_ARCH=$(uname -m)
PYTHON_VERSIONS=("3.10.13" "3.11.7" "3.12.1")
TARGETS=("nvidia" "nvidia --option fp64", "qpp-cpu")
FINAL_IMAGE="ghcr.io/nvidia/cudaqx-private-wheels-test:latest"

# Function to run Python tests
run_python_tests() {
    local container_name=$1
    local python_version=$2

    echo "Running Python tests for Python ${python_version} with target..."

    # Install pytest and other test dependencies
    docker exec ${container_name} bash -c "\
        PYENV_VERSION=${python_version} pyenv exec pip install pytest networkx --user"

    # Clone repository and run tests with specific target
    docker exec ${container_name} bash -c "\
        PYENV_VERSION=${python_version} pyenv exec python3 -m pytest /workspace/cudaqx-private/libs/ -v"

    local test_result=$?
    if [ ${test_result} -ne 0 ]; then
        echo "Python tests failed for Python ${python_version}"
        return 1
    fi

    echo "Python tests completed successfully for Python ${python_version}"
    return 0
}

# Function to test examples
test_examples() {
    local tag=$1
    local container_name="cudaqx-test-$(date +%s)"
    
    echo "Testing examples in ${tag}..."
    # Start container with a command that keeps it running
    docker run --net=host -d -it --name ${container_name} --gpus all ${tag}
    
    # Wait for container to be fully up
    sleep 2
    
    # Verify container is running
    if ! docker ps | grep -q ${container_name}; then
        echo "Container failed to start properly"
        docker logs ${container_name}
        return 1
    fi

    # Loop through Python versions
    for python_version in "${PYTHON_VERSIONS[@]}"; do
        echo "Testing with Python version: ${python_version}"
        
         # Run Python tests first
        if ! run_python_tests ${container_name} ${python_version} ${target}; then
            docker stop ${container_name}
            docker rm ${container_name}
            return 1
        fi

        # Loop through targets
        for target in "${TARGETS[@]}"; do
            echo "Testing with target: ${target}"

            # Test Python examples
            for domain in "solvers" "qec"; do
                if docker exec ${container_name} bash -c "[ -d /workspace/cudaqx-private/examples/${domain}/python ] && [ -n \"\$(ls -A /workspace/cudaqx-private/examples/${domain}/python/*.py 2>/dev/null)\" ]"; then
                    echo "Testing ${domain} Python examples with Python ${python_version} and target ${target}..."
                    if ! docker exec ${container_name} bash -c "cd /workspace/cudaqx-private/examples/${domain}/python && \
                        for f in *.py; do \
                            echo Testing \$f...; \
                            PYENV_VERSION=${python_version} pyenv exec python3 \$f --target ${target} || exit 1; \
                        done"; then
                        echo "Python tests failed for ${domain} with Python ${python_version} and target ${target}"
                        docker stop ${container_name}
                        docker rm ${container_name}
                        return 1
                    fi
                else
                    echo "Skipping ${domain} Python examples - directory empty or not found"
                fi
            done
        done
    done
    
    # Cleanup
    docker stop ${container_name}
    docker rm ${container_name}
}

# Main execution
echo "Starting CUDA-Q image validation for ${CURRENT_ARCH}..."

tag="${FINAL_IMAGE}-${CURRENT_ARCH}"
test_examples ${tag} || {
    echo "Tests failed for Python on ${CURRENT_ARCH}"
    exit 1
}

echo "Validation complete successfully for ${CURRENT_ARCH}!"