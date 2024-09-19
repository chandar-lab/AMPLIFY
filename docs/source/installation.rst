Installation
============

The local installation environment must support:

- Python 3.10
- CUDA 12.x

Given these requirements, the repository functions can be locally built as:

.. code-block:: bash

    python3 -m venv env && \
    source env/bin/activate && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --editable $REPO_DIR[dev]

Where ``$REPO_DIR`` is the location of the source code.

Verify the installation is working with:

.. code-block:: bash

    cd $REPO_DIR && python3 -m pytest
