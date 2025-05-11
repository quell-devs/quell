================================================================
Quell
================================================================

.. start-badges

|testing badge| |docs badge| |black badge| |torchapp badge|

.. |testing badge| image:: https://github.com/quell-devs/quell/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/quell-devs/quell/actions

.. |docs badge| image:: https://github.com/quell-devs/quell/actions/workflows/docs.yml/badge.svg
    :target: https://quell-devs.github.io/quell
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |torchapp badge| image:: https://img.shields.io/badge/MLOpps-torchapp-B1230A.svg
    :target: https://rbturnbull.github.io/torchapp/
    
.. end-badges

.. start-quickstart

An app for supervised denoising of 3D images.

Installation
==================================

Install using pip:

.. code-block:: bash

    pip install git+https://github.com/quell-devs/quell.git


Usage
==================================

See the options for training a model with the command:

.. code-block:: bash

    quell train --help

See the options for making inferences with the command:

.. code-block:: bash

    quell --help

.. end-quickstart

Sample data
==================================

Generate and preview sample data with the following commands:

.. code-block:: bash

    quell-sample-data 
    quell-preview-data

Development
==================================

Visit the `poetry docs <https://python-poetry.org/docs/>`_ for more information on how to install poetry.

Install using poetry:

.. code-block:: bash

    git clone https://github.com/quell-devs/quell.git
    cd quell
    poetry install


Credits
==================================

.. start-credits

Ashkan Pakzad, Robert Turnbull, Simon Mutch, Tim Gureyev

Created using torchapp (https://github.com/rbturnbull/torchapp).

.. end-credits

