========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/teaspoon/badge/?style=flat
    :target: https://readthedocs.org/projects/teaspoon
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/renatomatz/teaspoon.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/renatomatz/teaspoon

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/renatomatz/teaspoon?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/renatomatz/teaspoon

.. |requires| image:: https://requires.io/github/renatomatz/teaspoon/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/renatomatz/teaspoon/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/renatomatz/teaspoon/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/renatomatz/teaspoon

.. |version| image:: https://img.shields.io/pypi/v/teaspoon.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/teaspoon

.. |wheel| image:: https://img.shields.io/pypi/wheel/teaspoon.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/teaspoon

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/teaspoon.svg
    :alt: Supported versions
    :target: https://pypi.org/project/teaspoon

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/teaspoon.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/teaspoon

.. |commits-since| image:: https://img.shields.io/github/commits-since/renatomatz/teaspoon/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/renatomatz/teaspoon/compare/v0.0.0...master



.. end-badges

A simple workflow for time series predictions.

* Free software: MIT license

Installation
============

::

    pip install teaspoon

You can also install the in-development version with::

    pip install https://github.com/renatomatz/teaspoon/archive/master.zip


Documentation
=============


https://tea_spoon.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
