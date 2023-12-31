# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
Types of changes
* "Added" for new features.
* "Changed" for changes in existing functionality.
* "Deprecated" for soon-to-be removed features.
* "Removed" for now removed features.
* "Fixed" for any bug fixes.
* "Security" in case of vulnerabilities.
-->

## [Unreleased]
## [1.0.6] - 2023-09-25
### Fixed
- Vision pretrained models from model zoo problems on forward and preprocess methods
## [1.0.5] - 2023-09-24
### Fixed
- PretrainedSampleEncoder with ImageSamples not working, bugfix
## [1.0.4] - 2023-09-23
### Fixed
- Actually fixes what 1.0.3 wanted to do:
    - Models package not included with releases (`supertriplets.models`)
## [1.0.3] - 2023-09-23
### Fixed
- Models package not included with releases (`supertriplets.models`) _(did not work, see 1.0.4)_
## [1.0.2] - 2023-09-08
### Fixed
- Relative paths on README.md
- Missing PyPI package link

## [1.0.1] - 2023-09-08
### Fixed
- Fix readme image not displaying in PyPI
## [1.0.0] - 2023-09-08
### Added
- pytest tests for each module
- tinyimmdb dataset with model convergence test
- README.md updated with examples and library description
- Docstrings on most classes and functions
- Added classifiers on PyPI
### Changed
- Major refactor
- API changes
- Many bugfixes
## [0.1.0] - 2023-08-19
### Added

- Created initial (not production ready) version of SuperTriplets