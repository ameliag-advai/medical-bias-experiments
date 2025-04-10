# Project Alethia
A template repository for research repositories.

[<img src="https://img.shields.io/badge/python-3.8-green">](https://img.shields.io/badge/python-3.8-green)
[<img src="https://img.shields.io/badge/python-3.9-green">](https://img.shields.io/badge/python-3.9-green)
[<img src="https://img.shields.io/badge/python-3.10-green">](https://img.shields.io/badge/python-3.10-green)

[![CICD](https://github.com/Advai-Ltd/advai_python_template/actions/workflows/cicd.yaml/badge.svg)](https://github.com/Advai-Ltd/advai_python_template/actions/workflows/cicd.yaml)


## Usage

*This section provides guidance on how to use this template for your research projects. It links to various files and resources located in the `template_startup` directory.*

#### Setting Up the Base Environment

*1. Install the base environment from the `re.yml` (research environment) file and create a Conda environment with a custom name of your choice:*

``conda env create -f re.yml --name <your_environment_name>``

*2. Activate the environment:*

``conda activate <your_environment_name>``

*3. Feel free to install any additional packages you need for your project.*

### Creating a YAML File for the Final Project

*Once you have installed all the necessary packages, create a YAML file for the final project:*

``conda env export --no-builds > environment.yml``

*If you wish to update the conda environment .yml file (i.e. if you install any new packages or delete any), then you can do so using the following:*

``conda env update --name <your_environment_name> --file final_environment.yml --prune``

*You can also view existing installed packages using ``pip freeze``*.

### Project Structure

*The code must be delivered in a Python source layout in the `src` directory and should conform to the specified format; It muse use the advai namespace package as the parent package, as well as subpackages for relevant code. For more details, see the [library_structure_README.md](library_structure_README.md) file*


### Notebooks and Code Delivery

*Feel free to create as many Jupyter notebooks as you want in the `notebooks` directory for experimentation and prototyping. Ultimately, however, the code should be delivered in the Python source layout within the `src` directory.*

*At the end of the project, create one or more instructional notebooks in the `notebooks` directory. These notebooks should allow someone new to the project to understand and reproduce your work. Ensure that the notebooks are clean, well-documented, and that all bespoke objects and functions are imported from the top-level Python `src` directory.*

*Here is a [template for Instructional Jupyter notebook](instructional_notebooks/template_notebook.ipynb) that you can use as a starting point.*

_Also you will need to fill out all the sections in this this readme file (see below). You'll need to fill them out with the relevant information for your project. Inlcuding the following sections: **Introduction**, **Goals and Objectives**, **Prerequisites**, **Installation and Setup Instructions**, **Usage Instructions**, **Guidelines for Contributing**, and **Contact Information**._

_**Note:** Please remove this temporary section from the `README.md` file once you've finished your project._


## Research To Development Checklist (R2D2)

__Documentation__
- [ ] README
- [ ] Notebooks
- [ ] Docstrings

__Code__
- [ ] Import Paths
- [ ] Workbench Relocating/Refactoring

__Tests__
- [ ] Unit Tests
- [ ] Integration Tests
- [ ] Resources

__Pipeline__
- [ ] Build
- [ ] Test
- [ ] Static Documentation Analysis
- [ ] Static Code Analysis

## License
The license for this project can be found here: [License.txt](LICENSE.txt)
## Changelog
The changelog for this repository can be found here: [Changelog.md](CHANGELOG.md)
## Authors & Acknowledgement
- Jord-Advai (Jordan)
- damianruck (Damian)
### Maintainers
- Jord-Advai (Jordan)
- damianruck (Damian)
### Contributors

--------------------------------------------

# Project Name

## Introduction

Provide a brief introduction to the project. Explain the background, motivation, and a high-level overview of what the project aims to achieve.

## Goals and Objectives

List the main goals and objectives of the project. Clearly outline what the project aims to accomplish and any specific milestones you plan to reach.

## Prerequisites

List any required software, hardware, or data needed to run the project. Include version numbers, operating systems, and any other relevant information. Also, mention any optional dependencies that can enhance the project's functionality.

Make sure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

## Installation and Setup Instructions

Provide detailed instructions on how to set up the project. Include step-by-step instructions for installing dependencies, setting up the environment, and configuring any necessary files.

1. Clone the repository:

``git clone <repository_url>``

2. Create a Conda environment and activate it:

``conda env create -f environment.yml``
``conda activate <your_environment_name>``

3. Add any additional setup instructions specific to your project.

## Usage Instructions

Explain how to use the project once it has been set up. Include example commands, input files, or any other information needed to run the project and obtain the desired results.

## Guidelines for Contributing

Mention the steps to contribute to the project, such as creating a fork, cloning the repository, creating a new branch, making changes, and submitting a pull request. Refer to the `CONTRIBUTING.md` file for more detailed information on the contribution process.

## Contact Information

Provide the contact information for the project lead, including their name, email address, and any other relevant contact details. Encourage users to reach out with questions, feedback, or suggestions.


