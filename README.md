# Quell

[![arxiv](https://img.shields.io/badge/arXiv-2505.05812-b31b1b.svg)](https://arxiv.org/abs/2505.05812)
[![testing](https://github.com/quell-devs/quell/actions/workflows/testing.yml/badge.svg)](https://github.com/quell-devs/quell/actions)
[![docs](https://github.com/quell-devs/quell/actions/workflows/docs.yml/badge.svg)](https://quell-devs.github.io/quell)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![torchapp](https://img.shields.io/badge/MLOpps-torchapp-B1230A.svg)](https://rbturnbull.github.io/torchapp/)

An app for supervised denoising of 3D images.

## Installation

Install using pip:

``` bash
pip install git+https://github.com/quell-devs/quell.git
```


## Usage

See the options for training a model with the command:

``` bash
quell train --help
```

See the options for making inferences with the command:

``` bash
quell --help
```


## Sample data

Generate and preview sample data with the following commands:

``` bash
quell-sample-data 
quell-preview-data
```

## Development

Visit the [Poetry documentation](https://python-poetry.org/docs/) for more information on how to install poetry.

Install using poetry:

``` bash
git clone https://github.com/quell-devs/quell.git
cd quell
poetry install
```


## Credits

If you use this software in your research, please cite:

> 📄 **Towards order of magnitude X-ray dose reduction in breast cancer imaging using phase contrast and deep denoising**
>
> *Ashkan Pakzad, Robert Turnbull, Simon J. Mutch, Thomas A. Leatham, Darren Lockie, Jane Fox, Beena Kumar, Daniel Häsermann, Christopher J. Hall, Anton Maksimenko, Benedicta D. Arhatari, Yakov I. Nesterets, Amir Entezam, Seyedamir T. Taba, Patrick C. Brennan, Timur E. Gureyev, Harry M. Quiney*
>
> arXiv:2505.05812 (2025) • [Read Paper](https://arxiv.org/abs/2505.05812)


``` bibtex
@misc{2505.05812,
  Author = {Ashkan Pakzad and Robert Turnbull and Simon J. Mutch and Thomas A. Leatham and Darren Lockie and Jane Fox and Beena Kumar and Daniel Häsermann and Christopher J. Hall and Anton Maksimenko and Benedicta D. Arhatari and Yakov I. Nesterets and Amir Entezam and Seyedamir T. Taba and Patrick C. Brennan and Timur E. Gureyev and Harry M. Quiney},
  Title = {Towards order of magnitude X-ray dose reduction in breast cancer imaging using phase contrast and deep denoising},
  Year = {2025},
  Eprint = {arXiv:2505.05812},
}
```

Created using torchapp (https://github.com/rbturnbull/torchapp).

