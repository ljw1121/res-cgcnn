# Residual crystal graph convolutional neural network (Res-CGCNN)

This repository implements the residual crystal graph convolutional neural network (Res-CGCNN) introduced in our paper titled "[Accelerating spin Hall conductivity predictions via machine learning](https://onlinelibrary.wiley.com/doi/full/10.1002/mgea.67)". The model that takes as input a crystal structure and predicts SHCs.

The package provides code to train a Res-CGCNN model with a customized dataset. This is built on an existing model [CGCNN](https://github.com/txie-93/cgcnn) which the authors suggest to checkout as well. 

This package provides two major missions:

- Train a Res-CGCNN SHC regeression model and predict SHCs of new crystals with a pre-trained Res-CGCNN model.
- Train a Res-CGCNN SHC classification model and predict SHCs of new crystals with a pre-trained Res-CGCNN model.

## Table of Contents

- [Usage](#usage)
  - [Define a customized dataset](#define-a-customized-dataset)
  - [Train a Res-CGCNN model](#train-a-cgcnn-model)
- [License](#license)
- [Citation](#cite)

# Usage

### Define a customized dataset

To define a customized dataset, you need a list of CIFs for which you want to train the model.
In this work, we use data from [Materials Project](https://www.materialsproject.org/) and [ICSD](https://icsd.products.fiz-karlsruhe.de/) . 

A customized dataset stored in a folder `root_dir` will have the following files:

1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with multiple columns. The first column recodes a unique `ID` for each crystal. From second column onwards the value of respective target property is stored. For eg., if you wish to perform multi-task learning for `Formation energy` and `Band gap`, then the second column should have the target value for `Formation energy` of the crystal and thrid column should have the target value for `Band gap`.

2. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. An example of `atom_init.json` is `data/sample/atom_init.json`, which should be good for most applications.

3. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```
root_dir
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

Two example of customized dataset is provided in the repository at `classificaiton/cif_all_fermi_9249_train/` and `regression/cif_all_fermi_9249_train/`. This contains 9249 samples with SHCs as target properties.

### Train a Res-CGCNN model

Before training a new CGCNN model, you will need to:

[Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest. 

Then, you can train a Res-CGCNN model for your customized dataset by:

```bash
python main.py root_dir
```

You can set the number of training, validation, and test data with labels `--train-size`, `--val-size`, and `--test-size`. Alternatively, you may use the flags `--train-ratio`, `--val-ratio`, `--test-ratio` instead. Note that the ratio flags cannot be used with the size flags simultaneously. For instance, `regression/cif_all_fermi_9249_train/` has 9249 data points in total. You can train a model by:

```bash
python main.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 regression/cif_all_fermi_9249_train/
```

You can also train a classification model with label `--task classification`. For instance, you can use `classificaiton/cif_all_fermi_9249_train/` by:

```bash
python main.py --task classification --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 classification/cif_all_fermi_9249_train/
```

After training, you will get three files in directory.

- `model_best.pth.tar`: stores the Res-CGCNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the Res-CGCNN model at the last epoch.
- `test_results.csv`: stores the `ID`, target values, and predicted values for each crystal in test set.

## Predict properties with a pre-trained Res-CGCNN model

Before predicting the material properties, you will need to:

- [Define a customized dataset](https://github.com/txie-93/cgcnn#define-a-customized-dataset) at `root_dir` for all the crystal structures that you want to predict.
- Obtain a pre-trained Res-CGCNN model named `model_best.pth.tar`.

Two example of customized dataset is provided in the repository at `classificaiton/cif_mp_nomag_5719_test/` and `regression/cif_mp_nomag_5719_test/`. This contains 5719 samples get from [Materials Project](%5Bhttps://www.materialsproject.org/%5D(https://www.materialsproject.org/)) to predict SHCs.

Then, you can predict the properties of the crystals in `root_dir`:

```bash
python predict.py model_best.pth.tar rood_dir
```

After predicting, you will get one file in directory:

- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set. Here the target value is just any number that you set while defining the dataset in `id_prop.csv`, which is not important.

Note that for classification, the predicted values in `test_results.csv` is a probability between 0 and 1.

## License

Res-CGCNN is released under the MIT License.

## Citation

Please consider citing our paper if you use this code in your work

```
@article{https://doi.org/10.1002/mgea.67,
author = {Zhao, Jinbin and Lai, Junwen and Wang, Jiantao and Zhang, Yi-Chi and Li, Junlin and Chen, Xing-Qiu and Liu, Peitao},
title = {Accelerating spin Hall conductivity predictions via machine learning},
journal = {Materials Genome Engineering Advances},
year = {2024},
doi = {https://doi.org/10.1002/mgea.67},
}
```
