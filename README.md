# Monkey Species Classification
###  Project carried out by José Jesús Torronteras Hernández, for University of  Rome  "La Sapienza"
 
<br>

## Table of contents
- [Quick start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)

## Quick start

It is necessary to have installed: Python 3.5.2 and download [Monkey Species Dataset](https://www.kaggle.com/slothkong/10-monkey-species) in Input Folder.

The present code has been developed under python3 using Anaconda Notebook. The simplest way to run the program is opening the notebook.

[Monkey Species Classification Notebook Training](./notebooks/Train%20Model.ipynb)

## Project Structure

```
├── config
│   ├── config.ini
├── input
│   ├── monkey_labels.txt
│   ├── images
├── notebooks
│   ├── Train Model.ipynb
|   ├── Test Model.ipynb
├── source
│   ├── aux_functions.py
│   ├── cnn_model.py
│   ├── get_train_test_data.py
├── logs
````

## Usage
Download the dataset from Kaggle [Link](https://www.kaggle.com/datasets/slothkong/10-monkey-species/download?datasetVersionNumber=2) and put it in the Input folder.
Check the configuration file in the config folder.

```python
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
open jupyter notebook
```
