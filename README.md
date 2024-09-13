# Haskap Berry Antioxidant Capacity Prediction Model

This repository contains the code for the paper "Determination of antioxidant capacity and phenolic content of haskap berries (Lonicera caerulea L.) by attenuated total reflectance-Fourier transformed-infrared spectroscopy"

## Overview

This project uses machine learning models, primarily based on TensorFlow 2.16, to predict the antioxidant capacity and phenolic content of haskap berries using spectroscopy data. The main components of this project include:

1. Data preprocessing and model training (`train.py`)
2. Model evaluation with various machine learning algorithms (`test_other_model.py`)
3. Train-validation-test split for model assessment (`train_val_test.py`)
4. Recalculation of performance indicators (`recalculate_indicator.py`)

## Requirements

- Python 3.7+
- TensorFlow 2.16
- NumPy
- Pandas
- Scikit-learn
- Keras

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/haskap-berry-antioxidant-prediction.git
   cd haskap-berry-antioxidant-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data in Excel format (.xlsx) and place it in the project root directory.

2. Run the main training script:
   ```bash
   python train.py
   ```

3. To evaluate other machine learning models:
   ```bash
   python test_other_model.py
   ```

4. For train-validation-test split evaluation:
   ```bash
   python train_val_test.py
   ```

5. To recalculate performance indicators:
   ```bash
   python recalculate_indicator.py
   ```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Citation

If you use this code in your research, please cite our paper:

[Insert citation information here]

## Contact

For any questions or issues, please open an issue on GitHub or contact [Your Name] at [your.email@example.com].
