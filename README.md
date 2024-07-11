
# Predicting Human Choice in Language-Based Persuasion Games

## Authors
- **Samar Samara**
  - Technion - Israel Institute of Technology
  - Email: [samar.s@campus.technion.ac.il](mailto:samar.s@campus.technion.ac.il)
- **Rawan Badarneh**
  - Technion - Israel Institute of Technology
  - Email: [rawanb@campus.technion.ac.il](mailto:rawanb@campus.technion.ac.il)

## Abstract
Predicting human decisions is crucial for enhancing recommendation systems and boosting user engagement. Traditional models like Transformers and LSTMs have limitations: Transformers struggle with sequential data, while LSTMs may overlook broader contextual information. To address these challenges, we propose a hybrid architecture combining the strengths of both models. Our approach achieves 84.46% accuracy, significantly improving predictive modeling in complex decision-making scenarios.

## Introduction
The ability to predict human decisions in real-life scenarios is valuable for various applications, including recommendation systems. Existing models such as Transformers and LSTMs have specific limitations. We propose a hybrid model that leverages the sequential processing strengths of LSTMs and the contextual understanding of Transformers.

## Related Works
Our work builds on the advancements in neural network architectures for time series prediction, particularly the integration of LSTM and Transformer models. Previous research has shown the potential of hybrid models to enhance predictive performance by leveraging the best features of both architectures.

## Model Architectures
We introduce three neural network architectures:
1. **LSTM-Transformer Model**: An LSTM processes the sequential data, and its output is passed to a Transformer model.
2. **Transformer-LSTM Model**: A Transformer processes the input data first, followed by an LSTM layer.
3. **Stacked Transformer-LSTM Model**: This model alternates between Transformer and LSTM layers to capture both global and local dependencies effectively.

## Experiments and Results
We validated our models using a dataset from (Shapira et al., 2024) involving human decision-making in language-based persuasion games. Our Transformer-LSTM model achieved the highest accuracy of 84.46%.

### Dataset
We utilized a dataset comprising interactions between human decision-makers and rule-based experts in a language-based persuasion game, including 87,204 decisions made by 245 players.

### Hyper-Parameter Tuning
We trained each model with various learning rates, determining the optimal rates for each model to achieve the highest accuracy on the validation set.

### Model Selection
The Transformer-LSTM model outperformed others, achieving the highest accuracy of 84.46%.

### Results
After extensive testing, the Transformer-LSTM model demonstrated high reliability, with a 95% confidence interval for accuracy.

## Discussion
Our results indicate that the Transformer-LSTM model is highly effective in predicting human decisions, outperforming other models and demonstrating consistent performance across different tests.

## Installation
Before running this project, ensure you have the following software installed:
- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib

You can install the necessary libraries using pip:
```sh
pip install jupyter pandas numpy matplotlib
```

## How to Run This Project
1. Clone the repository:
    ```sh
    git clone <https://github.com/samarsamara/NLPProject>
    cd <NLPProjec>
    ```
2. Launch Jupyter Notebook:
    ```sh
    jupyter notebook
    ```
3. Open and run the relevant notebooks sequentially.

