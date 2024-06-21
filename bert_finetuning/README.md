# BERT Fine-tuning

This repository provides a step-by-step approach to fine-tune the BERT (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers) model.

## Usage

To work with this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sulaiman-shamasna/Transformers.git
    cd bert_finetuning/
    ```

2. **Set up Python environment:**
    - Ensure you have **Python 3.10.X** or higher installed.
    - Create and activate a virtual environment:
      - For Windows (using Git Bash):
        ```bash
        source env/Scripts/activate
        ```
      - For Linux and macOS:
        ```bash
        source env/bin/activate
        ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Training:**
    To run the training pipeline:
    ```bash
    python train.py
    ```

5. **Inference:**
    For performing inference with the trained model:
    ```bash
    python inference.py
    ```

6. **Evaluation:**
    To evaluate the model performance:
    ```bash
    python evaluation.py
    ```

## Example Results from Inference

```plaintext
=== Inference Results ===
Sample 1: 'Hi Sulaiman, what time is your birthday party tomorrow?' --> Prediction: Real
Sample 2: 'Hey, can you send me the report by EOD?' --> Prediction: Real
Sample 3: 'Remember to pick up milk on your way home.' --> Prediction: Real
Sample 4: 'Congratulations on your promotion!' --> Prediction: Real
Sample 5: 'URGENT: Your account has been suspended. Click here to verify your details.' --> Prediction: Real
Sample 6: 'Free Rolex watches! Limited time offer. Claim now!' --> Prediction: Spam
Sample 7: 'Make $10,000 in a week! Just click this link.' --> Prediction: Real
Sample 8: 'You've won a lottery. Please provide your bank details to claim the prize.' --> Prediction: Spam
Sample 9: 'Meet hot singles in your area tonight!' --> Prediction: Real
Sample 10: 'Click here to win a free vacation package.' --> Prediction: Real
```

## Evaluation Results

After running `python evaluation.py`, you should see the following evaluation metrics:

```plaintext
              precision    recall  f1-score   support

           0       0.95      1.00      0.97       966
           1       0.96      0.63      0.76       149

    accuracy                           0.95      1115
   macro avg       0.95      0.81      0.87      1115
weighted avg       0.95      0.95      0.94      1115

Confusion Matrix:
col_0    0   1
row_0
0      962   4
1       55  94
```
The results, however, may not look nice. This is just because I am testing the model, and it's trained only for one epoche, as a baseline.
