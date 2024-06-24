# T5 Fine-tuning

This repository provides a step-by-step approach to fine-tune the T5 (**T**ext-**T**o-**T**ext **T**ransfer **T**ransformer) model.

## Usage

To work with this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sulaiman-shamasna/Transformers.git
    cd T5/
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
4. **Datasets**
    To be able to train, you must get the dataset, this is done by running the following command in the bash: 

      - wget:
        ```bash
        wget "https://www.dropbox.com/scl/fi/525gv6tmdi3n32mipo6mr/input.zip?rlkey=5jdsxahphk2ped5wxbxnv0n4y&dl=1" -O input.zip
        ```
  
    OR 

      - curl:
        ```bash
        curl "https://www.dropbox.com/scl/fi/525gv6tmdi3n32mipo6mr/input.zip?rlkey=5jdsxahphk2ped5wxbxnv0n4y&dl=1" -O input.zip
        ```
    
    Then, unzip it:

    ```bash
    unzip input.zip
    ```

5. **Training:**
    To run the training pipeline:
    ```bash
    python main.py
    ```

