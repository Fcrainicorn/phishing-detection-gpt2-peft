# phishing-detection-gpt2-peft
``
Fine-tuning GPT-2 with LoRA (PEFT) to detect phishing URLs.
``


## Project Context

This project was developed as the final project for my **Generative AI Fundamentals Nanodegree**. It demonstrates how a general-purpose foundation model (**GPT-2**) can be adapted using **Parameter-Efficient Fine-Tuning (PEFT)** to detect phishing URLs.

The model is trained on the **shawhin/phishing-site-classification** dataset and classifies URLs as:

* SAFE
* NOT_SAFE (phishing)

Using **LoRA**, only a small portion of the model’s parameters are trained instead of retraining the full model.

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT (LoRA)

## Run

This project was developed in a **GPU-accelerated environment**. To run it locally or in a cloud environment (e.g., Google Colab):

Install dependencies:

```
pip install -r requirements.txt
```

Run the training script:

```
python phishing_classifier.py
```
