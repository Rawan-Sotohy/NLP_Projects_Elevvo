"""
Training Script for NER Model
Usage: python train.py --model bert-base-cased --epochs 3 --batch_size 16
"""

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np


# Label list for CoNLL-2003
LABEL_LIST = [
    "O",       # Outside any entity
    "B-PER",   # Beginning of person name
    "I-PER",   # Inside person name
    "B-ORG",   # Beginning of organization
    "I-ORG",   # Inside organization
    "B-LOC",   # Beginning of location
    "I-LOC",   # Inside location
    "B-MISC",  # Beginning of miscellaneous
    "I-MISC"   # Inside miscellaneous
]

label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for i, label in enumerate(LABEL_LIST)}


def load_conll_data():
    """Load CoNLL-2003 dataset"""
    print("📚 Loading CoNLL-2003 dataset...")
    
    # Load from HuggingFace datasets
    dataset = load_dataset("conll2003")
    
    print(f"✅ Loaded!")
    print(f"   - Train: {len(dataset['train'])} samples")
    print(f"   - Validation: {len(dataset['validation'])} samples")
    print(f"   - Test: {len(dataset['test'])} samples")
    
    return dataset


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize text and align labels with subword tokens
    Important: BERT uses WordPiece, so one word may become multiple tokens
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            # Special tokens have word_id = None
            if word_idx is None:
                label_ids.append(-100)  # Ignore in loss calculation
            # First subword of a word gets the label
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # Other subwords get -100 (ignored)
            else:
                label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(pred):
    """Compute NER metrics using seqeval"""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def main(args):
    """Main training function"""
    
    print("🚀 Starting NER Training Pipeline...\n")
    
    # 1. Load dataset
    dataset = load_conll_data()
    
    # 2. Load tokenizer and model
    print(f"\n🤖 Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )
    
    # 3. Tokenize dataset
    print("\n🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )
    
    # 4. Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 7. Train!
    print("\n🏋️ Training model...\n")
    trainer.train()
    
    # 8. Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print("\n✅ Test Results:")
    for key, value in test_results.items():
        print(f"   {key}: {value:.4f}")
    
    # 9. Save model
    print(f"\n💾 Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 10. Generate classification report
    print("\n📈 Detailed Classification Report:")
    predictions = trainer.predict(tokenized_dataset["test"])
    predictions_np = np.argmax(predictions.predictions, axis=2)
    
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_np, predictions.label_ids)
    ]
    
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_np, predictions.label_ids)
    ]
    
    print(classification_report(true_labels, true_predictions))
    
    print("\n🎉 Training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model")
    
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-cased",
        help="Pretrained model name (e.g., bert-base-cased, roberta-base)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/bert-ner",
        help="Output directory for saved model"
    )
    
    args = parser.parse_args()
    
    # Show configuration
    print("⚙️  Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Output Dir: {args.output_dir}\n")
    
    main(args)