"""
Teacher-correct / student-wrong error analysis for TinyBERT.

This script identifies SST-2 validation examples where the BERT-base teacher
predicts the correct label, but the distilled TinyBERT student predicts the
wrong label.

The goal is to analyze what kinds of examples are more difficult for the
compressed student model after distillation.

Results are written to:
  - `evaluation/outputs/error_analysis_sst2/teacher_correct_student_wrong.json`
"""

import os
import json
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEACHER_DIR = "checkpoints/sst2/teacher"
STUDENT_DIR = "checkpoints/sst2/student"
OUTPUT_DIR = "evaluation/outputs/error_analysis_sst2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return model


def predict(model, tokenizer, sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()

    return pred, confidence


def main():
    print("Device:", DEVICE)

    dataset = load_dataset("nyu-mll/glue", "sst2")["validation"]

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR)
    teacher = load_model(TEACHER_DIR)
    student = load_model(STUDENT_DIR)

    error_examples = []

    for idx, item in enumerate(dataset):
        sentence = item["sentence"]
        label = item["label"]

        teacher_pred, teacher_conf = predict(teacher, tokenizer, sentence)
        student_pred, student_conf = predict(student, tokenizer, sentence)

        if teacher_pred == label and student_pred != label:
            error_examples.append({
                "index": idx,
                "sentence": sentence,
                "label": int(label),
                "teacher_pred": int(teacher_pred),
                "teacher_confidence": float(teacher_conf),
                "student_pred": int(student_pred),
                "student_confidence": float(student_conf),
                "sentence_length_words": len(sentence.split()),
                "contains_negation": any(
                    word in sentence.lower().split()
                    for word in ["not", "n't", "no", "never", "neither", "nor"]
                )
            })

    out_path = os.path.join(OUTPUT_DIR, "teacher_correct_student_wrong.json")

    with open(out_path, "w") as f:
        json.dump(error_examples, f, indent=2)

    print(f"Found {len(error_examples)} teacher-correct/student-wrong examples.")
    print("Saved:", out_path)

    print("\nFirst 10 examples:")
    for ex in error_examples[:10]:
        print("-" * 80)
        print("Sentence:", ex["sentence"])
        print("True label:", ex["label"])
        print("Teacher:", ex["teacher_pred"], "conf:", round(ex["teacher_confidence"], 3))
        print("Student:", ex["student_pred"], "conf:", round(ex["student_confidence"], 3))
        print("Length:", ex["sentence_length_words"], "Negation:", ex["contains_negation"])


if __name__ == "__main__":
    main()