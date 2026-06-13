"""
Attention similarity vs prediction confidence analysis for TinyBERT.

This extension checks whether examples with higher teacher-student attention
similarity also lead to higher student prediction confidence on SST-2.

Results are written to:
  - evaluation/outputs/attention_confidence_sst2/attention_confidence_results.json
  - evaluation/outputs/attention_confidence_sst2/attention_similarity_vs_confidence.png
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEACHER_DIR = "checkpoints/sst2/teacher"
STUDENT_DIR = "checkpoints/sst2/student"
OUTPUT_DIR = "evaluation/outputs/attention_confidence_sst2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        output_attentions=True
    )
    model.to(DEVICE)
    model.eval()
    return model


def predict_with_attention(model, tokenizer, sentence):
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

    return pred, confidence, outputs.attentions


def average_attention(attentions, layer_index):
    attn = attentions[layer_index][0]
    attn = attn.mean(dim=0)
    return attn.detach().cpu().numpy()


def main():
    print("Device:", DEVICE)

    dataset = load_dataset("nyu-mll/glue", "sst2")["validation"]

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR)
    teacher = load_model(TEACHER_DIR)
    student = load_model(STUDENT_DIR)

    layer_mapping = {
        0: 2,
        1: 5,
        2: 8,
        3: 11,
    }

    results = []

    for idx, item in enumerate(dataset):
        sentence = item["sentence"]
        label = item["label"]

        teacher_pred, teacher_conf, teacher_attns = predict_with_attention(
            teacher, tokenizer, sentence
        )

        student_pred, student_conf, student_attns = predict_with_attention(
            student, tokenizer, sentence
        )

        layer_similarities = []

        for student_layer, teacher_layer in layer_mapping.items():
            teacher_matrix = average_attention(teacher_attns, teacher_layer)
            student_matrix = average_attention(student_attns, student_layer)

            mad = np.mean(np.abs(teacher_matrix - student_matrix))
            similarity = 1 - mad

            layer_similarities.append(similarity)

        avg_attention_similarity = float(np.mean(layer_similarities))

        results.append({
            "index": idx,
            "sentence": sentence,
            "label": int(label),
            "teacher_pred": int(teacher_pred),
            "student_pred": int(student_pred),
            "teacher_confidence": float(teacher_conf),
            "student_confidence": float(student_conf),
            "student_correct": bool(student_pred == label),
            "avg_attention_similarity": avg_attention_similarity
        })

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(dataset)} examples...")

    out_json = os.path.join(OUTPUT_DIR, "attention_confidence_results.json")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    similarities = [r["avg_attention_similarity"] for r in results]
    confidences = [r["student_confidence"] for r in results]

    correlation = np.corrcoef(similarities, confidences)[0, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(similarities, confidences, alpha=0.7)
    plt.xlabel("Average teacher-student attention similarity")
    plt.ylabel("Student prediction confidence")
    plt.title(f"Attention Similarity vs Student Confidence\nPearson r = {correlation:.3f}")
    plt.tight_layout()

    out_plot = os.path.join(OUTPUT_DIR, "attention_similarity_vs_confidence.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()

    print("\nSaved JSON:", out_json)
    print("Saved plot:", out_plot)
    print("Pearson correlation:", round(correlation, 4))


if __name__ == "__main__":
    main()