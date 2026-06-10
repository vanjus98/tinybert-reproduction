"""
Attention preservation analysis for the TinyBERT reproduction extension.

This script evaluates whether the distilled TinyBERT student preserves attention
patterns from its BERT-base teacher on the SST-2 validation set.

It loads:
  - Teacher model from `checkpoints/sst2/teacher`
  - Student model from `checkpoints/sst2/student`

The analysis:
  1. Loads both models with `output_attentions=True`
  2. Selects SST-2 validation examples where both teacher and student are correct
  3. Extracts attention matrices from corresponding layers:
       Student L1 -> Teacher L3
       Student L2 -> Teacher L6
       Student L3 -> Teacher L9
       Student L4 -> Teacher L12
  4. Averages attention across heads
  5. Computes mean absolute difference and attention similarity
  6. Saves teacher, student, and difference heatmaps for each selected example

Results are written to:
  - `evaluation/attention_results_sst2/attention_similarity_results.json`
  - `evaluation/attention_results_sst2/*.png`
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
OUTPUT_DIR = "evaluation/attention_results_sst2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        output_attentions=True
    )
    model.to(DEVICE)
    model.eval()
    return model


def get_prediction_and_attention(model, tokenizer, sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=-1).item()

    # attentions: list of layers
    # each: [batch, heads, tokens, tokens]
    attentions = outputs.attentions

    return pred, attentions, tokens


def average_attention(attentions, layer_index):
    """
    Takes one layer and averages attention over all heads.
    Returns matrix [tokens, tokens].
    """
    attn = attentions[layer_index][0]  # [heads, tokens, tokens]
    attn = attn.mean(dim=0)            # [tokens, tokens]
    return attn.detach().cpu().numpy()


def save_heatmap(matrix, tokens, title, path):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    print("Device:", DEVICE)

    dataset = load_dataset("nyu-mll/glue", "sst2")["validation"]

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR)

    teacher = load_model(TEACHER_DIR)
    student = load_model(STUDENT_DIR)

    # Paper mapping: student 4 layers maps to teacher layers 3, 6, 9, 12
    # Python indexes: student layers 0,1,2,3 and teacher layers 2,5,8,11
    layer_mapping = {
        0: 2,
        1: 5,
        2: 8,
        3: 11,
    }

    selected_examples = []
    results = []

    # Pick first few examples where both models are correct
    for item in dataset:
        sentence = item["sentence"]
        label = item["label"]

        teacher_pred, teacher_attns, tokens = get_prediction_and_attention(
            teacher, tokenizer, sentence
        )
        student_pred, student_attns, _ = get_prediction_and_attention(
            student, tokenizer, sentence
        )

        if teacher_pred == label and student_pred == label:
            selected_examples.append((sentence, label, teacher_attns, student_attns, tokens))

        if len(selected_examples) >= 5:
            break

    print(f"Selected {len(selected_examples)} examples.")

    for ex_id, (sentence, label, teacher_attns, student_attns, tokens) in enumerate(selected_examples):
        print(f"\nExample {ex_id}")
        print("Sentence:", sentence)
        print("Label:", label)

        example_result = {
            "example_id": ex_id,
            "sentence": sentence,
            "label": int(label),
            "layers": []
        }

        for student_layer, teacher_layer in layer_mapping.items():
            teacher_matrix = average_attention(teacher_attns, teacher_layer)
            student_matrix = average_attention(student_attns, student_layer)

            # Mean absolute difference
            mad = np.mean(np.abs(teacher_matrix - student_matrix))

            # Similarity score: simple interpretation, higher = more similar
            similarity = 1 - mad

            example_result["layers"].append({
                "student_layer": student_layer + 1,
                "teacher_layer": teacher_layer + 1,
                "mean_absolute_difference": float(mad),
                "attention_similarity": float(similarity)
            })

            base_name = f"example_{ex_id}_studentL{student_layer+1}_teacherL{teacher_layer+1}"

            save_heatmap(
                teacher_matrix,
                tokens,
                f"Teacher BERT Attention - Layer {teacher_layer + 1}",
                os.path.join(OUTPUT_DIR, base_name + "_teacher.png")
            )

            save_heatmap(
                student_matrix,
                tokens,
                f"TinyBERT Attention - Layer {student_layer + 1}",
                os.path.join(OUTPUT_DIR, base_name + "_student.png")
            )

            diff_matrix = np.abs(teacher_matrix - student_matrix)

            save_heatmap(
                diff_matrix,
                tokens,
                f"Attention Difference | Teacher L{teacher_layer + 1} vs Student L{student_layer + 1}",
                os.path.join(OUTPUT_DIR, base_name + "_difference.png")
            )

            print(
                f"Student layer {student_layer + 1} vs Teacher layer {teacher_layer + 1}: "
                f"MAD={mad:.4f}, similarity={similarity:.4f}"
            )

        results.append(example_result)

    output_json = os.path.join(OUTPUT_DIR, "attention_similarity_results.json")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to:", output_json)
    print("Saved heatmaps to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()