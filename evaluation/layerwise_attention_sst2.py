"""
Layer-wise attention preservation analysis for TinyBERT.

Computes average teacher-student attention similarity for each mapped layer pair:
  Student L1 -> Teacher L3
  Student L2 -> Teacher L6
  Student L3 -> Teacher L9
  Student L4 -> Teacher L12

Results are saved to:
  evaluation/outputs/layerwise_attention_sst2/layerwise_attention_results.json
"""

import os
import json
import torch
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEACHER_DIR = "checkpoints/sst2/teacher"
STUDENT_DIR = "checkpoints/sst2/student"
OUTPUT_DIR = "evaluation/outputs/layerwise_attention_sst2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        output_attentions=True
    )
    model.to(DEVICE)
    model.eval()
    return model


def get_attentions(model, tokenizer, sentence):
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

    return outputs.attentions


def average_attention(attentions, layer_index):
    attn = attentions[layer_index][0]   # [heads, tokens, tokens]
    attn = attn.mean(dim=0)             # [tokens, tokens]
    return attn.detach().cpu().numpy()


def main():
    print("Device:", DEVICE)

    dataset = load_dataset("nyu-mll/glue", "sst2")["validation"]

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR)
    teacher = load_model(TEACHER_DIR)
    student = load_model(STUDENT_DIR)

    layer_mapping = {
        0: 2,    # Student L1 -> Teacher L3
        1: 5,    # Student L2 -> Teacher L6
        2: 8,    # Student L3 -> Teacher L9
        3: 11,   # Student L4 -> Teacher L12
    }

    layer_scores = {
        "student_L1_teacher_L3": [],
        "student_L2_teacher_L6": [],
        "student_L3_teacher_L9": [],
        "student_L4_teacher_L12": [],
    }

    mapping_names = list(layer_scores.keys())

    for idx, item in enumerate(dataset):
        sentence = item["sentence"]

        teacher_attns = get_attentions(teacher, tokenizer, sentence)
        student_attns = get_attentions(student, tokenizer, sentence)

        for i, (student_layer, teacher_layer) in enumerate(layer_mapping.items()):
            teacher_matrix = average_attention(teacher_attns, teacher_layer)
            student_matrix = average_attention(student_attns, student_layer)

            mad = np.mean(np.abs(teacher_matrix - student_matrix))
            similarity = 1 - mad

            layer_scores[mapping_names[i]].append(float(similarity))

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(dataset)} examples...")

    results = {}

    print("\nLayer-wise attention similarity:")
    print(f"{'Student Layer':<15}{'Teacher Layer':<15}{'Mean Similarity':<18}{'Std':<10}")

    for key, scores in layer_scores.items():
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        results[key] = {
            "mean_similarity": mean_score,
            "std_similarity": std_score,
            "num_examples": len(scores)
        }

    table_rows = [
        ("Student L1", "Teacher L3", results["student_L1_teacher_L3"]),
        ("Student L2", "Teacher L6", results["student_L2_teacher_L6"]),
        ("Student L3", "Teacher L9", results["student_L3_teacher_L9"]),
        ("Student L4", "Teacher L12", results["student_L4_teacher_L12"]),
    ]

    for s_layer, t_layer, r in table_rows:
        print(
            f"{s_layer:<15}{t_layer:<15}"
            f"{r['mean_similarity']:<18.4f}{r['std_similarity']:<10.4f}"
        )

    out_path = os.path.join(OUTPUT_DIR, "layerwise_attention_results.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()