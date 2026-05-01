import os
import re

def parse_metrics_file(filepath):
    """
    Parses exact scientific metrics from a model's metrics.txt file.
    """
    metrics = {
        "model": "Unknown",
        "acc": "N/A",
        "params": "N/A",
        "size": "N/A",
        "latency": "N/A",
        "fps": "N/A"
    }
    
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, "r") as f:
        content = f.read()
        
    # Regex parsing
    model_match = re.search(r"Model:\s*(.+)", content)
    acc_match = re.search(r"Testing Accuracy:\s*([\d\.]+)%", content)
    params_match = re.search(r"Total Parameters:\s*([\d,]+)", content)
    size_match = re.search(r"Model Size on Disk:\s*([\d\.]+)\s*MB", content)
    latency_match = re.search(r"Latency:\s*([\d\.]+)\s*ms/image", content)
    fps_match = re.search(r"Throughput:\s*([\d\.]+)\s*FPS", content)

    if model_match:
        metrics["model"] = model_match.group(1).strip().capitalize()
    if acc_match:
        metrics["acc"] = f"{acc_match.group(1)}%"
    if params_match:
        # Convert total params to millions (M) for the table
        val = int(params_match.group(1).replace(",", ""))
        metrics["params"] = f"{val / 1_000_000:.2f}M"
    if size_match:
        metrics["size"] = f"{size_match.group(1)} MB"
    if latency_match:
        metrics["latency"] = f"{latency_match.group(1)} ms"
    if fps_match:
        metrics["fps"] = f"{fps_match.group(1)} FPS"
        
    return metrics


def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        return

    models = sorted(os.listdir(results_dir))
    all_metrics = []

    for model_dir in models:
        metrics_path = os.path.join(results_dir, model_dir, "metrics.txt")
        m = parse_metrics_file(metrics_path)
        if m:
            all_metrics.append(m)

    if not all_metrics:
        print("No metrics.txt files were found inside results/ folder.")
        return

    # ==========================================
    # 1. MARKDOWN TABLE
    # ==========================================
    md_table = "| Model Architecture | Test Accuracy | Parameters | Model Size | Latency | Inference Speed (FPS) |\n"
    md_table += "|---|---|---|---|---|---|\n"
    for m in all_metrics:
        md_table += f"| **{m['model']}** | {m['acc']} | {m['params']} | {m['size']} | {m['latency']} | {m['fps']} |\n"

    # ==========================================
    # 2. LATEX TABLE
    # ==========================================
    latex_table = "\\begin{table}[h]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Comparison of Deep Learning Architectures on Brain Tumor Classification}\n"
    latex_table += "\\label{tab:model_comparison}\n"
    latex_table += "\\begin{tabular}{lccccc}\n"
    latex_table += "\\hline\n"
    latex_table += "Model Architecture & Test Accuracy & Parameters & Model Size & Latency & Inference (FPS) \\\\\n"
    latex_table += "\\hline\n"
    for m in all_metrics:
        # Clean special chars if needed
        model_name = m['model'].replace("_", "\\_")
        acc = m['acc'].replace("%", "\\%")
        latex_table += f"{model_name} & {acc} & {m['params']} & {m['size']} & {m['latency']} & {m['fps']} \\\\\n"
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"

    # ==========================================
    # SAVE & PRINT RESULTS
    # ==========================================
    out_path = "results/master_comparison_table.txt"
    with open(out_path, "w") as f:
        f.write("=== SUMMARY COMPARISON (MARKDOWN) ===\n\n")
        f.write(md_table)
        f.write("\n\n=== SUMMARY COMPARISON (LATEX) ===\n\n")
        f.write(latex_table)

    print("\n" + "="*50)
    print("      MASTER BENCHMARK COMPARISON TABLE     ")
    print("="*50 + "\n")
    print(md_table)
    
    print("\n" + "="*50)
    print(f"✅ Success! Table generated & saved to: {out_path}")


if __name__ == "__main__":
    main()
