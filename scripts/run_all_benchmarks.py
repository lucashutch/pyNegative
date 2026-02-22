#!/usr/bin/env python3
"""
Master benchmark runner that executes multiple configurations and generates a summary report.
"""

import json
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
        return None
    return result.stdout


def generate_markdown_summary(results_cpu, results_gpu, output_path):
    lines = [
        "# Pipeline Optimization Baseline Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "This report establishes the performance baseline for the current image processing pipeline.",
        "",
        "| Category | Operation | Resolution | CPU (ms) | GPU/OpenCL (ms) | Speedup |",
        "|:---|:---|:---|:---|:---|:---|",
    ]

    # Map results by (operation, size)
    gpu_map = {(r["operation"], r["size"]): r for r in results_gpu["results"]}

    for r_cpu in results_cpu["results"]:
        op = r_cpu["operation"]
        size = r_cpu["size"]
        cpu_time = r_cpu["mean_ms"]

        gpu_time = "N/A"
        speedup = "N/A"

        if (op, size) in gpu_map:
            gt = gpu_map[(op, size)]["mean_ms"]
            gpu_time = f"{gt:.2f}"
            speedup = f"{cpu_time / gt:.2f}x"

        lines.append(
            f"| Pipeline | {op} | {size} | {cpu_time:.2f} | {gpu_time} | {speedup} |"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Summary report generated at {output_path}")


def main():
    image_path = "IMG_0051.CR3"
    results_path = "benchmark_results.json"

    if not Path(results_path).exists():
        if not Path(image_path).exists():
            # Look for any CR3/ARW/DNG in current dir
            found = (
                list(Path(".").glob("*.CR3"))
                + list(Path(".").glob("*.ARW"))
                + list(Path(".").glob("*.DNG"))
            )
            if found:
                image_path = str(found[0])
            else:
                print(
                    "Error: No RAW test image found and no benchmark_results.json exists."
                )
                sys.exit(1)

        print(f"Using test image: {image_path}")
        print("\n>>> Running benchmark...")
        run_cmd(
            [
                "uv",
                "run",
                "python",
                "scripts/benchmark_full_pipeline.py",
                "--image",
                image_path,
                "--output",
                results_path,
                "--iterations",
                "3",
                "--warmup",
                "1",
            ]
        )
    else:
        print(f"Using existing results from {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    # Generate a simple markdown report from the single run for now
    lines = [
        "# Pipeline Optimization Baseline Report",
        "",
        f"**Date:** {results['timestamp']}",
        f"**OpenCL Available:** {results['opencl_available']}",
        "",
        "## Results Table",
        "",
        "| Operation | Resolution | Mean (ms) | MP/s | Std (ms) |",
        "|:---|:---|:---|:---|:---|",
    ]

    for r in results["results"]:
        lines.append(
            f"| {r['operation']} | {r['size']} | {r['mean_ms']:.2f} | {r.get('throughput_mpps', 0):.2f} | {r.get('std_ms', 0):.2f} |"
        )

    report_path = "BASELINE_PERFORMANCE.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nFinal baseline report generated at {report_path}")


if __name__ == "__main__":
    main()
