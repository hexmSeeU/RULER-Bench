<h1 align="center">KRIS-Bench: Benchmarking Next-Level Intelligent Image Editing Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/250xxxx"><img src="https://img.shields.io/badge/Paper-arXiv%3A250xxxx-b31b1b?logo=arxiv&logoColor=red"></a>
  <a href="https://hexmseeu.github.io/RULER-Bench-proj/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-Website-8A2BE2"></a>
  <a href="https://huggingface.co/datasets/hexmSeeU/RULER-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue"></a>


## 📢 News

- **[2025-12-05]** 🚀 The **Paper**, **Project Page**, and **Dataset** are released !

---

## 🧩Overview of RULER-Bench

We propose RULER-Bench, a comprehensive benchmark designed to evaluate the rule-based reasoning abilities of video generation models.<b><i>Left:</i></b> Grounded in three fundamental domains, we formulate rule-based reasoning ability into six categories: <i>Science</i>, <i>Vision</i>, <i>Hypothesis Game</i>, <i>Semantics</i>, and <i>Humanity</i>.These categories are further subdivided into 40 tasks.<b><i>Center:</i></b> Using the collected samples, we evaluate 10 video models based on the corresponding checklist across four metrics. Each checklist question is scored by GPT-o3 with discrete labels.To validate the reliability of the evaluator, we conduct a human alignment study, in which GPT-o3 achieves 85\% agreement with human judgments. <b><i>Right:</i></b> Extensive experiments demonstrate that Veo3.1 achieves the best performance. However, all models exhibit limited reasoning ability across different rule categories.

<p align="center">
  <img src="assets/teaser.png" alt="KRIS-Bench Overview" width="85%">
</p>


---

## Requirements

- Python 3.8+

Set your **OpenAI API Key** as an environment variable before running:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

## 🚀 Usage

First, download the benchmark and place them in ``./KRIS_Bench`` directory. You can fetch the full dataset from [Hugging Face dataset KRIS-Bench](https://huggingface.co/datasets/Liang0223/KRIS_Bench). For convenience, we also keep the benchmark in this repository.

Evaluate a list of models across all categories using the main script:

```bash
python metrics_common.py --models doubao gpt gemini
```

**Arguments**

- `--models`: Space-separated model names to evaluate.

The script iterates over models × categories, calls GPT-4o (when applicable) for automated judging, and writes:

```
results/{model}/{category}/metrics.json
```

### Per-task Scripts

If you want to run specific task families, use:

```bash
# Viewpoint change with ground-truth image
python metrics_view_change.py --models your_model

# Knowledge plausibility tasks
python metrics_knowldge.py --models your_model

# Multi-element composition
python metrics_multi_element.py --models your_model

# Temporal prediction
python metrics_temporal_prediction.py --models your_model
```

> **Note:** Ensure model-generated images exist under:
>
> ```
> results/{model}/{category}/
> ```
>
> and are named as {image_id}, which corresponds to the index of the input sample.
>

------

## 📤 Output Format

Each category produces a `metrics.json` like:

```json
{
  "1": {
    "instruction": "...",
    "explain": "...",
    "consistency_score": 5,
    "consistency_reasoning": "...",
    "instruction_score": 5,
    "instruction_reasoning": "...",
    "quality_score": 4,
    "quality_reasoning": "..."
  },
  "2":{
      ...
  },
}
```

------

## 📮Notes

- Ensure `KRIS_Bench/{category}/annotation.json` and original images are present.
- Check your generated images are correctly named and placed in `results/{model}/{category}/`.
- OpenAI API usage is subject to rate limits and costs. Adjust `max_workers` and batch size as needed.

------

## 📜Related Repositories

[ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel/tree/main/eval/gen/kris): The Bagel team has evaluated their models on KRIS_Bench and released the evaluation code.

## ✍️Citation

If you find KRIS-Bench helpful, please cite:

```bibtex
@article{wu2025kris,
  title={KRIS-Bench: Benchmarking Next-Level Intelligent Image Editing Models},
  author={Wu, Yongliang and Li, Zonghui and Hu, Xinting and Ye, Xinyu and Zeng, Xianfang and Yu, Gang and Zhu, Wenbo and Schiele, Bernt and Yang, Ming-Hsuan and Yang, Xu},
  journal={arXiv preprint arXiv:2505.16707},
  year={2025}
}
```

------

## 📬 Contact

For questions or submissions, please open an issue or email **[yongliang0223@gmail.com](mailto:yongliang0223@gmail.com)**.