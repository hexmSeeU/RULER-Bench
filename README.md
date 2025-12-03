<h1 align="center">RULER-Bench: Probing Rule-based Reasoning Abilities of Next-level Video Generation Models for Vision Foundation Intelligence</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2512.02622"><img src="https://img.shields.io/badge/Paper-arXiv%3A2512.02622-b31b1b?logo=arxiv&logoColor=red"></a>
  <a href="https://hexmseeu.github.io/RULER-Bench-proj/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-Website-8A2BE2"></a>
  <a href="https://huggingface.co/datasets/hexmSeeU/RULER-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue"></a>


## 📢 News

- **[2025-12-05]** We have released the **Paper**, **Project Page**, and **Dataset** !

---

## 📋 TODOs

- [x] Release paper
- [x] Release dataset 
- [ ] Release evaluation code



---

## 🧩Overview of RULER-Bench

We propose **RULER-Bench**, a comprehensive benchmark designed to evaluate the **rule-based reasoning abilities** of video generation models. Grounded in three fundamental domains, we formulate rule-based reasoning ability into six categories: <i>Science</i>, <i>Vision</i>, <i>Hypothesis Game</i>, <i>Semantics</i>, and <i>Humanity</i>. These categories are further subdivided into 40 tasks. Based on the task paradigm, we curate 622 high quality instances. Using these samples, we evaluate 10 video models based on the corresponding checklist across four evaluation metrics: <i>Rule Coherence</i>, <i>Visual Consistency</i>, <i>Instruction Following</i>, and <i>Visual Fidelity</i>. Each checklist question is scored by GPT-o3 with discrete labels. To validate the reliability of using GPT-os as an evaluator, we conduct a human alignment study, in which GPT-o3 achieves 85% agreement with human judgments. Extensive experiments show that the state-of-the-art model achieves only 48.87% on the rule coherence metric, highlighting significant room for improvement in the reasoning capability of next-level video models.

<p align="center">
  <img src="assets/teaser.png" alt="KRIS-Bench Overview" width="100%">
</p>


---
## 📊Evaluation Result

| Rule Categories | Metric | Veo3.1 | Veo2 | Sora2 | PixelVerse-V5 | Wan2.5 | Seedance1.0-pro | HunyuanVideo | CogVideoX1.5 5B | Wan2.2 A14B | Wan2.1 14B |
| :-------------: | :----: | :-----: | :---: | :----: | :-----------: | :-----: | :-------------: | :----------: | :----------: | :--------: | :--------: |
|  Science Rule   |   IF   |  65.05  | 42.17 | 66.00  |     57.13     |  57.38  |      58.86      |    24.92     |    27.97     |   37.15    |   35.80    |
|                 |   VC   |  83.18  | 73.3  | 88.01  |     80.76     |  80.48  |      80.99      |    48.46     |    48.84     |   68.52    |   65.74    |
|                 |   VF   |  91.37  | 82.33 | 89.49  |     89.74     |  85.35  |      87.69      |    71.29     |    70.96     |   80.37    |   81.93    |
|                 |   RC   |  50.97  | 22.16 | 47.09  |     41.41     |  33.64  |      31.96      |    12.64     |    13.70     |   17.16    |   15.90    |
|                 |  Avg   |  72.64  | 54.99 | 72.65  |     67.26     |  64.21  |      64.87      |    39.33     |    40.37     |   50.80    |   49.84    |
|    Game Rule    |   IF   |  39.75  | 24.25 | 39.19  |     30.10     |  26.59  |      24.26      |    14.75     |    22.75     |   16.29    |   19.30    |
|                 |   VC   |  51.45  | 36.33 | 72.33  |     67.09     |  72.71  |      68.79      |    40.07     |    55.29     |   64.56    |   37.52    |
|                 |   VF   |  77.95  | 59.15 | 88.18  |     80.59     |  86.28  |      88.39      |    59.13     |    69.45     |   80.13    |   72.20    |
|                 |   RC   |  17.70  | 8.17  | 19.97  |     13.06     |  15.45  |      15.61      |     6.98     |     7.56     |   14.12    |   10.48    |
|                 |  Avg   |  46.71  | 31.98 | 54.92  |     47.71     |  50.26  |      49.26      |    30.23     |    38.76     |   43.77    |   34.88    |
| Semantics Rule  |   IF   |  71.83  | 56.44 | 68.12  |     65.08     |  59.91  |      61.28      |    38.51     |    46.06     |   48.77    |   46.27    |
|                 |   VC   |  92.65  | 91.18 | 90.85  |     91.18     |  87.33  |      87.67      |    80.39     |    75.82     |   82.35    |   80.72    |
|                 |   VF   |  91.62  | 82.50 | 83.43  |     89.02     |  82.19  |      84.55      |    79.17     |    70.69     |   82.70    |   83.09    |
|                 |   RC   |  67.57  | 44.13 | 53.69  |     56.80     |  49.95  |      49.42      |    32.01     |    37.34     |   37.73    |   38.40    |
|                 |  Avg   |  80.92  | 68.56 | 74.02  |     75.52     |  69.84  |      70.73      |    57.52     |    57.48     |   62.89    |   62.12    |
| Hypothesis Rule |   IF   |  86.97  | 58.55 | 72.44  |     80.13     |  71.93  |      64.32      |    44.44     |    41.45     |   61.11    |   61.75    |
|                 |   VC   |  85.90  | 64.32 | 77.35  |     81.62     |  66.45  |      67.74      |    51.92     |    50.43     |   64.74    |   55.56    |
|                 |   VF   |  92.20  | 81.54 | 82.50  |     85.73     |  76.86  |      79.66      |    73.89     |     63.8     |   77.03    |   75.17    |
|                 |   RC   |  46.79  | 12.50 | 41.35  |     46.69     |  18.31  |      28.31      |     9.62     |    11.00     |   12.93    |   17.84    |
|                 |  Avg   |  77.96  | 54.23 | 68.41  |     73.54     |  58.39  |      60.01      |    44.97     |    41.67     |   53.95    |   52.58    |
|  Humanity Rule  |   IF   |  79.90  | 53.46 | 80.04  |     72.87     |  63.28  |      68.93      |    46.56     |    42.32     |   49.76    |   52.28    |
|                 |   VC   |  87.37  | 73.10 | 88.06  |     84.25     |  79.83  |      83.13      |     70.6     |    54.23     |   72.34    |   70.47    |
|                 |   VF   |  94.49  | 84.38 | 88.08  |     89.65     |  83.90  |      88.52      |    80.94     |    67.76     |   83.15    |   82.32    |
|                 |   RC   |  61.23  | 35.23 | 56.78  |     50.63     |  33.41  |      38.75      |    27.78     |    20.60     |   30.21    |   29.21    |
|                 |  Avg   |  80.75  | 61.54 | 78.24  |     74.35     |  65.10  |      69.83      |    56.47     |    46.23     |   58.86    |   58.57    |
|   Vision Rule   |   VC   |  59.53  | 46.19 | 57.77  |     56.14     |  70.04  |      61.86      |    43.49     |    24.79     |   59.03    |   51.26    |
|                 |   VF   |  72.67  | 57.63 | 57.77  |     71.61     |  68.32  |      76.06      |    52.94     |    29.41     |   65.55    |   49.58    |
|                 |   RC   |  48.94  | 30.58 | 28.50  |     40.47     |  42.24  |      41.74      |    18.91     |    14.78     |   29.34    |   23.25    |
|                 |  Avg   |  60.38  | 44.80 | 48.02  |     56.07     |  60.20  |      59.89      |    38.45     |    22.99     |   51.31    |   41.36    |
|     Average     |   IF   |  68.7   | 46.97 | 65.16  |     61.06     |  55.82  |      55.53      |    33.84     |    36.11     |   42.62    |   43.08    |
|                 |   VC   |  76.68  | 64.07 | 79.06  |     76.84     |  76.14  |      75.03      |    55.82     |    51.57     |   68.59    |   60.21    |
|                 |   VF   |  86.72  | 74.59 | 81.58  |     84.39     |  80.48  |      84.14      |    69.56     |    62.01     |   78.15    |   74.05    |
|                 |   RC   |  48.87  | 25.46 | 41.23  |     41.51     |  32.17  |      34.3       |    17.99     |     17.5     |   23.58    |   22.51    |
|                 |  Avg   |  70.24  | 52.77 | 66.76  |     65.95     |  61.15  |      62.25      |     44.3     |     41.8     |   53.24    |   49.96    |
|    Win Rate     |        |  0.397  | 0.186 | 0.340  |     0.300     |  0.257  |      0.267      |    0.151     |    0.151     |   0.193    |   0.162    |

---

## ✍️Citation

If you find RULER-Bench helpful, please cite:

```bibtex
@article{he2025ruler,
  title={RULER-Bench: Probing Rule-based Reasoning Abilities of Next-level Video Generation Models for Vision Foundation Intelligence},
  author={He, Xuming and Fan, Zehao and Li, Hengjia and Zhuo, Fan and Xu, Hankun and Cheng, Senlin and Weng, Di and Liu, Haifeng and Ye, Can and Wu, Boxi},
  journal={arXiv preprint arXiv:2512.02622},
  year={2025}
}
```

------

## 📬 Contact

For questions or submissions, please open an issue or email **[hexuming773@gmail.com](mailto:hexuming773@gmail.com)**.