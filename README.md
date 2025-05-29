<h1 align="center">🦅 FALCON: An ML Framework for Fully Automated Layout-Constrained Analog Circuit Design</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/></a>
  <a href="https://github.com/AsalMehradfar/FALCON/stargazers"><img src="https://img.shields.io/github/stars/AsalMehradfar/FALCON?style=social" alt="GitHub Stars"/></a>
</p>

<p align="justify">
<strong>FALCON</strong> (Fully Automated Layout-Constrained analOg circuit desigN) is a unified machine learning framework for end-to-end analog and RF circuit synthesis. Given a target performance specification, FALCON selects a suitable circuit topology and infers layout-constrained parameters through a modular, gradient-based optimization pipeline powered by graph neural networks (GNNs).
</p>

Repository Includes:
- **Dataset** of 1 million Cadence-simulated analog/mm-wave circuits across 20 topologies and 5 circuit families.
- **ML Models** for performance-driven topology selection (MLP), forward performance prediction (GNN), and parameter inference.
- **Graph Representation** pipeline converting Cadence netlists into rich, edge-centric graph structures.
- **Layout-Aware Loss** integrated into inverse design via analytical models capturing parasitic and area constraints.

<p align="justify">
FALCON achieves over <strong>99% topology selection accuracy</strong> and <strong>sub-10% relative error</strong> in performance prediction, with <strong>layout-constrained design generation in under 1 second</strong> per instance. Its generalizability to unseen topologies and tight integration of schematic and layout objectives make it a practical foundation for fully automated, real-world analog circuit design.
</p>

## 📖 Table of Contents

  * [Environment Setup](#%EF%B8%8F-environment-setup)
  * [Usage](#-usage)
  * [Citation](#-citation)
  * [Where to Ask for Help](#-where-to-ask-for-help)

## ⚙️ Environment Setup

We recommend using [Conda](https://docs.conda.io/en/latest/) to manage dependencies for FALCON.

#### 📦 Install via `falcon.yml`

Clone the repository and create the environment:

```bash
# Clone the repository
git clone https://github.com/AsalMehradfar/FALCON.git
cd FALCON

# Create the environment from the YAML file
conda env create -f falcon.yml

# Activate the environment
conda activate falcon
```

#### 🔄 Optional: Update Environment

If you make changes to the YAML or add packages later:

```bash
conda env update -f falcon.yml --prune
```

## 🚀 Usage

FALCON consists of a three-stage pipeline for analog circuit design:

- **Stage 1**: Topology Selection
- **Stage 2**: Performance Prediction
- **Stage 3**: Layout-Aware Parameter Inference

Each stage can be run independently once data is prepared.

---

#### 📂 Step 1: Preprocessing (only needed if raw data is provided)

If `graphs.json` is **not** available, first run:

```bash
python scripts/process_netlists.py
```

If index splits are **not** available, run:

```bash
python scripts/generate_splits.py
```
#### 🧪 Step 2: Generate Model Inputs

To generate the MLP and GNN inputs and save the required scalers:

```bash
python scripts/save_mlp_data.py
python scripts/save_gnn_data.py
```

#### 🏋️ Step 3: Train Models

To train the models for each stage of the FALCON pipeline, run:

```bash
# Stage 1: Train the MLP for topology classification
python scripts/train_mlp.py

# Stage 2: Train the GNN for performance prediction
python scripts/train_gnn.py

# (Optional) Fine-tune the GNN on unseen topologies
python scripts/finetune_gnn.py
```

#### 📊 Step 4: Evaluate Each Stage

To evaluate each stage of the FALCON pipeline, run:

```bash
# Stage 1 Evaluation: Topology selection accuracy
python evaluation/mlp_eval.py

# Stage 2 Evaluation: Forward performance prediction accuracy
python evaluation/gnn_forward_eval.py

# Stage 3 Evaluation: Inverse parameter inference with layout-aware optimization
python evaluation/gnn_backward_eval.py
```
> All scripts assume data is organized under `dataset/` directory as expected.

## 🎯 Citation 

If you use FALCON in a research paper, please cite our [paper](https://arxiv.org/abs/2505.21923):

```bibtex
@article{Mehradfar2025FALCON,
      title={FALCON: An ML Framework for Fully Automated Layout-Constrained Analog Circuit Design}, 
      author={Asal Mehradfar and Xuzhe Zhao and Yilun Huang and Emir Ceyani and Yankai Yang and Shihao Han and Hamidreza Aghasi and Salman Avestimehr},
      year={2025},
      url={https://arxiv.org/abs/2505.21923}
}
```

## ❓ Where to Ask for Help

<p align="justify" > 
If you have any questions, feel free to open a <a href="https://github.com/AsalMehradfar/FALCON/discussions">Discussion</a> and ask your question. You can also email <a href="mailto:mehradfa@usc.edu">mehradfa@usc.edu</a> (Asal Mehradfar).
</p>
