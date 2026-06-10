# ⚡ Smart Energy Grid Optimizer

**Deep Reinforcement Learning + Constraint Programming for energy dispatch under uncertainty.**

Final exam prototype — *Reasoning and Decision Making Under Uncertainty* (MAIB), **Topic 5**.

A smart-city grid must balance renewable (solar, wind) and fossil (coal, gas)
generation to meet uncertain, fluctuating demand. A **Deep Q-Network** learns a
dispatch policy, and a **Constraint Programming** optimiser provides a
feasibility-guaranteed benchmark. An interactive dashboard exposes the
trade-offs between **sustainability, reliability and cost**.

## 🔑 Four key concepts (named for the rubric)
1. **Markov Decision Process** — `SmartGridEnv`
2. **Deep Q-Network (Deep Reinforcement Learning)** — `DQNAgent`, `train_dqn`
3. **Constraint Programming / Optimisation** — `evaluate_constraint_day` (OR-Tools LP)
4. **Sensitivity Analysis under Uncertainty** — `sensitivity_emission_cap` + Monte-Carlo days

## 🖥️ Dashboard preview
| Live Dispatch | AI vs Optimiser |
|:---:|:---:|
| ![Live dispatch](screenshots/01_live_dispatch.png) | ![AI vs optimiser](screenshots/02_ai_vs_optimiser.png) |
| **Sensitivity** | **How it works** |
| ![Sensitivity](screenshots/03_sensitivity.png) | ![How it works](screenshots/04_how_it_works.png) |

## 📁 Files
| File | Purpose |
|------|---------|
| `model.py` | The engine: environment (MDP), DQN agent, constraint solver, metrics |
| `app.py` | Streamlit dashboard (the live demo / "Dashboard" deliverable) |
| `train_colab.ipynb` | Self-contained Google Colab notebook (graded code artifact) |
| `dqn_weights.pth` · `train_history.json` | Pre-trained policy so the app starts instantly |
| `train_weights.py` | Regenerates the weights above |
| `requirements.txt` | Pinned dependencies for a reliable cloud build |
| `screenshots/` | Dashboard images (for the slide deck) |

---

## 🚀 Deploy the dashboard (browser only — no local install)

1. **Create a GitHub repo** (e.g. `smart-grid-optimizer`), set it **Public**.
2. **Upload** `app.py`, `model.py`, `requirements.txt` (and `README.md`) to the repo.
3. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub.
4. **New app** → pick your repo, branch `main`, main file `app.py`.
5. *(Recommended)* **Advanced settings → Python version → 3.11**.
6. **Deploy.** First build takes a few minutes (it installs PyTorch). The DQN
   trains once on first load (progress bar), then the dashboard is instant.

## 📓 Run the notebook (Google Colab)
1. Open [colab.research.google.com](https://colab.research.google.com) → **Upload** `train_colab.ipynb`.
2. **Runtime → Run all.** It installs OR-Tools, trains the DQN, and produces all charts.
3. **Share → Anyone with the link → Viewer**, and submit that link.

## 💻 Run locally (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```
