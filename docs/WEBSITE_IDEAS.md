# 🌐 Website Ideas — Frontend & Backend Roadmap

> Ideas for a web application to showcase the self-training pipeline.
> We'll build this after upscaling the model.

---

## Core Feature: X-Ray Disease Predictor

### Upload & Predict
- User uploads a chest X-ray image
- Model runs inference → returns 14 disease probabilities
- Display confidence bars (0-100%) for each disease, color-coded:
  - 🟢 Green (< 30%) — likely healthy
  - 🟡 Yellow (30-60%) — uncertain
  - 🔴 Red (> 60%) — likely disease

### Grad-CAM Heatmap (Disease Localization)
- **Our current model can NOT draw bounding boxes** — it only classifies
- **BUT** we can add **Grad-CAM** — generates a heatmap showing WHERE the model looked
- Overlay the heatmap on the original X-ray → shows suspicious regions
- Easy to implement (~50 lines of code), looks impressive in demos

---

## Showcase Features

### Self-Training Progression Dashboard
- Interactive chart showing AUROC improving across 5 rounds
- Animated: watch the labeled pool grow from 784 → 43,020
- Show how pseudo-labels get added each round

### Before/After Comparison
- Same X-ray, two predictions side by side:
  - LEFT: Baseline model (AUROC 0.67) — weak predictions
  - RIGHT: Self-trained model (AUROC 0.78) — confident predictions
- Visually demonstrates the improvement

### Method Comparison
- Interactive bar chart comparing all 5 methods
- Click on a method → see details (epochs, data used, training time)

### Threshold Explorer
- Slider to adjust confidence threshold
- See how many pseudo-labels pass at different thresholds
- Educational: shows the tradeoff between quantity and quality

---

## Sample Gallery
- Pre-loaded X-ray samples so users don't need their own
- Include examples of: clear lungs, pneumonia, cardiomegaly, effusion
- Click any sample → runs inference immediately

---

## Educational Section
- Step-by-step animated explainer of self-training
- "How does the model learn from its own predictions?"
- Show the pipeline visually

---

## Tech Stack Options

| Option | Pros | Cons |
|--------|------|------|
| **Flask + HTML/CSS/JS** | Simple, full control | More manual work |
| **Streamlit** | Fastest to build, free hosting | Less design control |
| **FastAPI + React** | Professional, fast API | More complex |
| **Gradio** | Built for ML demos, 5-min setup | Limited customization |

### Recommended: **Streamlit** for quick demo, **Flask** for polished showcase

---

## Deployment Options
- **Streamlit Cloud** — free, auto-deploys from GitHub
- **Hugging Face Spaces** — free, great for ML demos
- **Local** — just `python app.py` for presentations

---

## Implementation Priority
1. 🥇 Upload X-ray → disease predictions with confidence bars
2. 🥈 Grad-CAM heatmap overlay
3. 🥉 Sample gallery with pre-loaded images
4. 📊 Self-training progression chart
5. 🔄 Before/after comparison
6. 📖 Educational explainer
