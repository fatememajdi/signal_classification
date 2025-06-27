# 🧠 Signal Classification Using Deep Learning

This project focuses on classifying **vibration signals** from industrial machines into two categories: **Healthy** and **Faulty**. Two deep learning pipelines are implemented:

- ✅ 1D CNN: Applied directly on raw time-series signals.
- ✅ 2D CNN: Applied on spectrograms (frequency-domain representations of signals).

The project supports full preprocessing, training (with random hyperparameter search), and evaluation pipelines.

---

## 📁 Project Structure

```
CNN-Project/
├── data/                   # Raw and processed signal data
│   ├── raw/
│       ├── Normal Condition/   # Healthy signals (.txt)
│       └── Looseness/          # Faulty signals (.txt)
│   └── Looseness/          # Faulty signals (.txt)
├── models/                # Trained models (.h5 files)
├── notebooks/             # Jupyter notebooks for training and evaluation
├── src/                   # Source code modules
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## ⚙️ Prerequisites

- Python >= 3.11  
- Install all required libraries using:

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- numpy==2.0.2  
- pandas==1.5.3  
- scikit-learn==1.6.1  
- tensorflow==2.18.0  
- scipy==1.15.3  
- matplotlib==3.10.0  

---

## 🧪 How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/CNN-Project.git
cd CNN-Project
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run notebooks in order:**

| Step | Notebook | Description |
|------|----------|-------------|
| 1️⃣   | `train_hyperparameter_search.ipynb` | Train and tune 1D CNN |
| 2️⃣   | `train_hyperparameter_search_2d.ipynb` | Train and tune 2D CNN |
| 3️⃣   | `evaluate.ipynb` | Load a signal and classify using trained models |

> Make sure `.txt` signal files are placed inside:
> - `data/raw/Normal Condition/`  
> - `data/raw/Looseness/`

---

## 🧱 Models

### 🔷 Model 1: 1D CNN
- **Input:** Raw signal (shape: `(25000, 1)`)
- **Architecture:**
  - Conv1D → MaxPool → Conv1D → Dropout → Conv1D → GlobalAveragePooling → Dense(2)
- **Use Case:** Direct time-domain classification

### 🔷 Model 2: 2D CNN
- **Input:** Spectrogram (shape: `(128, 128, 1)`)
- **Architecture:**
  - Conv2D → BatchNorm → MaxPool → Conv2D → MaxPool → Flatten → Dropout → Dense → Dense(2)
- **Use Case:** Classification in frequency domain

---

## 📊 Data

- **Source:** Vibration signals from industrial IoT sensors
- **Labels:**  
  - `0`: Healthy  
  - `1`: Faulty  

- **Format:** `.txt` files with metadata headers  
- **Preprocessing:**
  - Normalize to length = 25000
  - MinMax scaling to [0, 1]
  - Spectrogram transformation for 2D CNN

---

## 🧾 Outputs

- Trained models saved to `models/` directory:
  - `best_1D_model_randomsearch.h5`
  - `best_2DCNN_model_randomsearch.h5`

---

## 🔧 Maintenance

- Update dependencies in `requirements.txt` as needed.
- Retrain models if preprocessing or architecture changes.
- Monitor data quality over time to avoid model drift.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♀️ Author

**Fateme Majdi**  
Master’s student – Artificial Intelligence & Robotics  
Shahid Bahonar University of Kerman

---

## 🌟 Acknowledgments

Special thanks to our mentors and contributors for guidance throughout this project.