# 🔧 Predictive Maintenance Using Industrial Sensor Time Series

### A Deep Learning–Based Remaining Useful Life (RUL) Prediction System

Predictive maintenance is an important application of Artificial Intelligence in modern industries. Instead of waiting for machinery to fail, predictive maintenance uses historical sensor data and machine-learning/deep-learning techniques to estimate the **Remaining Useful Life (RUL)** of industrial equipment.

This project presents a comparative study of **LSTM, GRU, and Transformer-based deep learning models** for predicting the remaining useful life of turbofan engines using the **NASA C-MAPSS dataset**.

The project also includes a **Streamlit-based application** for interacting with the trained predictive-maintenance models.

---

## 📌 Project Overview

Industrial machines continuously generate sensor measurements such as temperature, pressure, rotational speed, and other operational parameters.

Over time, these measurements contain patterns that indicate equipment degradation.

The objective of this project is to:

- Analyze industrial sensor time-series data
- Preprocess and normalize sensor measurements
- Generate Remaining Useful Life (RUL) targets
- Create sequential time-series samples
- Train deep learning models
- Compare **LSTM, GRU, and Transformer** architectures
- Predict the RUL of industrial equipment
- Save trained models and predictions
- Provide an interactive prediction interface

---

## 🎯 Objectives

The main objectives of this project are:

1. Perform exploratory analysis of industrial sensor data.
2. Preprocess the NASA C-MAPSS dataset.
3. Perform feature selection and scaling.
4. Transform raw sensor data into sequential time-series windows.
5. Develop LSTM and GRU baseline models.
6. Develop a Transformer-based RUL prediction model.
7. Compare the performance of different architectures.
8. Generate RUL predictions for multiple datasets.
9. Build an interactive Streamlit application.
10. Demonstrate how deep learning can support predictive maintenance.

---

## 🏭 Dataset

This project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset.

C-MAPSS provides simulated turbofan engine run-to-failure data containing multiple operational and sensor parameters.

The project uses four datasets:

| Dataset | Description |
|---|---|
| FD001 | Single operating condition and single fault mode |
| FD002 | Multiple operating conditions and single fault mode |
| FD003 | Single operating condition and multiple fault modes |
| FD004 | Multiple operating conditions and multiple fault modes |

Each dataset contains information about:

- Engine/unit ID
- Operational cycle
- Operational settings
- Multiple sensor measurements

The target variable is:

**Remaining Useful Life (RUL)**

---

## 🧠 Models Used

### 1. LSTM

Long Short-Term Memory networks are recurrent neural networks designed to learn long-term dependencies in sequential data.

LSTM is useful for predictive maintenance because sensor readings collected over multiple operating cycles form a time series.

**Advantages:**

- Handles sequential information
- Captures long-term dependencies
- Effective for sensor time-series prediction

---

### 2. GRU

Gated Recurrent Unit is another recurrent neural network architecture similar to LSTM but with a simpler gating mechanism.

GRU generally requires fewer parameters than LSTM while still being capable of learning temporal dependencies.

**Advantages:**

- Faster training than some LSTM configurations
- Fewer parameters
- Effective for sequential sensor data

---

### 3. Transformer

The main focus of this project is the **Transformer architecture**.

Transformers use self-attention mechanisms to learn relationships between different time steps in a sequence.

Unlike traditional recurrent networks, Transformers can process sequence relationships without processing every time step strictly one after another.

**Advantages:**

- Self-attention mechanism
- Captures long-range dependencies
- Highly parallelizable
- Suitable for complex multivariate time-series data

---

## 🔄 Project Workflow

```text
NASA C-MAPSS Dataset
        │
        ▼
Data Loading
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Selection
        │
        ▼
RUL Target Generation
        │
        ▼
Feature Scaling
        │
        ▼
Sequence / Sliding Window Creation
        │
        ▼
┌───────────────┬───────────────┬────────────────┐
│     LSTM      │      GRU      │   Transformer  │
└───────────────┴───────────────┴────────────────┘
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
RUL Prediction
        │
        ▼
Prediction CSV Files
        │
        ▼
Streamlit Application
```

---

## 📂 Repository Structure

```text
Predictive_maintenance_using_industrial_sensor-time_series/
│
├── dataset/
│   ├── FD001/
│   ├── FD002/
│   ├── FD003/
│   └── FD004/
│
├── Predictive_maintainance.ipynb
│
├── app.py
│
├── test_app.py
│
├── transformer_FD001.pth
├── transformer_FD002.pth
├── transformer_FD003.pth
├── transformer_FD004.pth
│
├── FD001_transformer_predictions.csv
├── FD002_transformer_predictions.csv
├── FD003_transformer_predictions.csv
├── FD004_transformer_predictions.csv
│
└── README.md
```

---

## ⚙️ Technologies Used

### Programming Language

- Python

### Machine Learning / Deep Learning

- PyTorch
- Scikit-learn
- NumPy
- Pandas

### Data Visualization

- Matplotlib
- Seaborn

### Application

- Streamlit

### Development Environment

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## 🛠️ Data Preprocessing

The raw C-MAPSS datasets require several preprocessing steps before they can be used for deep learning.

### Main preprocessing steps

1. Load the raw sensor datasets.
2. Assign meaningful column names.
3. Remove irrelevant or low-information sensor features.
4. Calculate the RUL for each engine cycle.
5. Apply feature scaling.
6. Group observations according to engine/unit ID.
7. Generate fixed-length sequences using a sliding window.
8. Split the data into training and testing sets.

### RUL Calculation

For an engine unit, RUL can be calculated using:

```text
RUL = Maximum Cycle of Engine - Current Cycle
```

For example:

```text
Maximum Cycle = 200
Current Cycle = 150

RUL = 200 - 150
    = 50 cycles
```

Therefore, the model learns to estimate how many operational cycles remain before failure.

---

## 🧩 Sequence Generation

Since the input is time-series data, individual sensor rows are not sufficient.

A sliding-window approach is used to create sequences.

For example, with a sequence length of `30`:

```text
Cycle 1  → Sensor Values
Cycle 2  → Sensor Values
Cycle 3  → Sensor Values
...
Cycle 30 → Sensor Values
             ↓
          Model
             ↓
        Predicted RUL
```

The model therefore learns degradation patterns across multiple consecutive operating cycles.

---

## 🤖 Transformer Architecture

The Transformer model processes the sensor sequence using an attention mechanism.

A simplified architecture is:

```text
Input Sensor Sequence
        │
        ▼
Feature Projection
        │
        ▼
Positional Information
        │
        ▼
Transformer Encoder
        │
        ▼
Attention-Based Representation
        │
        ▼
Fully Connected Layer
        │
        ▼
Predicted RUL
```

The model learns which time steps and sensor relationships are most important for estimating the remaining useful life.

---

## 📊 Prediction Outputs

The repository contains Transformer prediction files for all four C-MAPSS datasets:

```text
FD001_transformer_predictions.csv
FD002_transformer_predictions.csv
FD003_transformer_predictions.csv
FD004_transformer_predictions.csv
```

These files contain the model's predicted RUL values and can be used for further analysis and visualization.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/amankr0804/Predictive_maintenance_using_industrial_sensor-time_series.git
```

Navigate to the project directory:

```bash
cd Predictive_maintenance_using_industrial_sensor-time_series
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch streamlit
```

---

## ▶️ Running the Notebook

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Predictive_maintainance.ipynb
```

Run the notebook cells sequentially to perform:

```text
Data Loading
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Sequence Creation
      ↓
Model Training
      ↓
Evaluation
      ↓
RUL Prediction
```

---

## 🌐 Running the Streamlit Application

The project includes a Streamlit application.

Run:

```bash
streamlit run app.py
```

After starting the application, open the local URL displayed in the terminal.

The application provides an interactive interface for exploring the predictive-maintenance workflow and generating predictions using the trained model.

---

## 📈 Model Comparison

The project is designed to compare three deep-learning architectures:

| Model | Architecture | Main Strength |
|---|---|---|
| LSTM | Recurrent Neural Network | Long-term sequential dependencies |
| GRU | Recurrent Neural Network | Efficient sequence learning |
| Transformer | Attention-based | Long-range temporal relationships |

The comparison helps identify which architecture is better suited for multivariate industrial sensor time-series data.

---

## 💡 Why Predictive Maintenance?

Traditional maintenance strategies generally fall into two categories:

### Reactive Maintenance

```text
Machine Fails
     ↓
Maintenance
     ↓
Downtime
```

This can result in:

- Unexpected downtime
- High repair costs
- Production losses
- Equipment damage

### Preventive Maintenance

```text
Fixed Maintenance Schedule
          ↓
     Maintenance
```

This can lead to:

- Unnecessary maintenance
- Increased maintenance cost
- Replacement of healthy components

### Predictive Maintenance

```text
Sensor Data
     ↓
AI / ML Model
     ↓
Degradation Detection
     ↓
RUL Prediction
     ↓
Planned Maintenance
```

The goal is to perform maintenance **before catastrophic failure occurs**, while avoiding unnecessary maintenance.

---

## 🌍 Real-World Applications

The techniques demonstrated in this project can be adapted to:

- ✈️ Aircraft engine maintenance
- 🏭 Manufacturing equipment
- 🚗 Automotive systems
- ⚡ Power plants
- 🛢️ Oil & gas equipment
- 🚂 Railway systems
- 🤖 Industrial robots
- 🔋 Battery health monitoring
- 🌐 Industrial IoT systems

---

## 🔮 Future Improvements

The project can be extended with:

- [ ] Advanced hyperparameter optimization
- [ ] Attention visualization
- [ ] Explainable AI for RUL predictions
- [ ] Real-time sensor-data streaming
- [ ] REST API for model inference
- [ ] Docker deployment
- [ ] Cloud deployment
- [ ] Model monitoring
- [ ] Automated model retraining
- [ ] Real-time maintenance alerts
- [ ] SHAP-based feature importance
- [ ] Ensemble models
- [ ] Improved Transformer architectures

---

## 📌 Limitations

The NASA C-MAPSS dataset is a simulated dataset, so the results may not directly represent every real-world industrial environment.

Real industrial deployment would require:

- Real-time sensor integration
- Reliable sensor calibration
- Handling missing sensor values
- Sensor drift detection
- Domain-specific feature engineering
- Real failure-event data
- Continuous model monitoring

Therefore, this project should be considered a **research and educational predictive-maintenance implementation**, rather than a production safety-critical system.

---

## 📚 Key Learning Outcomes

Through this project, the following concepts were explored:

- Time-series data processing
- Industrial sensor analytics
- Remaining Useful Life prediction
- Feature engineering
- Data normalization
- Sliding-window sequence generation
- LSTM networks
- GRU networks
- Transformer architectures
- Self-attention
- Deep learning model training
- Model evaluation
- PyTorch
- Streamlit deployment
- Predictive maintenance

---

## 👨‍💻 Author

### Aman Kumar

**B.Tech CSE (Data Science)**

Interested in:

- Data Science
- Machine Learning
- Deep Learning
- Full Stack Development
- Artificial Intelligence
- Networking & Cybersecurity

### GitHub

[amankr0804](https://github.com/amankr0804)

### Project Repository

[Predictive Maintenance Using Industrial Sensor Time Series](https://github.com/amankr0804/Predictive_maintenance_using_industrial_sensor-time_series)

---

## ⭐ If You Find This Project Useful

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

Feedback, suggestions, and contributions are welcome.

---

## 📄 License

This project is intended primarily for educational and research purposes.
