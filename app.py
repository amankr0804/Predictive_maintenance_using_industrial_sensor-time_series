# ============================================================
# NASA C-MAPSS TRANSFORMER PREDICTIVE MAINTENANCE
# Streamlit Frontend
# ============================================================

import os
import io
import math

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NASA Aircraft Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .metric-card {
        background: #1a1d24;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30343b;
        text-align: center;
    }

    .metric-title {
        color: #9ca3af;
        font-size: 14px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        margin-top: 5px;
    }

    .health-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
        background: #1a1d24;
        border: 1px solid #30343b;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

DATASETS = [
    "FD001",
    "FD002",
    "FD003",
    "FD004"
]

SEQ_LEN_DEFAULT = 30
RUL_CAP_DEFAULT = 125


# ============================================================
# 4. C-MAPSS COLUMN NAMES
# ============================================================

COLUMN_NAMES = [
    "unit",
    "cycle",
    "op1",
    "op2",
    "op3"
]

COLUMN_NAMES += [
    f"sensor_{i}"
    for i in range(1, 22)
]


# ============================================================
# 5. DEFAULT FEATURES
# ============================================================

DEFAULT_FEATURES = COLUMN_NAMES[2:]


# ============================================================
# 6. POSITIONAL ENCODING
# EXACTLY MATCHES TRAINING MODEL
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=30
    ):

        super().__init__()

        position = torch.arange(
            max_len
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            *
            (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        return (
            x
            +
            self.pe[
                :,
                :x.size(1)
            ]
        )


# ============================================================
# 7. TRANSFORMER MODEL
# EXACTLY MATCHES TRAINING MODEL
# ============================================================

class TransformerRUL(nn.Module):

    def __init__(
        self,
        input_features,
        d_model=64,
        n_heads=4,
        num_layers=3,
        dropout=0.1
    ):

        super().__init__()

        # ----------------------------------------------------
        # Input Projection
        # ----------------------------------------------------

        self.input_projection = nn.Linear(
            input_features,
            d_model
        )

        # ----------------------------------------------------
        # Positional Encoding
        # ----------------------------------------------------

        self.positional_encoding = PositionalEncoding(
            d_model,
            max_len=30
        )

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(
            d_model
        )

        # ----------------------------------------------------
        # Regression Head
        # ----------------------------------------------------

        self.regressor = nn.Sequential(

            nn.Linear(
                d_model,
                64
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                32
            ),

            nn.GELU(),

            nn.Linear(
                32,
                1
            )
        )

    def forward(self, x):

        # x:
        # [batch, sequence, features]

        x = self.input_projection(
            x
        )

        x = self.positional_encoding(
            x
        )

        x = self.transformer(
            x
        )

        # Last timestep
        x = x[:, -1, :]

        # Normalization
        x = self.norm(
            x
        )

        # Regression
        x = self.regressor(
            x
        )

        return x.squeeze(-1)


# ============================================================
# 8. MODEL PATH
# ============================================================

def get_model_path(dataset):

    current_folder = os.path.dirname(
        os.path.abspath(__file__)
    )

    filename = (
        f"transformer_{dataset}.pth"
    )

    return os.path.join(
        current_folder,
        filename
    )


# ============================================================
# 9. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(dataset):

    model_path = get_model_path(
        dataset
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    # --------------------------------------------------------
    # Read saved configuration
    # --------------------------------------------------------

    features = checkpoint.get(
        "features",
        DEFAULT_FEATURES
    )

    sequence_length = checkpoint.get(
        "sequence_length",
        SEQ_LEN_DEFAULT
    )

    rul_cap = checkpoint.get(
        "rul_cap",
        RUL_CAP_DEFAULT
    )

    scaler_mean = checkpoint.get(
        "scaler_mean",
        None
    )

    scaler_scale = checkpoint.get(
        "scaler_scale",
        None
    )

    # Convert feature names safely
    features = list(features)

    # --------------------------------------------------------
    # Create exact training architecture
    # --------------------------------------------------------

    model = TransformerRUL(
        input_features=len(features),
        d_model=64,
        n_heads=4,
        num_layers=3,
        dropout=0.1
    )

    # --------------------------------------------------------
    # Extract model state
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    # --------------------------------------------------------
    # Restore StandardScaler
    # --------------------------------------------------------

    scaler = None

    if (
        scaler_mean is not None
        and scaler_scale is not None
    ):

        scaler = {
            "mean": np.asarray(
                scaler_mean,
                dtype=np.float32
            ),

            "scale": np.asarray(
                scaler_scale,
                dtype=np.float32
            )
        }

    return (
        model,
        features,
        sequence_length,
        rul_cap,
        scaler
    )


# ============================================================
# 10. READ C-MAPSS DATA
# ============================================================

def read_cmapss_file(
    uploaded_file
):

    raw = uploaded_file.getvalue()

    # --------------------------------------------------------
    # Standard NASA C-MAPSS files are whitespace separated
    # --------------------------------------------------------

    df = pd.read_csv(
        io.BytesIO(raw),
        sep=r"\s+",
        header=None,
        engine="python"
    )

    # Remove empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # --------------------------------------------------------
    # Keep first 26 columns
    # --------------------------------------------------------

    if df.shape[1] < 26:

        raise ValueError(
            f"Expected 26 columns, "
            f"but found {df.shape[1]}."
        )

    df = df.iloc[
        :,
        :26
    ].copy()

    df.columns = COLUMN_NAMES

    # --------------------------------------------------------
    # Convert numeric
    # --------------------------------------------------------

    for column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        how="any"
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# 11. READ RUL FILE
# ============================================================

def read_rul_file(
    uploaded_file
):

    raw = uploaded_file.getvalue()

    rul = pd.read_csv(
        io.BytesIO(raw),
        header=None
    )

    rul = pd.to_numeric(
        rul.iloc[:, 0],
        errors="coerce"
    )

    rul = rul.dropna()

    return rul.to_numpy(
        dtype=np.float32
    )


# ============================================================
# 12. SCALE FEATURES
# ============================================================

def scale_features(
    values,
    scaler
):

    if scaler is None:

        return values

    mean = np.asarray(
        scaler["mean"],
        dtype=np.float32
    )

    scale = np.asarray(
        scaler["scale"],
        dtype=np.float32
    )

    scale = np.where(
        scale == 0,
        1.0,
        scale
    )

    return (
        values - mean
    ) / scale


# ============================================================
# 13. PREPARE ENGINE SEQUENCE
# ============================================================

def prepare_engine_sequence(
    engine_df,
    features,
    sequence_length,
    scaler
):

    # Sort by cycle
    engine_df = engine_df.sort_values(
        "cycle"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Extract model features
    # --------------------------------------------------------

    values = engine_df[
        features
    ].values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Scale exactly like training
    # --------------------------------------------------------

    values = scale_features(
        values,
        scaler
    )

    # --------------------------------------------------------
    # Create last sequence
    # --------------------------------------------------------

    if len(values) >= sequence_length:

        sequence = values[
            -sequence_length:
        ]

    else:

        padding_count = (
            sequence_length
            - len(values)
        )

        # Pad using first observation
        padding = np.repeat(
            values[:1],
            padding_count,
            axis=0
        )

        sequence = np.concatenate(
            [
                padding,
                values
            ],
            axis=0
        )

    # --------------------------------------------------------
    # Convert to PyTorch tensor
    # --------------------------------------------------------

    sequence = torch.tensor(
        sequence,
        dtype=torch.float32
    )

    # [sequence, features]
    # ->
    # [1, sequence, features]

    sequence = sequence.unsqueeze(
        0
    )

    return sequence


# ============================================================
# 14. PREDICT RUL
# ============================================================

def predict_rul(
    model,
    sequence,
    rul_cap
):

    model.eval()

    with torch.no_grad():

        prediction = model(
            sequence
        )

    prediction = float(
        prediction.item()
    )

    # RUL cannot be negative
    prediction = max(
        0.0,
        prediction
    )

    # Cap prediction
    prediction = min(
        float(rul_cap),
        prediction
    )

    return prediction


# ============================================================
# 15. HEALTH STATUS
# ============================================================

def get_health_status(
    rul
):

    if rul <= 10:

        return (
            "🔴 CRITICAL",
            "Immediate maintenance is recommended.",
            "critical"
        )

    elif rul <= 30:

        return (
            "🟠 WARNING",
            "Maintenance should be planned soon.",
            "warning"
        )

    elif rul <= 60:

        return (
            "🟡 MODERATE",
            "Continue monitoring engine condition.",
            "moderate"
        )

    else:

        return (
            "🟢 HEALTHY",
            "Engine currently has a relatively high estimated RUL.",
            "healthy"
        )


# ============================================================
# 16. RUL GAUGE
# ============================================================

def create_rul_gauge(
    rul,
    rul_cap
):

    fig = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=rul,

            number={
                "suffix": " cycles",
                "font": {
                    "size": 36
                }
            },

            title={
                "text":
                "Predicted Remaining Useful Life"
            },

            gauge={

                "axis": {
                    "range": [
                        0,
                        rul_cap
                    ]
                },

                "bar": {
                    "color": "#4ade80"
                },

                "steps": [

                    {
                        "range": [
                            0,
                            10
                        ],
                        "color": "#7f1d1d"
                    },

                    {
                        "range": [
                            10,
                            30
                        ],
                        "color": "#9a3412"
                    },

                    {
                        "range": [
                            30,
                            60
                        ],
                        "color": "#854d0e"
                    },

                    {
                        "range": [
                            60,
                            rul_cap
                        ],
                        "color": "#166534"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(
            l=30,
            r=30,
            t=70,
            b=20
        )
    )

    return fig


# ============================================================
# 17. SENSOR PLOT
# ============================================================

def create_sensor_plot(
    engine_df,
    sensor
):

    fig = px.line(
        engine_df,
        x="cycle",
        y=sensor,
        title=f"{sensor} Trend"
    )

    fig.update_layout(
        height=400,
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30
        )
    )

    return fig


# ============================================================
# 18. HEADER
# ============================================================

st.markdown(
    """
    <div class="title">
        ✈️ Predictive Maintenance
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Transformer-based Remaining Useful Life (RUL) Prediction
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 19. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Configuration"
    )

    dataset = st.selectbox(
        "Select Dataset",
        DATASETS
    )

    st.divider()

    st.subheader(
        "Dataset Information"
    )

    dataset_info = {

        "FD001":
            "1 operating condition • 1 fault mode",

        "FD002":
            "6 operating conditions • 1 fault mode",

        "FD003":
            "1 operating condition • 2 fault modes",

        "FD004":
            "6 operating conditions • 2 fault modes"
    }

    st.info(
        dataset_info[
            dataset
        ]
    )

    st.divider()

    st.subheader(
        "Model"
    )

    st.write(
        "Architecture: Transformer"
    )

    st.write(
        "Sequence Length: 30"
    )

    st.write(
        "Hidden Dimension: 64"
    )

    st.write(
        "Attention Heads: 4"
    )

    st.write(
        "Encoder Layers: 3"
    )

    st.divider()

    st.caption(
        "NASA C-MAPSS Predictive Maintenance"
    )


# ============================================================
# 20. LOAD SELECTED MODEL
# ============================================================

try:

    (
        model,
        features,
        sequence_length,
        rul_cap,
        scaler
    ) = load_model(
        dataset
    )

    st.sidebar.success(
        f"✓ {dataset} model loaded"
    )

except Exception as error:

    st.error(
        "❌ Could not load the Transformer model."
    )

    st.error(
        str(error)
    )

    st.code(
        get_model_path(dataset)
    )

    st.info(
        """
        Make sure the model file is in the same
        folder as app.py.

        Example:

        transformer_FD001.pth
        """
    )

    st.stop()


# ============================================================
# 21. MODEL DETAILS
# ============================================================

with st.expander(
    "🔍 Model Details"
):

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.metric(
            "Input Features",
            len(features)
        )

    with col2:

        st.metric(
            "Sequence Length",
            sequence_length
        )

    with col3:

        st.metric(
            "RUL Cap",
            rul_cap
        )

    with col4:

        st.metric(
            "Device",
            "CPU"
        )

    st.write(
        "Features used by this model:"
    )

    st.code(
        ", ".join(features)
    )


# ============================================================
# 22. FILE UPLOAD
# ============================================================

st.header(
    "📂 Upload C-MAPSS Test Data"
)

uploaded_file = st.file_uploader(
    "Upload test_FD00X.txt",
    type=[
        "txt",
        "csv"
    ],
    help=(
        "Upload the C-MAPSS test file "
        "corresponding to the selected dataset."
    )
)

rul_file = st.file_uploader(
    "Optional: Upload RUL_FD00X.txt",
    type=[
        "txt",
        "csv"
    ],
    help=(
        "Optional actual RUL file for evaluating "
        "the selected engine."
    )
)


# ============================================================
# 23. WAIT FOR FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a C-MAPSS test file to begin."
    )

    st.markdown(
        """
        ### Expected C-MAPSS format

        Each row contains:

        1. Unit number
        2. Cycle
        3. Operating setting 1
        4. Operating setting 2
        5. Operating setting 3
        6. Sensor 1
        7. Sensor 2
        8. ...
        26. Sensor 21

        **Total = 26 columns**
        """
    )

    st.stop()


# ============================================================
# 24. READ UPLOADED FILE
# ============================================================

try:

    df = read_cmapss_file(
        uploaded_file
    )

except Exception as error:

    st.error(
        "❌ Could not read uploaded file."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# 25. DATASET OVERVIEW
# ============================================================

st.header(
    "📊 Dataset Overview"
)

col1, col2, col3, col4 = st.columns(
    4
)

with col1:

    st.metric(
        "Total Rows",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "Engines",
        df["unit"].nunique()
    )

with col3:

    st.metric(
        "Sensors",
        21
    )

with col4:

    st.metric(
        "Model Features",
        len(features)
    )


# ============================================================
# 26. ENGINE SELECTION
# ============================================================

st.header(
    "🔧 Engine Selection"
)

engine_ids = sorted(
    df["unit"].unique()
)

selected_engine = st.selectbox(
    "Select Engine",
    engine_ids
)

engine_df = df[
    df["unit"] == selected_engine
].copy()

engine_df = engine_df.sort_values(
    "cycle"
).reset_index(
    drop=True
)


# ============================================================
# 27. ENGINE SUMMARY
# ============================================================

col1, col2, col3 = st.columns(
    3
)

with col1:

    st.metric(
        "Engine ID",
        int(selected_engine)
    )

with col2:

    st.metric(
        "Current Cycle",
        int(
            engine_df["cycle"].max()
        )
    )

with col3:

    st.metric(
        "Observed Cycles",
        len(engine_df)
    )


# ============================================================
# 28. PREPARE SEQUENCE
# ============================================================

try:

    sequence = prepare_engine_sequence(
        engine_df,
        features,
        sequence_length,
        scaler
    )

except Exception as error:

    st.error(
        "❌ Could not prepare model input."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# 29. PREDICT
# ============================================================

try:

    predicted_rul = predict_rul(
        model,
        sequence,
        rul_cap
    )

except Exception as error:

    st.error(
        "❌ RUL prediction failed."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# 30. HEALTH STATUS
# ============================================================

(
    health_status,
    health_message,
    health_level
) = get_health_status(
    predicted_rul
)


# ============================================================
# 31. RUL PREDICTION
# ============================================================

st.header(
    "🔮 RUL Prediction"
)

col1, col2 = st.columns(
    [1.2, 1]
)

with col1:

    st.plotly_chart(
        create_rul_gauge(
            predicted_rul,
            rul_cap
        ),
        use_container_width=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-title">
                Predicted Remaining Useful Life
            </div>

            <div class="metric-value">
                {predicted_rul:.2f} cycles
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="health-card">

            <h2>
                {health_status}
            </h2>

            <p>
                {health_message}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if health_level == "critical":

        st.error(
            "⚠️ Immediate maintenance attention is recommended."
        )

    elif health_level == "warning":

        st.warning(
            "⚠️ Maintenance planning should be considered soon."
        )

    elif health_level == "moderate":

        st.info(
            "ℹ️ Continue monitoring the engine condition."
        )

    else:

        st.success(
            "✓ Engine currently appears to be relatively healthy."
        )


# ============================================================
# 32. ACTUAL RUL COMPARISON
# ============================================================

if rul_file is not None:

    try:

        actual_ruls = read_rul_file(
            rul_file
        )

        # C-MAPSS engine IDs normally start at 1
        engine_index = int(
            selected_engine
        ) - 1

        if (
            0 <= engine_index
            < len(actual_ruls)
        ):

            actual_rul = float(
                actual_ruls[
                    engine_index
                ]
            )

            absolute_error = abs(
                predicted_rul
                -
                actual_rul
            )

            st.header(
                "🎯 Prediction Evaluation"
            )

            c1, c2, c3 = st.columns(
                3
            )

            with c1:

                st.metric(
                    "Predicted RUL",
                    f"{predicted_rul:.2f}"
                )

            with c2:

                st.metric(
                    "Actual RUL",
                    f"{actual_rul:.2f}"
                )

            with c3:

                st.metric(
                    "Absolute Error",
                    f"{absolute_error:.2f}"
                )

        else:

            st.warning(
                "No matching RUL value was found "
                "for this engine."
            )

    except Exception as error:

        st.warning(
            f"Could not evaluate actual RUL: {error}"
        )


# ============================================================
# 33. SENSOR TRENDS
# ============================================================

st.header(
    "📈 Sensor Trend Analysis"
)

sensor_columns = [
    column
    for column in COLUMN_NAMES
    if column.startswith("sensor_")
]

selected_sensor = st.selectbox(
    "Select Sensor",
    sensor_columns
)

st.plotly_chart(
    create_sensor_plot(
        engine_df,
        selected_sensor
    ),
    use_container_width=True
)


# ============================================================
# 34. MULTI-SENSOR ANALYSIS
# ============================================================

st.subheader(
    "🔬 Compare Multiple Sensors"
)

selected_sensors = st.multiselect(
    "Select sensors",
    sensor_columns,
    default=sensor_columns[:3]
)

if selected_sensors:

    chart_df = engine_df[
        ["cycle"]
        +
        selected_sensors
    ].copy()

    chart_df = chart_df.melt(
        id_vars="cycle",
        var_name="Sensor",
        value_name="Value"
    )

    fig = px.line(
        chart_df,
        x="cycle",
        y="Value",
        color="Sensor",
        title="Selected Sensor Trends"
    )

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 35. LATEST SENSOR VALUES
# ============================================================

st.header(
    "📋 Latest Sensor Measurements"
)

latest_row = engine_df.iloc[-1]

latest_values = []

for sensor in sensor_columns:

    latest_values.append(
        {
            "Sensor": sensor,
            "Latest Value": float(
                latest_row[sensor]
            )
        }
    )

latest_df = pd.DataFrame(
    latest_values
)

latest_df[
    "Latest Value"
] = latest_df[
    "Latest Value"
].round(4)

st.dataframe(
    latest_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 36. RAW ENGINE DATA
# ============================================================

with st.expander(
    "🔎 View Raw Engine Data"
):

    st.dataframe(
        engine_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 37. FOOTER
# ============================================================

st.divider()

st.caption(
    "NASA C-MAPSS • Transformer-based Predictive Maintenance • "
    "Remaining Useful Life Prediction"
)