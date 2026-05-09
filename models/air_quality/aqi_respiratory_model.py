"""
ClimateShield AI — AQI-Driven Respiratory Risk Model (LSTM)

Predicts 3-day ahead respiratory illness risk in children from air
quality time series (PM2.5, PM10, NO2, O3) from OpenAQ stations.

Children under 12 have narrower airways and higher breathing rates:
  - PM2.5 lung damage threshold: 25 μg/m³ (vs 35 for adults)
  - NO2 sensitization: increases asthma frequency in under-5s
  - Ozone: peak school-hour exposure (10am–4pm) is most harmful

Architecture: LSTM encoder → Risk classifier
  Input:  14-day rolling window of hourly/daily AQI measurements
  Output: risk_label (0=Low, 1=Moderate, 2=High, 3=Emergency)

References:
  - WHO Air Quality Guidelines 2021
    https://www.who.int/publications/i/item/9789240034228
  - Dockery et al. (1993) PM2.5 and childhood respiratory mortality
  - HEI Panel (2010) Traffic-Related Air Pollution and Children's Health
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Child-specific WHO AQI thresholds (stricter than adult guidelines)
CHILD_PM25_THRESHOLDS = {
    "Low":       (0, 15),     # WHO annual guideline: 5 μg/m³; daily: 15
    "Moderate":  (15, 35),    # Increased respiratory symptoms in children
    "High":      (35, 75),    # Asthma exacerbation risk, school alert
    "Emergency": (75, 9999),  # Outdoor activity ban for children
}


class AQIRespiratoryRiskModel:
    """
    LSTM-based respiratory risk forecasting from air quality time series.

    Input shape: (batch, sequence_length=14, n_features=4)
    Features: [pm25, pm10, no2, o3] — daily averages
    """

    def __init__(
        self,
        sequence_length: int = 14,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 4,
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_classes = n_classes
        self.model = None
        self.scaler = None

    def _build_model(self):
        """Build PyTorch LSTM model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("pip install torch")

        class LSTMRiskClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, n_classes),
                )

            def forward(self, x):
                # x: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)
                # Use last timestep
                last_hidden = lstm_out[:, -1, :]
                out = self.dropout(last_hidden)
                return self.classifier(out)

        return LSTMRiskClassifier(
            input_size=4,  # pm25, pm10, no2, o3
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            n_classes=self.n_classes,
            dropout=self.dropout,
        )

    def prepare_sequences(self, df: pd.DataFrame) -> tuple:
        """
        Convert daily AQI dataframe into (X, y) sequences for LSTM.

        df must have columns: date, pm25, pm10, no2, o3, respiratory_label
        """
        from sklearn.preprocessing import StandardScaler

        features = ["pm25", "pm10", "no2", "o3"]
        df = df.sort_values("date").copy()
        df[features] = df[features].fillna(df[features].median())

        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(df[features])
        else:
            scaled = self.scaler.transform(df[features])

        X, y = [], []
        for i in range(self.sequence_length, len(df)):
            X.append(scaled[i - self.sequence_length:i])
            if "respiratory_label" in df.columns:
                y.append(df["respiratory_label"].iloc[i])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64) if y else None
        return X, y

    def train(self, df: pd.DataFrame, epochs: int = 50, lr: float = 1e-3):
        """Train LSTM on labelled AQI time series."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("pip install torch")

        X, y = self.prepare_sequences(df)
        if y is None:
            raise ValueError("DataFrame must include 'respiratory_label' column for training.")

        self.model = self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                log.info(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(loader):.4f}")

        log.info("AQI LSTM training complete.")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict respiratory risk for the next 3 days."""
        import torch

        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X, _ = self.prepare_sequences(df)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X))
            probas = torch.softmax(logits, dim=1).numpy()
            labels = np.argmax(probas, axis=1)

        label_names = list(CHILD_PM25_THRESHOLDS.keys())
        result = df.iloc[self.sequence_length:].copy().reset_index(drop=True)
        result["respiratory_risk_label"] = [label_names[l] for l in labels]
        result["respiratory_risk_score"] = (labels / 3 * 100).astype(float)
        for i, name in enumerate(label_names):
            result[f"proba_{name.lower()}"] = probas[:, i]

        return result

    def simple_rule_based_score(self, pm25: float, pm10: float = None,
                                no2: float = None) -> dict:
        """
        Fallback rule-based scoring when LSTM model is not loaded.
        Uses WHO child-adjusted thresholds directly.
        Used by offline mobile app.
        """
        for label, (lo, hi) in CHILD_PM25_THRESHOLDS.items():
            if lo <= pm25 < hi:
                score = ((pm25 - lo) / (hi - lo) * 25 +
                         list(CHILD_PM25_THRESHOLDS.keys()).index(label) * 25)
                return {
                    "respiratory_risk_label": label,
                    "respiratory_risk_score": min(score, 100),
                    "pm25_μgm3": pm25,
                    "who_child_threshold_exceeded": pm25 > 15,
                    "school_outdoor_ban": pm25 > 75,
                }
        return {"respiratory_risk_label": "Emergency", "respiratory_risk_score": 100}

    def save(self, path: str):
        import torch, joblib
        torch.save(self.model.state_dict(), path + ".pt")
        joblib.dump(self.scaler, path + "_scaler.joblib")
        log.info(f"Model saved to {path}")

    def load(self, path: str):
        import torch, joblib
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(path + ".pt", map_location="cpu"))
        self.model.eval()
        self.scaler = joblib.load(path + "_scaler.joblib")
        log.info(f"Model loaded from {path}")


if __name__ == "__main__":
    # Quick test with rule-based scoring (no training data needed)
    model = AQIRespiratoryRiskModel()

    # Lahore smog season (November–January AQI often 300+)
    lahore_smog = model.simple_rule_based_score(pm25=280.0, pm10=420.0)
    print("Lahore smog season:")
    for k, v in lahore_smog.items():
        print(f"  {k}: {v}")

    # Islamabad clear day
    islamabad_clear = model.simple_rule_based_score(pm25=8.0)
    print("\nIslamabad clear day:")
    for k, v in islamabad_clear.items():
        print(f"  {k}: {v}")
