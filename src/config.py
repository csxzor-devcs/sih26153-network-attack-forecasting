"""
SIH26153 -- Network Attack Forecasting Configuration

Stage mapping: CIC-IDS2017 labels → attack stages
MITRE mapping: stages → ATT&CK tactics
Sequence configuration
"""

# Stage map: CIC-IDS2017 label strings → stage names
STAGE_MAP = {
    "BENIGN": "Benign",
    "PortScan": "Recon",
    "FTP-Patator": "CredAccess",
    "SSH-Patator": "CredAccess",
    "DoS Hulk": "Impact",
    "DoS GoldenEye": "Impact",
    "DoS slowloris": "Impact",
    "DoS Slowhttptest": "Impact",
    "Heartbleed": "Exploit",
    "Web Attack – Brute Force": "CredAccess",
    "Web Attack – XSS": "Exploit",
    "Web Attack – Sql Injection": "Exploit",
    "Infiltration": "LateralMove",
    "Bot": "C2",
    "DDoS": "Impact",
}

# MITRE ATT&CK map: stage names → (tactic ID, tactic name)
# P2 FIX: Corrected to use actual ATT&CK tactic definitions
MITRE_MAP = {
    "Recon": ("TA0043", "Reconnaissance"),
    "CredAccess": ("TA0006", "Credential Access"),
    "Exploit": ("TA0001", "Initial Access"),
    "LateralMove": ("TA0011", "Lateral Movement"),
    "C2": ("TA0011", "Command and Control"),
    "Impact": ("TA0040", "Impact"),
    "Benign": ("", "Benign"),
}

# Stage order: natural progression of an attack campaign
STAGE_ORDER = ["Benign", "Recon", "CredAccess", "Exploit", "LateralMove", "C2", "Impact"]

# Sliding window length (number of flows to look at for prediction)
WINDOW_SIZE = 20

# Forecast horizon: predict H steps ahead from end of window
FORECAST_HORIZON = 1

# Stage to index mapping
STAGE_TO_IDX = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}

# Forecast lead time: number of flow intervals between window end and target
FORECAST_LEAD_TIME = FORECAST_HORIZON  # = 1 flow ahead for H=1

# Feature columns: 20 most informative CIC-IDS features
FEATURE_COLS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Max",
    "Packet Length Min",
    "SYN Flag Count",
    "FIN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Average Packet Size",
    "Fwd Segment Size",
    "Bwd Segment Size",
    "Init Win Bytes",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

# Paths
MODEL_DIR = "models"
SEQUENCE_DIR = "data/sequences"
PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"