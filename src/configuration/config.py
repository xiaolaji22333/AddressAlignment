from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoint'
PRE_TRAINED_DIR = ROOT_DIR / 'pretrained'
LOGS_DIR = ROOT_DIR / 'logs'
LABELS=PROCESSED_DATA_DIR / 'labels.txt'


BATCH_SIZE = 16
MAX_SEQ_LENGTH=32



LABELS = [
  "O",
  "B-assist",
  "I-assist",
  "S-assist",
  "E-assist",
  "B-cellno",
  "I-cellno",
  "E-cellno",
  "B-city",
  "I-city",
  "E-city",
  "B-community",
  "I-community",
  "S-community",
  "E-community",
  "B-devzone",
  "I-devzone",
  "E-devzone",
  "B-district",
  "I-district",
  "S-district",
  "E-district",
  "B-floorno",
  "I-floorno",
  "E-floorno",
  "B-houseno",
  "I-houseno",
  "E-houseno",
  "B-poi",
  "I-poi",
  "S-poi",
  "E-poi",
  "B-prov",
  "I-prov",
  "E-prov",
  "B-road",
  "I-road",
  "E-road",
  "B-roadno",
  "I-roadno",
  "E-roadno",
  "B-subpoi",
  "I-subpoi",
  "E-subpoi",
  "B-town",
  "I-town",
  "E-town",
  "B-intersection",
  "I-intersection",
  "S-intersection",
  "E-intersection",
  "B-distance",
  "I-distance",
  "E-distance",
  "B-village_group",
  "I-village_group",
  "E-village_group",
]