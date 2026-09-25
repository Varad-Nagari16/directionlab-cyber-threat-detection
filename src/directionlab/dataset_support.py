
import csv
import re
from pathlib import Path


# Features required by the current NF-UNSW-NB15-v3 model.
NFV3_REQUIRED_FEATURES = [
    "PROTOCOL",
    "L7_PROTO",
    "FLOW_DURATION_MILLISECONDS",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
]


def normalize_column(name: str) -> str:
    """Normalize a column name for comparisons."""
    name = name.strip().replace("\ufeff", "").lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def read_csv_columns(csv_path: str | Path) -> list[str]:
    """Read the CSV header without loading the entire dataset."""
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as file:
        sample = file.read(65536)
        file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=",;\t|",
            )
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(file, dialect)

        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("The CSV file is empty.")

    columns = [column.strip() for column in header]

    if not columns or all(not column for column in columns):
        raise ValueError(
            "Could not find a valid header row in the CSV."
        )

    return columns


def detect_dataset(columns: list[str]) -> dict:
    """
    Guess a dataset family from column names.

    Detection is heuristic. It does not prove the dataset's
    origin and does not guarantee model compatibility.
    """
    normalized = {
        normalize_column(column)
        for column in columns
    }

    # NF-UNSW-NB15-v3 / current model feature schema.
    required = {
        normalize_column(feature)
        for feature in NFV3_REQUIRED_FEATURES
    }

    nfv3_matches = len(required & normalized)

    if required.issubset(normalized):
        dataset = "NF-UNSW-NB15-v3"
        confidence = "feature-schema match"

    # Common CICFlowMeter columns.
    elif {
        "flow_duration",
        "total_fwd_packets",
        "total_backward_packets",
    }.issubset(normalized):
        dataset = "CIC-IDS2017 / CIC-IDS2018"
        confidence = "likely CICFlowMeter format"

    # Common original UNSW-NB15 fields.
    elif {
        "srcip",
        "dstip",
        "sport",
        "dport",
        "proto",
    }.issubset(normalized):
        dataset = "UNSW-NB15"
        confidence = "likely UNSW-NB15 format"

    # Some UGR16-derived files use expanded categorical fields.
    elif (
        "srcipprivate" in normalized
        or "dporthttp" in normalized
        or "protocoltcp" in normalized
    ):
        dataset = "UGR16 / one-hot encoded flow data"
        confidence = "possible UGR16-derived format"

    else:
        dataset = "Unknown / custom CSV"
        confidence = "no known schema matched"

    missing_nfv3 = sorted(required - normalized)

    return {
        "detected_dataset": dataset,
        "detection_basis": confidence,
        "column_count": len(columns),
        "nfv3_feature_matches": nfv3_matches,
        "nfv3_required_feature_count": len(required),
        "compatible_with_current_nfv3_schema": not missing_nfv3,
        "missing_nfv3_features": missing_nfv3,
        "columns": columns,
    }


def inspect_csv(csv_path: str | Path) -> dict:
    """Inspect a CSV and return its dataset/schema report."""
    columns = read_csv_columns(csv_path)
    return detect_dataset(columns)