"""Prepare NetFlow v3 using only source-to-destination directional features.

The source archive includes both directions. This experiment deliberately excludes
IN/OUT, DST_TO_SRC, server-side, and cumulative bidirectional fields before modeling.
Splits are source-host held out by deterministic source-address hashing.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "PROTOCOL", "L7_PROTO", "FLOW_DURATION_MILLISECONDS", "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "SRC_TO_DST_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV", "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
]

def bucket(address: object) -> int:
    return int(hashlib.sha256(str(address).encode()).hexdigest()[:8], 16) % 10

def host_id(address: object) -> str:
    return hashlib.sha256(str(address).encode()).hexdigest()[:16]

def clean_label(value: object) -> str:
    return "BENIGN" if str(value).strip().lower() == "benign" else str(value).strip()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/nfv3_directional")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    writers = {}
    counts = {name: 0 for name in ("train", "calibration", "test")}
    label_counts = {name: {} for name in counts}
    usecols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"] + FEATURES
    attack_hosts: set[str] = set()
    for chunk in pd.read_csv(args.input, usecols=["IPV4_SRC_ADDR", "Attack"], chunksize=200_000, low_memory=False):
        attack_hosts.update(chunk.loc[chunk["Attack"].astype(str).str.strip().str.lower() != "benign", "IPV4_SRC_ADDR"].astype(str).unique())
    attack_hosts = set(sorted(attack_hosts))
    if len(attack_hosts) < 3:
        raise ValueError(f"Need at least three attack-bearing source hosts, found {len(attack_hosts)}")
    attack_hosts = sorted(attack_hosts)
    host_assignment = {host: ("train" if index < len(attack_hosts) - 2 else "calibration" if index == len(attack_hosts) - 2 else "test") for index, host in enumerate(attack_hosts)}
    try:
        for chunk in pd.read_csv(args.input, usecols=usecols, chunksize=100_000, low_memory=False):
            chunk["Label"] = chunk["Attack"].map(clean_label)
            groups = chunk["IPV4_SRC_ADDR"].astype(str).map(lambda host: host_assignment.get(host, f"benign_{bucket(host)}"))
            chunk["label_binary"] = (chunk["Label"] != "BENIGN").astype(np.int8)
            chunk["source_host_group"] = groups.astype(str)
            chunk["source_host_id"] = chunk["IPV4_SRC_ADDR"].map(host_id)
            chunk["source_file"] = Path(args.input).name
            for name, mask in (("train", groups.eq("train") | groups.eq("benign_0") | groups.eq("benign_1") | groups.eq("benign_2") | groups.eq("benign_3") | groups.eq("benign_4") | groups.eq("benign_5") | groups.eq("benign_6")), ("calibration", groups.eq("calibration") | groups.eq("benign_7")), ("test", groups.eq("test") | groups.eq("benign_8") | groups.eq("benign_9"))):
                part = chunk.loc[mask, FEATURES + ["Label", "label_binary", "source_file", "source_host_group", "source_host_id"]].copy()
                if part.empty: continue
                for col in FEATURES:
                    part[col] = pd.to_numeric(part[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
                if name not in writers:
                    writers[name] = (output / f"{name}.csv").open("w", encoding="utf-8", newline="")
                    part.to_csv(writers[name], index=False)
                else:
                    part.to_csv(writers[name], index=False, header=False)
                counts[name] += len(part)
                for label, count in part["Label"].value_counts().items():
                    label_counts[name][label] = label_counts[name].get(label, 0) + int(count)
    finally:
        for handle in writers.values(): handle.close()
    summary = {name: {"rows": counts[name], "attack_rate": sum(v for k,v in label_counts[name].items() if k != "BENIGN") / max(counts[name], 1), "labels": label_counts[name]} for name in counts}
    manifest = {"policy": "source-host-held-out by SHA-256 source-address bucket; train buckets 0-7, calibration 8, test 9", "features": FEATURES, "excluded": "source/destination identifiers, all IN/OUT fields, DST_TO_SRC fields, server-side fields, cumulative TCP flags, labels", "summary": summary, "directionality_caveat": "The raw source includes both directions. This experiment excludes reverse-direction features before modeling but cannot prove one-way collection provenance."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(output), "summary": summary}, indent=2))

if __name__ == "__main__": main()
