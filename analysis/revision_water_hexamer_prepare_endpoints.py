#!/usr/bin/env python3
"""Prepare the preregistered water-hexamer H-bond graph and six DIRS endpoints.

This script uses geometry only. It must be run before any DIRS response metric is
computed for the JPCL R1 water-hexamer scalability extension.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

HBOND_MAX_HO = 2.50
HBOND_MIN_ANGLE = 140.0


def read_xyz(path: Path):
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    n = int(lines[0])
    atom_lines = lines[2:2+n]
    atoms, xyz = [], []
    for line in atom_lines:
        p = line.split()
        atoms.append(p[0])
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
    if len(atoms) != n:
        raise ValueError(f"XYZ atom count mismatch: header={n}, parsed={len(atoms)}")
    return atoms, np.asarray(xyz, dtype=float)


def distance(xyz, i, j):
    return float(np.linalg.norm(xyz[i] - xyz[j]))


def angle_deg(xyz, i, j, k):
    # angle i-j-k, vertex at j
    a = xyz[i] - xyz[j]
    b = xyz[k] - xyz[j]
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def shortest_path(adj, src, dst):
    if src == dst:
        return 0
    q = deque([(src, 0)])
    seen = {src}
    while q:
        u, d = q.popleft()
        for v in adj.get(u, ()):
            if v == dst:
                return d + 1
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    atoms, xyz = read_xyz(args.xyz)
    oxygens = [i for i, a in enumerate(atoms) if a.upper() == "O"]
    hydrogens = [i for i, a in enumerate(atoms) if a.upper() == "H"]
    if len(oxygens) != 6 or len(hydrogens) != 12:
        raise ValueError("Expected H12O6 atom composition")

    # Covalent assignment: each H belongs to its nearest O, exactly as preregistered.
    donor_o_for_h = {}
    waters = {o: [] for o in oxygens}
    for h in hydrogens:
        o = min(oxygens, key=lambda oi: distance(xyz, h, oi))
        donor_o_for_h[h] = o
        waters[o].append(h)

    # Stable water labels are defined by O atom index order.
    water_label = {o: k + 1 for k, o in enumerate(sorted(oxygens))}

    contacts = []
    for h in hydrogens:
        od = donor_o_for_h[h]
        candidates = []
        for oa in oxygens:
            if oa == od:
                continue
            dho = distance(xyz, h, oa)
            if dho > HBOND_MAX_HO:
                continue
            ang = angle_deg(xyz, od, h, oa)
            if ang < HBOND_MIN_ANGLE:
                continue
            candidates.append((dho, oa, ang))
        if candidates:
            # If one H has multiple acceptors, choose shortest H...O, then lowest atom index.
            dho, oa, ang = sorted(candidates, key=lambda x: (x[0], x[1]))[0]
            contacts.append({
                "donor_O": od + 1,
                "donor_H": h + 1,
                "acceptor_O": oa + 1,
                "donor_water": water_label[od],
                "acceptor_water": water_label[oa],
                "H_acceptor_distance_A": dho,
                "O_H_acceptor_angle_deg": ang,
                "O_O_distance_A": distance(xyz, od, oa),
            })

    contacts.sort(key=lambda c: (c["donor_O"], c["donor_H"], c["acceptor_O"]))
    for i, c in enumerate(contacts, 1):
        c["contact_id"] = f"HB{i}"

    if len(contacts) < 2:
        raise ValueError("Fewer than two H-bond contacts after deterministic graph construction")

    # Undirected water graph for topology/distance tests.
    adj = {w: set() for w in range(1, 7)}
    for c in contacts:
        u, v = c["donor_water"], c["acceptor_water"]
        adj[u].add(v); adj[v].add(u)

    endpoints = []

    # E1-E2: first two H bonds in deterministic order: H...O vs donor-O...acceptor-O.
    for ei, c in enumerate(contacts[:2], 1):
        endpoints.append({
            "endpoint_id": f"E{ei}",
            "class": "local_contact_heavy_atom",
            "coord_A": {"type": "distance", "atoms": [c["donor_H"], c["acceptor_O"]]},
            "coord_B": {"type": "distance", "atoms": [c["donor_O"], c["acceptor_O"]]},
            "contacts": [c["contact_id"]],
        })

    # E3-E4: first two contact pairs sharing at least one water molecule.
    adjacent = []
    for i in range(len(contacts)):
        wi = {contacts[i]["donor_water"], contacts[i]["acceptor_water"]}
        for j in range(i + 1, len(contacts)):
            wj = {contacts[j]["donor_water"], contacts[j]["acceptor_water"]}
            if wi & wj:
                adjacent.append((i, j))
    for k, (i, j) in enumerate(adjacent[:2], 3):
        ci, cj = contacts[i], contacts[j]
        endpoints.append({
            "endpoint_id": f"E{k}",
            "class": "adjacent_network_contacts",
            "coord_A": {"type": "distance", "atoms": [ci["donor_H"], ci["acceptor_O"]]},
            "coord_B": {"type": "distance", "atoms": [cj["donor_H"], cj["acceptor_O"]]},
            "contacts": [ci["contact_id"], cj["contact_id"]],
        })

    # E5-E6: first two contact pairs sharing no water, with minimum water-graph
    # separation between their edge endpoints >= 1. This is equivalent to a line-graph
    # edge distance >= 2 for distinct contacts in this simple graph.
    nonlocal_pairs = []
    for i in range(len(contacts)):
        wi = {contacts[i]["donor_water"], contacts[i]["acceptor_water"]}
        for j in range(i + 1, len(contacts)):
            wj = {contacts[j]["donor_water"], contacts[j]["acceptor_water"]}
            if wi & wj:
                continue
            pair_dists = [shortest_path(adj, a, b) for a in wi for b in wj]
            pair_dists = [d for d in pair_dists if d is not None]
            if not pair_dists:
                continue
            edge_distance = min(pair_dists) + 1
            if edge_distance >= 2:
                nonlocal_pairs.append((i, j, edge_distance))
    for k, (i, j, edge_distance) in enumerate(nonlocal_pairs[:2], 5):
        ci, cj = contacts[i], contacts[j]
        endpoints.append({
            "endpoint_id": f"E{k}",
            "class": "nonlocal_network_contacts",
            "line_graph_edge_distance": edge_distance,
            "coord_A": {"type": "distance", "atoms": [ci["donor_H"], ci["acceptor_O"]]},
            "coord_B": {"type": "distance", "atoms": [cj["donor_H"], cj["acceptor_O"]]},
            "contacts": [ci["contact_id"], cj["contact_id"]],
        })

    out = {
        "source_xyz": str(args.xyz),
        "thresholds": {"H_acceptor_max_A": HBOND_MAX_HO, "O_H_acceptor_min_angle_deg": HBOND_MIN_ANGLE},
        "water_assignment": {f"W{water_label[o]}": {"O": o + 1, "H": [h + 1 for h in sorted(waters[o])]} for o in sorted(oxygens)},
        "contacts": contacts,
        "endpoints": endpoints,
        "endpoint_class_counts": {
            "local_contact_heavy_atom": sum(e["class"] == "local_contact_heavy_atom" for e in endpoints),
            "adjacent_network_contacts": sum(e["class"] == "adjacent_network_contacts" for e in endpoints),
            "nonlocal_network_contacts": sum(e["class"] == "nonlocal_network_contacts" for e in endpoints),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")
    print(json.dumps(out["endpoint_class_counts"], indent=2))
    print(f"contacts={len(contacts)} endpoints={len(endpoints)} output={args.out}")


if __name__ == "__main__":
    main()
