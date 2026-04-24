"""
生成每个快照中新实体的距离、节点度、旧图支撑和综合难度。
"""

import os
from collections import defaultdict, deque


def _list_datasets(base_path):
    dataset_names = []
    for dataset_name in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        if os.path.isdir(os.path.join(dataset_path, "0")):
            dataset_names.append(dataset_name)
    return sorted(dataset_names)


def _load_entity2id(snapshot_path):
    entity2id = {}
    entity2id_path = os.path.join(snapshot_path, "entity2id.txt")
    with open(entity2id_path, "r", encoding="utf-8") as rf:
        for line in rf:
            line_split = line.strip().split()
            if len(line_split) < 2:
                continue
            ent, ent_id = line_split[0], line_split[1]
            entity2id[ent] = int(ent_id)
    return entity2id


def _load_train_triples(snapshot_path, entity2id):
    triples = []
    train_path = os.path.join(snapshot_path, "train.txt")
    with open(train_path, "r", encoding="utf-8") as rf:
        for line in rf:
            line_split = line.strip().split("\t")
            if len(line_split) < 3:
                continue
            head, relation, tail = line_split[0], line_split[1], line_split[2]
            triples.append((entity2id[head], relation, entity2id[tail]))
    return triples


def _compute_entity_metrics(triples, seen_entities):
    adjacency = defaultdict(set)
    degree = defaultdict(float)
    old_support = defaultdict(float)
    relation_sets = defaultdict(set)

    for h, relation, t in triples:
        adjacency[h].add(t)
        adjacency[t].add(h)
        degree[h] += 1.0
        degree[t] += 1.0
        if h in seen_entities and t not in seen_entities:
            old_support[t] += 1.0
        elif t in seen_entities and h not in seen_entities:
            old_support[h] += 1.0
        if h not in seen_entities:
            relation_sets[h].add(relation)
        if t not in seen_entities:
            relation_sets[t].add(relation)

    seed_nodes = [eid for eid in seen_entities if eid in adjacency]
    visited = set(seed_nodes)
    bfs_queue = deque((eid, 0) for eid in seed_nodes)
    distance = defaultdict(lambda: 100.0)

    while bfs_queue:
        node, dist = bfs_queue.popleft()
        for nxt in adjacency[node]:
            if nxt in visited:
                continue
            visited.add(nxt)
            next_dist = dist + 1
            distance[nxt] = float(next_dist)
            bfs_queue.append((nxt, next_dist))

    all_new_entities = sorted({h for h, _, _ in triples if h not in seen_entities} | {t for _, _, t in triples if t not in seen_entities})
    max_distance = max([distance[eid] for eid in all_new_entities] + [1.0])
    max_degree = max([degree[eid] for eid in all_new_entities] + [1.0])
    max_old_support = max([old_support[eid] for eid in all_new_entities] + [1.0])
    max_new_relation = max([float(len(relation_sets[eid])) for eid in all_new_entities] + [1.0])

    metrics = {}
    for entity in all_new_entities:
        dist_norm = distance[entity] / max_distance
        degree_norm = degree[entity] / max_degree
        old_support_norm = old_support[entity] / max_old_support
        new_relation_value = float(len(relation_sets[entity]))
        new_relation_norm = new_relation_value / max_new_relation
        difficulty = (
            dist_norm
            + 0.5 * (1.0 - degree_norm)
            + 0.75 * (1.0 - old_support_norm)
            + 0.35 * new_relation_norm
        )
        metrics[entity] = {
            "distance": distance[entity],
            "degree": degree[entity],
            "old_support": old_support[entity],
            "new_relation": new_relation_value,
            "difficulty": difficulty,
        }
    return metrics


def nodes_sort():
    base_path = "./data/"
    for dataset_name in _list_datasets(base_path):
        seen_entities = set()
        dataset_path = os.path.join(base_path, dataset_name)
        snapshot_dirs = sorted(
            [item for item in os.listdir(dataset_path) if item.isdigit()],
            key=lambda x: int(x),
        )
        for snapshot_name in snapshot_dirs:
            snapshot = int(snapshot_name)
            snapshot_path = os.path.join(dataset_path, snapshot_name)
            entity2id = _load_entity2id(snapshot_path)
            triples = _load_train_triples(snapshot_path, entity2id)
            if snapshot == 0:
                for head, _, tail in triples:
                    seen_entities.add(head)
                    seen_entities.add(tail)
                continue

            metrics = _compute_entity_metrics(triples, seen_entities)
            distance_output_path = os.path.join(snapshot_path, "train_distance_nodes.txt")
            difficulty_output_path = os.path.join(snapshot_path, "train_difficulty_nodes.txt")

            ordered_by_distance = sorted(
                metrics.items(),
                key=lambda kv: (kv[1]["distance"], kv[1]["degree"], kv[0]),
            )
            with open(distance_output_path, "w", encoding="utf-8") as wf:
                for key, value in ordered_by_distance:
                    wf.write(f"{key}\t{value['distance']}\t{value['degree']}\n")

            ordered_by_difficulty = sorted(
                metrics.items(),
                key=lambda kv: (kv[1]["difficulty"], kv[1]["distance"], kv[0]),
            )
            with open(difficulty_output_path, "w", encoding="utf-8") as wf:
                for key, value in ordered_by_difficulty:
                    wf.write(
                        f"{key}\t{value['distance']}\t{value['degree']}\t"
                        f"{value['old_support']}\t{value['new_relation']}\t{value['difficulty']}\n"
                    )

            for head, _, tail in triples:
                seen_entities.add(head)
                seen_entities.add(tail)
    print("ok")


if __name__ == "__main__":
    nodes_sort()