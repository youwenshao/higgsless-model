import random

import numpy as np
from PIL import Image

from higgsless.analysis.cluster import cluster
from higgsless.worker_tasks.analyze import people_from_masks


def test_cluster_groups_similar_vectors():
    # Python's random (numpy.random's DLL can be blocked by Windows Application Control policies).
    rng = random.Random(0)

    def noisy(base: list[float]) -> list[float]:
        return [x + 0.05 * rng.gauss(0, 1) for x in base]

    a = [rng.gauss(0, 1) for _ in range(16)]
    b = [rng.gauss(0, 1) for _ in range(16)]
    vecs = np.array([noisy(a) for _ in range(5)] + [noisy(b) for _ in range(3)])
    labels = cluster(vecs, threshold=0.8)
    assert labels[:5] == [0] * 5 and labels[5:] == [1] * 3


def test_people_from_colored_masks(tmp_path):
    paths = []
    for i in range(4):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[20:90, 10 + i : 50 + i] = (255, 0, 0)  # left person
        if i >= 1:
            frame[30:95, 120:170] = (0, 255, 0)  # right person enters on frame 1
        path = tmp_path / f"m_{i:04d}.png"
        Image.fromarray(frame).save(path)
        paths.append(path)
    people = people_from_masks(paths)
    assert [p["track"] for p in people] == ["1", "2"]
    assert people[0]["first_frame"] == 0 and people[1]["first_frame"] == 1
    assert people[0]["presence"] == 1.0 and people[1]["presence"] == 0.75
    assert 0.6 < people[0]["height_frac"] <= 0.71
