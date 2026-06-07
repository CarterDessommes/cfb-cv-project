import sys
import types

import numpy as np

from src.team_classifier import TeamClassifier


class CountingEmbed:
    """Small _embed stand-in that counts crops and uses mean brightness."""

    def __init__(self):
        self.calls = 0
        self.crops_embedded = 0

    def __call__(self, crops):
        self.calls += 1
        self.crops_embedded += len(crops)
        return np.array([[float(np.mean(crop))] for crop in crops], dtype=np.float32)


def make_frame() -> np.ndarray:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[0:20, 30:50] = 255
    frame[30:50, 60:80] = 255
    return frame


def make_classifier(monkeypatch):
    clf = TeamClassifier.__new__(TeamClassifier)
    clf._centroids = np.array([[0.0], [255.0]], dtype=np.float32)
    clf._track_clusters = {}
    clf._offense_cluster = 0
    clf._ball_votes = []
    clf._VOTE_WIN = 30
    counter = CountingEmbed()
    monkeypatch.setattr(clf, "_embed", counter)
    return clf, counter


def install_fake_clustering(monkeypatch):
    class FakeUMAP:
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, embeddings):
            return embeddings

    class FakeKMeans:
        def __init__(self, *args, **kwargs):
            self.labels_ = None

        def fit(self, embeddings):
            self.labels_ = (embeddings[:, 0] > 100).astype(int)
            return self

    umap = types.ModuleType("umap")
    umap.UMAP = FakeUMAP
    sklearn = types.ModuleType("sklearn")
    sklearn_cluster = types.ModuleType("sklearn.cluster")
    sklearn_cluster.KMeans = FakeKMeans
    sklearn.cluster = sklearn_cluster

    monkeypatch.setitem(sys.modules, "umap", umap)
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", sklearn_cluster)
