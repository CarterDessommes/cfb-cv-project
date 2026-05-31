import numpy as np

from helpers import install_fake_clustering, make_classifier, make_frame


def test_fit_caches_track_clusters(monkeypatch):
    install_fake_clustering(monkeypatch)
    clf, counter = make_classifier(monkeypatch)
    frame = make_frame()
    boxes = [
        [0, 0, 20, 20, 10, 0],
        [30, 0, 50, 20, 20, 0],
        [0, 30, 20, 50, 11, 0],
        [60, 30, 80, 50, 21, 0],
    ]

    assert clf.fit(frame, boxes) is True

    assert counter.crops_embedded == 4
    assert clf._track_clusters == {10: 0, 20: 1, 11: 0, 21: 1}


def test_classify_embeds_only_uncached_tracks(monkeypatch):
    clf, counter = make_classifier(monkeypatch)
    frame = make_frame()
    boxes = [
        [0, 0, 20, 20, 1, 0],
        [30, 0, 50, 20, 2, 0],
    ]

    assert clf.classify(frame, boxes) == ["offense", "defense"]
    assert counter.crops_embedded == 2

    assert clf.classify(frame, boxes) == ["offense", "defense"]
    assert counter.crops_embedded == 2

    labels = clf.classify(frame, boxes + [[60, 30, 80, 50, 3, 0]])
    assert labels == ["offense", "defense", "defense"]
    assert counter.crops_embedded == 3


def test_label_flip_reuses_cached_embeddings(monkeypatch):
    clf, counter = make_classifier(monkeypatch)
    frame = make_frame()
    boxes = [
        [0, 0, 20, 20, 1, 0],
        [30, 0, 50, 20, 2, 0],
    ]
    labels = clf.classify(frame, boxes)
    embeds_after_classify = counter.crops_embedded

    changed = clf.update_offense_from_ball(np.array([40.0, 10.0]), boxes, labels)
    labels_after_flip = clf.classify(frame, boxes)

    assert changed is True
    assert labels_after_flip == ["defense", "offense"]
    assert counter.crops_embedded == embeds_after_classify


def test_invalid_small_crops_remain_unknown(monkeypatch):
    clf, counter = make_classifier(monkeypatch)
    frame = make_frame()

    assert clf.classify(frame, [[0, 0, 5, 5, 1, 0]]) == ["unknown"]
    assert counter.crops_embedded == 0
