from pathlib import Path

import pytest

from genre_classifier.data import Track, assert_split_integrity, split_tracks


def test_split_is_reproducible_and_disjoint():
    tracks = [Track(Path(f"{genre}/{i}.wav"), genre, f"{genre}:{i}")
              for genre in ("jazz", "rock") for i in range(20)]
    a = split_tracks(tracks, seed=7)
    b = split_tracks(tracks, seed=7)
    assert {k: [x.track_id for x in v] for k, v in a.items()} == {
        k: [x.track_id for x in v] for k, v in b.items()}
    assert sum(map(len, a.values())) == len(tracks)
    assert_split_integrity(a)
    assert all({t.label for t in rows} == {"jazz", "rock"} for rows in a.values())


def test_integrity_rejects_duplicate_track():
    track = Track(Path("a.wav"), "jazz", "jazz:1")
    with pytest.raises(ValueError, match="occurs in both"):
        assert_split_integrity({"train": [track], "test": [track]})
