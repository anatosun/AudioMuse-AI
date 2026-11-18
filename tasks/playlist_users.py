# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Personalise generated playlists for one media-server user.

The functions run inside ``tasks.mediaserver.playlist_user_scope``, so the media server reports the play counts of the bound user.

Main Features:
* ``top_song_ids`` returns the catalogue ids of the user's most-played songs.
* ``playlist_centroids`` and ``playlists_for_user`` rank the clustering playlists of a run by the cosine similarity between each playlist's mean embedding and the mean embedding of the user's most-played songs.
"""

import logging

import numpy as np

import config
from database import get_tracks_by_ids

from .mediaserver import context as ms_context, get_top_played_songs
from .mediaserver.registry import canonical_input_ids

logger = logging.getLogger(__name__)


def top_song_ids(limit=None):
    """Catalogue ids of the bound user's most-played songs, most played first."""
    limit = limit or config.SONIC_FINGERPRINT_TOP_N_SONGS
    try:
        songs = get_top_played_songs(limit=limit) or []
    except Exception:
        logger.exception("Could not read the most-played songs of the user")
        return []
    provider_ids = [str(song['Id']) for song in songs if song.get('Id')]
    if not provider_ids:
        return []
    canonical = canonical_input_ids(provider_ids, ms_context.active_server_id())
    return list(dict.fromkeys(canonical.get(pid, pid) for pid in provider_ids))


def _mean_unit_vector(item_ids):
    vectors = [
        row['embedding_vector']
        for row in get_tracks_by_ids(list(item_ids))
        if row.get('embedding_vector') is not None and row['embedding_vector'].size
    ]
    if not vectors or len({v.shape for v in vectors}) != 1:
        return None
    mean = np.mean(np.stack(vectors), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else None


def playlist_centroids(playlists):
    """``{name: unit mean embedding}`` for ``{name: [(item_id, title, author), ...]}``."""
    centroids = {}
    for name, songs in playlists.items():
        centroid = _mean_unit_vector(item_id for item_id, _title, _author in songs)
        if centroid is not None:
            centroids[name] = centroid
    return centroids


def playlists_for_user(playlists, centroids, limit=None):
    """The ``limit`` playlists closest to the bound user's most-played songs.

    A user without play counts, or whose songs have no embeddings, gets every playlist.
    """
    limit = limit or config.PER_USER_CLUSTERING_PLAYLISTS
    taste = _mean_unit_vector(top_song_ids()) if centroids else None
    if taste is None:
        return dict(playlists)
    ranked = sorted(
        (name for name in playlists if name in centroids and centroids[name].shape == taste.shape),
        key=lambda name: float(np.dot(centroids[name], taste)),
        reverse=True,
    )
    return {name: playlists[name] for name in ranked[:limit]}
