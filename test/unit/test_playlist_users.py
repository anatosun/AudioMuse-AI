# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Per-user generated playlists.

Main Features:
* the registry keeps the chosen users of a server, and the server API lists a server's users and saves the choice
* Jellyfin, Emby and Navidrome report their users only when the configured account is an admin
* ``playlist_users_for`` turns per-user playlists on only for an admin account, and ``playlist_user_scope`` binds one user's credentials
* Navidrome playlists made for a user are private, given to the user, and found by owner
* clustering, the Sonic Fingerprint and the Album of the Week create one set per chosen user
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tasks.mediaserver import context

ALICE = {'id': 'u-alice', 'name': 'alice'}
BOB = {'id': 'u-bob', 'name': 'bob'}


def _jellyfin_server(playlist_users=None):
    return {
        'server_id': 's1',
        'name': 'Home',
        'server_type': 'jellyfin',
        'creds': {'url': 'http://jf', 'token': 'tok', 'user_id': 'u-admin'},
        'music_libraries': '',
        'is_default': True,
        'playlist_users': playlist_users or [],
    }


class TestRegistry:
    def test_normalize_row_keeps_valid_users(self):
        from tasks.mediaserver.registry import normalize_row

        row = dict(_jellyfin_server(), playlist_users=[{'id': 1, 'name': 'alice'}, {'name': 'no id'}, 'junk', {'id': 'u2'}])

        assert normalize_row(row)['playlist_users'] == [{'id': '1', 'name': 'alice'}, {'id': 'u2', 'name': 'u2'}]

    def test_normalize_row_without_the_column(self):
        from tasks.mediaserver.registry import normalize_row

        row = _jellyfin_server()
        del row['playlist_users']

        assert normalize_row(row)['playlist_users'] == []


class TestJellyfinUsers:
    _USERS = [
        {'Id': 'u-admin', 'Name': 'admin', 'Policy': {'IsAdministrator': True}},
        {'Id': 'u-alice', 'Name': 'alice', 'Policy': {'IsAdministrator': False}},
    ]

    def test_admin_found_by_id(self):
        from tasks.mediaserver.helper import playlist_users_from_jellyfin_users

        info = playlist_users_from_jellyfin_users(self._USERS, 'u-admin')

        assert info['admin'] is True
        assert info['listening'] is True
        assert info['users'] == [{'id': 'u-admin', 'name': 'admin'}, {'id': 'u-alice', 'name': 'alice'}]

    def test_admin_found_by_name(self):
        from tasks.mediaserver.helper import playlist_users_from_jellyfin_users

        assert playlist_users_from_jellyfin_users(self._USERS, 'ADMIN')['admin'] is True

    def test_non_admin_gets_no_users(self):
        from tasks.mediaserver.helper import playlist_users_from_jellyfin_users

        info = playlist_users_from_jellyfin_users(self._USERS, 'u-alice')

        assert info['admin'] is False
        assert info['users'] == []

    def test_unreadable_user_list_is_not_admin(self):
        from tasks.mediaserver import jellyfin

        with context.use_server(_jellyfin_server()), patch.object(jellyfin, '_jellyfin_get_users', return_value=None):
            info = jellyfin.list_playlist_users()

        assert info['admin'] is False

    def test_user_creds_replace_the_user_id(self):
        from tasks.mediaserver import emby, jellyfin

        assert jellyfin.playlist_user_creds(ALICE) == {'user_id': 'u-alice'}
        assert emby.playlist_user_creds(ALICE) == {'user_id': 'u-alice'}


class TestDispatcher:
    def _info(self, **overrides):
        info = {'supported': True, 'admin': True, 'listening': True, 'users': [ALICE, BOB]}
        info.update(overrides)
        return info

    def test_no_chosen_users_is_shared(self):
        from tasks import mediaserver

        with patch.object(mediaserver, 'list_playlist_users') as listing:
            assert mediaserver.playlist_users_for(_jellyfin_server()) == ([], False)
        listing.assert_not_called()

    def test_chosen_users_on_an_admin_account(self):
        from tasks import mediaserver

        server = _jellyfin_server([ALICE, {'id': 'u-gone', 'name': 'gone'}])
        with patch.object(mediaserver, 'list_playlist_users', return_value=self._info()):
            assert mediaserver.playlist_users_for(server) == ([ALICE], True)

    def test_non_admin_account_is_shared(self):
        from tasks import mediaserver

        with patch.object(mediaserver, 'list_playlist_users', return_value=self._info(admin=False, users=[])):
            assert mediaserver.playlist_users_for(_jellyfin_server([ALICE])) == ([], False)

    def test_unsupported_server_is_shared(self):
        from tasks import mediaserver

        with patch.object(mediaserver, 'list_playlist_users', return_value=self._info(supported=False)):
            assert mediaserver.playlist_users_for(_jellyfin_server([ALICE])) == ([], False)

    def test_failing_user_list_is_shared(self):
        from tasks import mediaserver

        with patch.object(mediaserver, 'list_playlist_users', side_effect=RuntimeError('down')):
            assert mediaserver.playlist_users_for(_jellyfin_server([ALICE])) == ([], False)

    @patch('tasks.mediaserver.config')
    def test_plex_and_lyrion_are_unsupported(self, mock_config):
        from tasks import mediaserver

        for server_type in ('plex', 'lyrion'):
            mock_config.MEDIASERVER_TYPE = server_type
            assert mediaserver.list_playlist_users()['supported'] is False

    def test_user_scope_binds_the_user_and_restores_the_server(self):
        from tasks import mediaserver

        server = _jellyfin_server()
        with context.use_server(server):
            with mediaserver.playlist_user_scope(ALICE):
                creds = context.active_creds()
                assert creds['user_id'] == 'u-alice'
                assert creds['token'] == 'tok'
                assert context.active_server_id() == 's1'
            assert context.active_creds()['user_id'] == 'u-admin'

    @patch('tasks.mediaserver.config')
    @patch('tasks.mediaserver.navidrome.get_all_playlists')
    @patch('tasks.mediaserver.navidrome.delete_playlist', return_value=True)
    def test_cleanup_deletes_only_the_active_owners_playlists(self, mock_delete, mock_get, mock_config):
        from tasks import mediaserver

        mock_config.MEDIASERVER_TYPE = 'navidrome'
        marked = mediaserver.AUTOMATIC_PLAYLIST_MARKER
        mock_get.return_value = [
            {'Id': '1', 'Name': 'Mix', 'Description': marked, 'Owner': 'alice'},
            {'Id': '2', 'Name': 'Mix', 'Description': marked, 'Owner': 'bob'},
            {'Id': '3', 'Name': 'Mix', 'Description': marked, 'Owner': None},
        ]
        server = {'server_id': 's1', 'server_type': 'navidrome', 'creds': {'url': 'http://nd', 'user': 'admin', 'password': 'pw'}}

        with context.use_server(server), mediaserver.playlist_user_scope(ALICE):
            mediaserver.delete_automatic_playlists()

        assert sorted(call.args[0] for call in mock_delete.call_args_list) == ['1', '3']


def _navidrome_server():
    return {'server_id': 'n1', 'server_type': 'navidrome', 'creds': {'url': 'http://nd', 'user': 'admin', 'password': 'pw'}}


def _response(body):
    response = MagicMock()
    response.json.return_value = body
    return response


@pytest.fixture(autouse=True)
def _clear_navidrome_tokens():
    from tasks.mediaserver import navidrome

    navidrome._NATIVE_TOKENS.clear()
    yield
    navidrome._NATIVE_TOKENS.clear()


class TestNavidrome:
    @patch('tasks.mediaserver.navidrome.requests')
    def test_admin_lists_users_without_listening(self, mock_requests):
        from tasks.mediaserver import navidrome

        mock_requests.post.return_value = _response({'token': 't', 'isAdmin': True})
        mock_requests.get.return_value = _response([{'id': 'u-alice', 'userName': 'alice', 'name': 'Alice'}])

        with context.use_server(_navidrome_server()):
            info = navidrome.list_playlist_users()
            navidrome.list_playlist_users()

        assert info == {'supported': True, 'admin': True, 'listening': False, 'users': [ALICE]}
        assert mock_requests.post.call_count == 1
        assert mock_requests.get.call_args.kwargs['headers'] == {'X-ND-Authorization': 'Bearer t'}

    @patch('tasks.mediaserver.navidrome.requests')
    def test_non_admin_lists_no_users(self, mock_requests):
        from tasks.mediaserver import navidrome

        mock_requests.post.return_value = _response({'token': 't', 'isAdmin': False})

        with context.use_server(_navidrome_server()):
            assert navidrome.list_playlist_users()['admin'] is False
        mock_requests.get.assert_not_called()

    @patch('tasks.mediaserver.navidrome.requests')
    def test_api_key_cannot_log_in(self, mock_requests):
        from tasks.mediaserver import navidrome

        server = {'server_id': 'n1', 'server_type': 'navidrome', 'creds': {'url': 'http://nd', 'api_key': 'k'}}
        with context.use_server(server):
            info = navidrome.list_playlist_users()

        assert info['admin'] is False
        assert 'API key' in info['error']
        mock_requests.post.assert_not_called()

    @patch('tasks.mediaserver.navidrome.requests')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_user_playlist_is_private_and_given_to_the_user(self, mock_request, mock_requests):
        from tasks import mediaserver
        from tasks.mediaserver import navidrome

        mock_request.side_effect = [{'status': 'ok', 'playlist': {'id': 'p1', 'name': 'Mix'}}, {'status': 'ok'}]
        mock_requests.post.return_value = _response({'token': 't', 'isAdmin': True})

        with context.use_server(_navidrome_server()), mediaserver.playlist_user_scope(ALICE):
            result = navidrome.create_playlist('Mix', ['t1'], description='marker')

        assert result['Id'] == 'p1'
        assert mock_request.call_args_list[1].args[1] == {'playlistId': 'p1', 'comment': 'marker'}
        put = mock_requests.put.call_args
        assert put.args[0] == 'http://nd/api/playlist/p1'
        assert put.kwargs['json'] == {'id': 'p1', 'ownerId': 'u-alice'}

    @patch('tasks.mediaserver.navidrome.requests')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_playlist_that_cannot_be_given_is_deleted(self, mock_request, mock_requests):
        from tasks import mediaserver
        from tasks.mediaserver import navidrome

        mock_request.side_effect = [{'status': 'ok', 'playlist': {'id': 'p1', 'name': 'Mix'}}, {'status': 'ok'}, {'status': 'ok'}]
        mock_requests.post.return_value = _response({'token': 't', 'isAdmin': True})
        mock_requests.put.return_value.raise_for_status.side_effect = RuntimeError('403')

        with context.use_server(_navidrome_server()), mediaserver.playlist_user_scope(ALICE):
            assert navidrome.create_playlist('Mix', ['t1']) is None

        assert mock_request.call_args_list[2].args[:2] == ('deletePlaylist', {'id': 'p1'})

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_shared_playlist_stays_public(self, mock_request):
        from tasks.mediaserver import navidrome

        mock_request.side_effect = [{'status': 'ok', 'playlist': {'id': 'p1', 'name': 'Mix'}}, {'status': 'ok'}]

        with context.use_server(_navidrome_server()):
            navidrome.create_playlist('Mix', ['t1'])

        assert mock_request.call_args_list[1].args[1] == {'playlistId': 'p1', 'public': 'true'}

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_playlist_by_name_is_the_owners(self, mock_request):
        from tasks import mediaserver
        from tasks.mediaserver import navidrome

        mock_request.return_value = {'playlists': {'playlist': [
            {'id': 'p-bob', 'name': 'Sonic Fingerprint', 'owner': 'bob'},
            {'id': 'p-alice', 'name': 'Sonic Fingerprint', 'owner': 'alice'},
            {'id': 'p-admin', 'name': 'Sonic Fingerprint', 'owner': 'admin'},
        ]}}

        with context.use_server(_navidrome_server()):
            assert navidrome.get_playlist_by_name('Sonic Fingerprint')['id'] == 'p-admin'
            with mediaserver.playlist_user_scope(ALICE):
                assert navidrome.get_playlist_by_name('Sonic Fingerprint')['id'] == 'p-alice'


class TestRanking:
    _PLAYLISTS = {
        'Calm': [('c1', 'T', 'A'), ('c2', 'T', 'A')],
        'Loud': [('l1', 'T', 'A')],
        'Odd': [('o1', 'T', 'A')],
    }
    _VECTORS = {
        'c1': [1.0, 0.0], 'c2': [0.9, 0.1], 'l1': [0.0, 1.0], 'o1': [-1.0, 0.0],
        'fav1': [0.1, 1.0], 'fav2': [0.0, 0.9],
    }

    def _tracks(self, ids):
        return [{'item_id': i, 'embedding_vector': np.array(self._VECTORS[i], dtype=np.float32)} for i in ids if i in self._VECTORS]

    def test_closest_playlists_first(self):
        from tasks import playlist_users

        with patch.object(playlist_users, 'get_tracks_by_ids', side_effect=self._tracks), \
                patch.object(playlist_users, 'top_song_ids', return_value=['fav1', 'fav2']):
            centroids = playlist_users.playlist_centroids(self._PLAYLISTS)
            picked = playlist_users.playlists_for_user(self._PLAYLISTS, centroids, limit=2)

        assert list(picked) == ['Loud', 'Calm']

    def test_user_without_plays_gets_every_playlist(self):
        from tasks import playlist_users

        with patch.object(playlist_users, 'get_tracks_by_ids', side_effect=self._tracks), \
                patch.object(playlist_users, 'top_song_ids', return_value=[]):
            centroids = playlist_users.playlist_centroids(self._PLAYLISTS)
            picked = playlist_users.playlists_for_user(self._PLAYLISTS, centroids, limit=1)

        assert picked == self._PLAYLISTS

    def test_top_song_ids_are_canonical(self):
        from tasks import playlist_users

        with patch.object(playlist_users, 'get_top_played_songs', return_value=[{'Id': 'p1'}, {'Id': 'p2'}, {'Id': 'p1'}]), \
                patch.object(playlist_users, 'canonical_input_ids', return_value={'p1': 'c1'}):
            assert playlist_users.top_song_ids() == ['c1', 'p2']


class TestClusteringPublish:
    _PLAYLISTS = {'Calm': [('c1', 'T', 'A')], 'Loud': [('l1', 'T', 'A')]}

    @contextmanager
    def _scope(self, user, scopes):
        scopes.append(user['name'])
        yield

    def _run(self, users, listening):
        from tasks import clustering

        created, cleanups, scopes = [], [], []
        with patch.object(clustering, 'get_playlist_names', return_value=['Calm']), \
                patch.object(clustering, 'delete_automatic_playlists', side_effect=lambda names: cleanups.append(list(scopes))), \
                patch.object(clustering, 'playlist_users_for', return_value=(users, listening)), \
                patch.object(clustering, 'playlist_user_scope', side_effect=lambda user: self._scope(user, scopes)), \
                patch.object(clustering, 'playlist_centroids', return_value={'Loud': 'v'}), \
                patch.object(clustering, 'playlists_for_user', side_effect=lambda playlists, centroids: {'Loud': playlists['Loud']}), \
                patch.object(clustering, 'create_playlist', side_effect=lambda name, ids, description: created.append((list(scopes), name))):
            clustering._publish_playlists(self._PLAYLISTS, {'server_id': 's1'}, 'Home')
        return created, cleanups

    def test_without_users_one_shared_set(self):
        created, cleanups = self._run([], False)

        assert created == [([], 'Calm'), ([], 'Loud')]
        assert cleanups == [[]]

    def test_each_user_gets_the_closest_playlists(self):
        created, cleanups = self._run([ALICE, BOB], True)

        assert created == [(['alice'], 'Loud'), (['alice', 'bob'], 'Loud')]
        assert cleanups == [[], ['alice'], ['alice', 'bob']]

    def test_without_listening_each_user_gets_every_playlist(self):
        created, _cleanups = self._run([ALICE], False)

        assert [name for _scopes, name in created] == ['Calm', 'Loud']


class TestScheduledPlaylists:
    def _run(self, users, listening, needs_listening):
        from tasks import task_run

        @contextmanager
        def fake_beat(*args, **kwargs):
            yield

        @contextmanager
        def fake_guard(*args, **kwargs):
            yield lambda force=False: None

        scopes, upserts, personal = [], [], []

        @contextmanager
        def fake_scope(user):
            scopes.append(user['name'])
            yield
            scopes.pop()

        def build_ids(is_personal):
            personal.append(is_personal)
            return ['a']

        with patch('tasks.recovery.row_heartbeat', fake_beat), \
                patch('tasks.task_run.cancel_guard', fake_guard), \
                patch('tasks.task_run.save_task_status', return_value=True), \
                patch('tasks.task_run.taskqueue.current_task_id', return_value=None), \
                patch('tasks.mediaserver.registry.servers_for_scope', return_value=[None]), \
                patch('tasks.mediaserver.playlist_users_for', return_value=(users, listening)), \
                patch('tasks.mediaserver.playlist_user_scope', fake_scope), \
                patch('tasks.mediaserver.create_or_replace_playlist', side_effect=lambda name, ids: upserts.append(list(scopes)) or {'Id': 'p'}):
            summary = task_run.run_playlist_task_per_server(
                'album_of_the_week', 'album of the week', 'Album', 'Album of the Week',
                build_ids, 'all', 'inline-1', needs_listening=needs_listening,
            )
        return summary, upserts, personal

    def test_each_user_gets_a_personal_playlist(self):
        summary, upserts, personal = self._run([ALICE, BOB], True, False)

        assert upserts == [['alice'], ['bob']]
        assert personal == [True, True]
        assert summary['playlists_created'] == 2

    def test_users_without_listening_get_a_random_album(self):
        _summary, upserts, personal = self._run([ALICE], False, False)

        assert upserts == [['alice']]
        assert personal == [False]

    def test_fingerprint_stays_shared_without_listening(self):
        _summary, upserts, personal = self._run([ALICE], False, True)

        assert upserts == [[]]
        assert personal == [False]

    def test_album_of_the_week_seeds_from_the_users_favourites(self):
        from tasks import album_creation_manager as acm

        tried = []

        def fake_album(seed_type, item_id=None):
            tried.append(item_id)
            if item_id == 'fav':
                return {'seed': {'label': 'fav'}, 'tracks': []}
            raise acm.AlbumSeedError('no')

        with patch.object(acm, 'weekly_seed_ids', return_value=['r1', 'fav']), \
                patch('tasks.playlist_users.top_song_ids', return_value=['fav']), \
                patch.object(acm, 'create_album', side_effect=fake_album):
            acm.create_album_of_the_week(personal=True)

        assert tried == ['fav']


class TestServerApi:
    @pytest.fixture
    def client(self, monkeypatch):
        from flask import Flask

        import app_music_servers as msrv

        monkeypatch.setattr(msrv, '_forbid_non_admin', lambda: None)
        monkeypatch.setattr(msrv.registry, 'bind', lambda server, conn=None: context.use_server(server))
        app = Flask('playlist-users-test')
        app.register_blueprint(msrv.music_servers_bp)
        return app.test_client(), msrv

    def test_users_endpoint_lists_users_and_the_choice(self, client, monkeypatch):
        test_client, msrv = client
        server = _jellyfin_server([ALICE])
        monkeypatch.setattr(msrv.registry, 'get_server', lambda server_id, conn=None: server)
        monkeypatch.setattr(msrv, 'list_playlist_users', lambda: {'supported': True, 'admin': True, 'listening': True, 'users': [ALICE, BOB]})

        body = test_client.get('/api/servers/s1/users').get_json()

        assert body['users'] == [ALICE, BOB]
        assert body['selected'] == [ALICE]

    def test_users_endpoint_reports_an_unreachable_server(self, client, monkeypatch):
        test_client, msrv = client
        monkeypatch.setattr(msrv.registry, 'get_server', lambda server_id, conn=None: _jellyfin_server())

        def fail():
            raise RuntimeError('down')

        monkeypatch.setattr(msrv, 'list_playlist_users', fail)

        body = test_client.get('/api/servers/s1/users').get_json()

        assert body['admin'] is False
        assert body['error']

    def test_unknown_server(self, client, monkeypatch):
        test_client, msrv = client
        monkeypatch.setattr(msrv.registry, 'get_server', lambda server_id, conn=None: None)

        assert test_client.get('/api/servers/nope/users').status_code == 404

    def test_saving_users_does_not_restart_the_workers(self, client, monkeypatch):
        test_client, msrv = client
        server = _jellyfin_server()
        saved = {}
        monkeypatch.setattr(msrv.registry, 'get_server', lambda server_id, conn=None: server)
        monkeypatch.setattr(msrv.registry, 'get_default_server_id', lambda conn=None: 's1')
        monkeypatch.setattr(msrv, 'get_db', lambda: None)
        monkeypatch.setattr(msrv.registry, 'update_server', lambda server_id, **kw: saved.update(kw))
        restart = MagicMock(return_value=True)
        monkeypatch.setattr(msrv, '_apply_default_to_config', restart)

        response = test_client.put('/api/servers/s1', json={'playlist_users': [ALICE]})

        assert response.status_code == 200
        assert saved['playlist_users'] == [ALICE]
        restart.assert_not_called()

    def test_playlist_users_must_be_a_list(self, client, monkeypatch):
        test_client, msrv = client
        monkeypatch.setattr(msrv.registry, 'get_server', lambda server_id, conn=None: _jellyfin_server())

        assert test_client.put('/api/servers/s1', json={'playlist_users': 'alice'}).status_code == 400
