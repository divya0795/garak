# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest

from garak.langproviders.remote import GoogleTranslator


@pytest.mark.parametrize(
    "project_id, expected_kwargs",
    [
        (None, {}),
        ("garak-project", {"project": "garak-project"}),
    ],
)
def test_google_translator_service_account_auth(
    tmp_path, monkeypatch, project_id, expected_kwargs
):
    from google.cloud import translate_v2 as translate

    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(GoogleTranslator.ENV_VAR, str(credentials))
    config_root = {
        "langproviders": {
            "remote.GoogleTranslator": {"language": "en,fr", "project_id": project_id}
        }
    }

    with mock.patch.object(
        translate.Client, "from_service_account_json", return_value=mock.MagicMock()
    ) as from_json:
        GoogleTranslator(config_root=config_root)

    assert from_json.call_args_list == [
        mock.call(str(credentials), **expected_kwargs)
    ], "credentials file should be loaded, scoped to project only when project_id is set"
