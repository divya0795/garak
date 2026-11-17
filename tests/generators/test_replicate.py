# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest

from garak.attempt import Conversation, Message, Turn
from garak.generators.replicate import InferenceEndpoint


@pytest.fixture
def replicate_env(monkeypatch):
    monkeypatch.setenv(InferenceEndpoint.ENV_VAR, "r8_fake_token")


@pytest.mark.usefixtures("replicate_env")
def test_inference_endpoint_joins_streamed_output():
    generator = InferenceEndpoint("garak/test-deployment")
    prediction = mock.MagicMock()
    prediction.output = ["Hello", " world"]
    client = mock.MagicMock()
    client.deployments.get.return_value.predictions.create.return_value = prediction
    generator.client = client

    conv = Conversation([Turn("user", Message("Say hello"))])
    output = generator.generate(conv, generations_this_call=1)

    assert len(output) == 1, "one generation requested, so one Message expected"
    assert (
        output[0].text == "Hello world"
    ), "streamed tokens should be joined into one Message"
