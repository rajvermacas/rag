import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning

import app.main as main_module


def test_openrouter_embedding_import_suppresses_validate_default_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnsupportedFieldAttributeWarning)
        embedding_model = main_module._build_openrouter_embedding_model(
            api_key="test-openrouter-key",
            embed_model="text-embedding-3-small",
        )
    assert embedding_model is not None
