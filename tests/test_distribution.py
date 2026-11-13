"""Keep public examples and package branding aligned with the runnable API."""

import json
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from litjev.api import create_app
from litjev.data.schema import SystemOneRequest

ROOT = Path(__file__).resolve().parents[1]


def test_readme_request_is_a_valid_schema():
    request = SystemOneRequest.model_validate_json((ROOT / "examples/request.json").read_text())
    schema = request.to_schema()
    # The file the playground loads and the file the README points at are the
    # same file: they drifted once, and a reader who pasted one got a request
    # the other's assertions did not describe.
    assert schema.names == ("tool_selection", "intent", "review_trap", "service_credit")
    assert len(schema["tool_selection"].choices) == 5
    client = TestClient(create_app(lambda: None))
    assert client.get("/example").json() == request.model_dump()


def test_public_package_metadata_and_bundled_frontend():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "litjev"
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE", "THIRD_PARTY_LICENSES/*.txt"]
    assert project["scripts"]["litjev"] == "litjev.cli:serve"
    client = TestClient(create_app(lambda: None))
    assert "LitJev" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert json.loads(client.get("/health").text)["model_loaded"] is False


def test_attribution_and_citation_are_present():
    assert "Version 2.0, January 2004" in (ROOT / "LICENSE").read_text()
    assert "Copyright 2026 ZhengxuYu" in (ROOT / "NOTICE").read_text()
    citation = (ROOT / "CITATION.cff").read_text()
    assert "license: Apache-2.0" in citation
    assert 'version: "0.1.0"' in citation
    readme = (ROOT / "README.md").read_text()
    assert readme.rfind("## Citation") > readme.rfind("## License")
