from httpx import Client, codes

from src.app import PATHS


def test_health_endpoint(client: Client) -> None:
    resp = client.get(url=PATHS.health_check)
    assert resp.status_code == codes.OK
