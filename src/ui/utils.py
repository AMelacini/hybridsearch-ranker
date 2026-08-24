import os
from functools import wraps
from time import sleep
from typing import Any

import requests

from ui.logger import get_custom_logger

logger = get_custom_logger(logger_name="hsr-frontend")

NUM_RETRY: int = 5


class BackendConnectionException(Exception):
    def __init__(self, message: str):
        self.message = message


def retry(call: Any) -> Any:
    @wraps(call)
    def _retry(*args: Any, **kwargs: dict[str, Any]) -> Any:
        for i in range(NUM_RETRY):
            try:
                return call(*args, **kwargs)
            except requests.exceptions.ConnectionError as e:
                msg = f"Failed to connect to HSR backend - attempt {i + 1} of {NUM_RETRY}"
                logger.warning(msg)
                sleep(pow(2, i))
                msg = str(e)
            except Exception as e:
                msg = str(e)
                logger.error(msg)
                raise BackendConnectionException(message=msg)
        msg = f"Unrecoverable failure to connect to HSR backend after {NUM_RETRY} attempts: {msg}"
        logger.error(msg)
        raise BackendConnectionException(message=msg)

    return _retry


def get_server_url() -> str:
    server_protocol = os.environ.get("BACKEND_PROTOCOL", "http")
    server_domain = os.environ.get("BACKEND_HOST", "localhost")
    server_port = int(os.environ.get("HSR_REST_SERVER_PORT", os.environ.get("REST_SERVER_PORT", "8000")))

    return f"{server_protocol}://{server_domain}:{server_port}"


@retry
def make_query(server_endpoint: str, payload: dict[str, float | int | str]) -> requests.Response:
    r = requests.post(server_endpoint, json=payload)
    return r


@retry
def make_handshake() -> requests.Response:
    server_url = get_server_url()
    server_endpoint = f"{server_url}/v1/admin/probe"
    r = requests.get(server_endpoint)
    return r
