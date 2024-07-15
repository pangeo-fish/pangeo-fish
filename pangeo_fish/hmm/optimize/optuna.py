import logging
import threading
import time
import warnings
from contextlib import contextmanager

from distributed import Client, LocalCluster

from pangeo_fish.hmm.optimize.logging import setup_logging

clients = {}
logger = setup_logging(logging.getLogger(__name__))


def get_client():
    thread_id = threading.get_ident()

    try:
        return clients[thread_id]
    except KeyError:
        pass

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="distributed",
            message=".*Port 8787 is already in use.",
        )

        cluster = LocalCluster(n_workers=1, memory_limit="2GB")
        logger.info(f"opened cluster dashboard at: {cluster.dashboard_link}")
    client = Client(cluster, set_as_default=False)

    clients[thread_id] = client

    return client


@contextmanager
def isolated_clients():
    global clients

    backup = clients

    try:
        clients = {}
        yield
    finally:
        for thread_id, client in clients.items():
            # make sure we don't cancel anything
            while [_ for _ in client.processing().values() if _]:
                time.sleep(2)
            client.shutdown()
            client.close()

        clients = backup
