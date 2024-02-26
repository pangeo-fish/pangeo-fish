import re

scheduler_address_re = re.compile(r"tcp://\[?[0-9a-f:.]+\]?:[0-9]+")
scheme_re = re.compile(r"(?P<type>[^:]+):(?P<data>.+)")


def create_cluster(spec, **kwargs):
    """create a cluster from a spec and return a client

    Parameters
    ----------
    spec : str
        The method of creating a cluster:

        - "local": local cluster
        - the scheduler address of a running cluster: `tcp://<ip>:<port>`
        - `dask-jobqueue:<path-to-spec-file>`
        - `dask-hpcconfig:<name>`
    **kwargs
        Additional keyword arguments passed to the cluster
    """
    if spec == "local":
        from distributed import LocalCluster

        cluster = LocalCluster(**kwargs)

        return cluster.get_client()
    elif scheduler_address_re.match(spec) is not None:
        from distributed import Client

        client = Client(spec, **kwargs)

        return client

    match = scheme_re.match(spec)
    if match is None:
        raise ValueError(f"invalid scheme format: {spec!r}")

    groups = match.groupdict()
    if groups["type"] == "dask-jobqueue":
        raise NotImplementedError("does not work yet")
    elif groups["type"] == "dask-hpcconfig":
        import dask_hpcconfig

        cluster = dask_hpcconfig.cluster(groups["data"], **kwargs)

        return cluster.get_client()

    raise ValueError("unknown cluster type")
