"""Process-local MessagePack codec for the AirSim RPC client.

AirSim's Python client depends on the legacy ``msgpack-rpc-python`` stack, whose pure-Python MessagePack makes image
RPCs slow (several seconds per step). Inside ``compiled_rpc_transport()`` the client encodes and decodes with the
compiled ``msgpack`` 1.1.2 extension instead, with the same wire format. The modern package is installed in its own
directory (see ``scripts/setup_env.sh``), so that the environment's legacy ``msgpack-python`` that ``airsim`` was built
against stays installed; ``aerorsi`` puts that directory first on ``sys.path`` at startup.
"""
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

COMPILED_MSGPACK = Path(sys.prefix) / "share" / "aerorsi_rpc"  # where scripts/setup_env.sh installs msgpack 1.1.2


def prefer_compiled_msgpack():
    """Put the compiled msgpack first on sys.path; call before anything imports msgpack (spawned workers inherit it)."""
    if COMPILED_MSGPACK.is_dir() and str(COMPILED_MSGPACK) not in sys.path:
        sys.path.insert(0, str(COMPILED_MSGPACK))


@contextmanager
def compiled_rpc_transport():
    import msgpack
    import msgpackrpc.transport.tcp as transport

    if msgpack.version != (1, 1, 2) or msgpack.Unpacker.__module__ != "msgpack._cmsgpack":
        raise RuntimeError(f"Navigation needs the compiled msgpack 1.1.2 in {COMPILED_MSGPACK} (see scripts/setup_env.sh)")

    class Packer:
        def __init__(self, *, encoding, default):
            if encoding != "utf-8":
                raise ValueError("Unexpected request encoding")
            self.packer = msgpack.Packer(use_bin_type=False, default=default)

        def pack(self, value):
            return self.packer.pack(value)

    class Unpacker:
        def __init__(self, *, encoding):
            if encoding not in ("utf-8", None):
                raise ValueError("Unexpected response encoding")
            limit = 2147483647
            self.unpacker = msgpack.Unpacker(raw=encoding is None, strict_map_key=False, max_buffer_size=limit,
                                             max_str_len=limit, max_bin_len=limit, max_array_len=limit,
                                             max_map_len=limit, max_ext_len=limit)

        def feed(self, wire):
            self.unpacker.feed(wire)

        def __iter__(self):
            yield from self.unpacker

    original = transport.msgpack
    transport.msgpack = SimpleNamespace(Packer=Packer, Unpacker=Unpacker)
    try:
        yield
    finally:
        transport.msgpack = original
