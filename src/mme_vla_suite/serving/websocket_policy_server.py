import asyncio
import http
import logging
import time
import traceback

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from mme_vla_suite.policies.policy import MME_VLA_Policy
from mme_vla_suite.serving.batched_robottt_policy import BatchedRoboTTTPolicy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: MME_VLA_Policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        parallel_clients: int = 1,
        inference_batch_size: int = 2,
        batch_wait_ms: float = 50,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._active_connection = False
        self._batched_policy = None
        self._queue = None
        self._model_lock = None
        self._batch_wait = batch_wait_ms / 1000
        if parallel_clients > 1:
            self._batched_policy = BatchedRoboTTTPolicy(
                policy, parallel_clients, inference_batch_size)
            self._queue = asyncio.Queue()
            self._model_lock = asyncio.Lock()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        if self._batched_policy is not None:
            asyncio.create_task(self._batch_loop())
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        if self._batched_policy is not None:
            await self._batched_handler(websocket)
            return
        if self._active_connection:
            await websocket.close(code=1013, reason="RoboTTT state already has a client")
            return
        self._active_connection = self._policy._uses_robottt
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))
        
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                
                if obs.get("reset", False):
                    tstart = time.monotonic()
                    self._policy.reset(obs.get("robottt_mode", "normal"))
                    tend = time.monotonic() - tstart
                    await websocket.send(packer.pack(
                        {"reset_finished": True, "reset_time_ms": tend * 1000}))
                elif obs.get("add_buffer", False):
                    tstart = time.monotonic()
                    self._policy.add_buffer(obs)
                    tend = time.monotonic() - tstart
                    await websocket.send(packer.pack(
                        {"add_buffer_finished": True, "add_buffer_time_ms": tend * 1000}))
                else:
                    outputs = self._policy.infer(obs)
                    await websocket.send(packer.pack(outputs))

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                self._active_connection = False
                break
            except Exception:
                self._active_connection = False
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _batch_loop(self):
        while True:
            requests = [await self._queue.get()]
            await asyncio.sleep(self._batch_wait)
            while len(requests) < self._batched_policy.inference_batch_size:
                try:
                    requests.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            async with self._model_lock:
                outputs = await asyncio.to_thread(self._batched_policy.infer, requests)
            for future, output in outputs:
                future.set_result(output)

    async def _batched_handler(self, websocket: _server.ServerConnection):
        try:
            slot = self._batched_policy.allocate()
        except RuntimeError as error:
            await websocket.close(code=1013, reason=str(error))
            return
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        try:
            while True:
                observation = msgpack_numpy.unpackb(await websocket.recv())
                if observation.get("reset", False):
                    async with self._model_lock:
                        self._batched_policy.reset(
                            slot, observation.get("robottt_mode", "normal"))
                    output = {"reset_finished": True, "reset_time_ms": 0}
                elif observation.get("add_buffer", False):
                    started = time.monotonic()
                    async with self._model_lock:
                        await asyncio.to_thread(
                            self._batched_policy.add_buffer, slot, observation)
                    output = {
                        "add_buffer_finished": True,
                        "add_buffer_time_ms": (time.monotonic() - started) * 1000,
                    }
                else:
                    future = asyncio.get_running_loop().create_future()
                    await self._queue.put((slot, observation, future))
                    output = await future
                await websocket.send(packer.pack(output))
        except websockets.ConnectionClosed:
            pass
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error. Traceback included in previous frame.",
            )
            raise
        finally:
            self._batched_policy.release(slot)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
