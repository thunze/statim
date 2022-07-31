"""Extraction of ISO image files available on a local file system or via HTTP."""

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from contextlib import ExitStack, closing, contextmanager
from io import SEEK_CUR, SEEK_END, SEEK_SET, IOBase, UnsupportedOperation
from multiprocessing import Value
from pathlib import Path, PurePosixPath
from queue import Empty, Full, Queue
from shutil import disk_usage
from threading import Event
from typing import (
    TYPE_CHECKING,
    BinaryIO,
    Callable,
    Generator,
    Iterator,
    NamedTuple,
    Optional,
    Union,
)

import requests
from pycdlib import PyCdlib
from pycdlib.facade import PyCdlibISO9660, PyCdlibJoliet, PyCdlibRockRidge, PyCdlibUDF
from pycdlib.pycdlibexception import PyCdlibInvalidInput
from requests.adapters import HTTPAdapter, Retry

from .plan import LocalSource, RemoteSource

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

__all__ = ['extract', 'ExtractProgress', 'tps_default', 'tps_win10_uefi']


log = logging.getLogger(__name__)


HTTP_TIMEOUT = 10.0  # seconds
HTTP_RETRY_STRATEGY = Retry(
    total=5, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)
)

_KIB = 1024
_MIB = 1024 * 1024
CHUNK_SIZE_LOCAL = 4 * _MIB
CHUNK_SIZE_REMOTE = 1 * _MIB  # plain HttpIO
CHUNK_SIZE_REMOTE_CHUNKED = 4 * _KIB  # HttpIO with chunked transfer
TARGET_PATHS_EXTRA_FREE_SPACE = 1 * _MIB  # safety buffer for additional files

MAX_WORKERS_LOCAL = 1
MAX_WORKERS_REMOTE = 16
EXTRACT_QUEUE_TIMEOUT = 0.05  # seconds
EXC_QUEUE_TIMEOUT = 0.05  # seconds
PAUSE_TIMEOUT = 0.05  # seconds

PROGRESS_UPDATE_MIN_GAP = 0.5  # min seconds between extraction progress updates
PROGRESS_UPDATE_MAX_GAP = 5.0  # max seconds between extraction progress updates


class HttpIO(IOBase, BinaryIO):
    """Access a file available via HTTP like a file-like object.

    Supported operations are seeking and reading. This is accomplished by utilizing
    the HTTP `Range` header.

    Raises an ``OSError`` if the server doesn't support range requests.
    """

    def __init__(self, url: str):
        super().__init__()

        self._url = url
        self._session = requests.Session()

        # retry strategy
        adapter = HTTPAdapter(
            max_retries=HTTP_RETRY_STRATEGY,
            pool_maxsize=MAX_WORKERS_REMOTE,
            pool_block=True,
        )
        # noinspection HttpUrlsUsage
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)

        # always request an unencoded response
        self._session.headers.update({'accept-encoding': 'identity'})

        head = self._session.head(url, timeout=HTTP_TIMEOUT)
        head.raise_for_status()
        headers = head.headers

        accept_ranges = headers.get('accept-ranges', 'none')
        if accept_ranges.lower() != 'bytes':
            raise OSError('Server does not support range requests')

        # server could ignore the Accept-Encoding header
        # see https://datatracker.ietf.org/doc/html/rfc7231#section-5.3.4
        if headers.get('content-encoding', ''):
            raise OSError('Range requests are not supported for encoded content')

        # https://datatracker.ietf.org/doc/html/rfc7230#section-3.3.2
        if 'content-length' not in headers:
            raise OSError("Server does not provide a 'Content-Length' header")

        self._length = int(headers['content-length'])
        self._pos = 0

        # chunked transfer: tuple of (chunks, chunk size)
        self._chunked_transfer: Optional[tuple[Iterator[bytes], int]] = None

    def __repr__(self) -> str:
        """Return a printable representation of the object."""
        return f'{self.__class__.__name__}(url={self._url!r})'

    def _check_closed(self) -> None:
        """Raise ``ValueError`` if the file is closed."""
        if self.closed:  # skipcq: PYL-W0125
            raise ValueError('I/O operation on closed file')

    def _check_chunked_transfer(self) -> None:
        """Raise ``ValueError`` if a chunked transfer is currently active."""
        if self._chunked_transfer is not None:
            raise ValueError('Random-access operation during chunked transfer')

    def __enter__(self) -> 'HttpIO':
        """Context management protocol.

        Returns ``self`` -- an instance of ``HttpIO``.
        """
        self._check_closed()
        return self

    # The following three properties / methods are only implemented to satisfy the
    # typing.BinaryIO protocol.

    @property
    def mode(self) -> str:
        """File mode indicator, always 'rb'."""
        return 'rb'

    @property
    def name(self) -> str:
        """URL the instance reads from."""
        return self._url

    def write(self, b: bytes) -> int:
        """Write the given buffer to the IO stream. Unsupported."""
        raise UnsupportedOperation(f'{self.__class__.__name__}.write() not supported')

    def readable(self) -> bool:
        """Return a ``bool`` indicating whether the object supports reading.

        True for all ``HttpIO`` instances.
        """
        return True

    def seekable(self) -> bool:
        """Return a ``bool`` indicating whether the object supports random access.

        True if no chunked transfer is active.
        """
        return self._chunked_transfer is None

    def _range_request(
        self, first_byte_pos: int, last_byte_pos: int, stream: bool = False
    ) -> requests.Response:
        """Request the byte range from ``first_byte_pos`` to ``last_byte_pos`` (both
        inclusive) and return the resulting ``requests.Response`` object.

        If the ``stream`` flag is set, the content is not downloaded immediately.
        """
        headers = {'range': f'bytes={first_byte_pos}-{last_byte_pos}'}

        response = self._session.get(
            self._url, headers=headers, timeout=HTTP_TIMEOUT, stream=stream
        )
        response.raise_for_status()

        if response.status_code != 206 or 'content-range' not in response.headers:
            raise OSError('Server did not send a partial response')
        return response

    @contextmanager
    def chunked_transfer(self, chunk_size: int) -> Iterator[Iterator[bytes]]:
        """Start a chunked transfer.

        While operating in the context this context manager provides, reading is done
        using *chunked transfer encoding* instead of sending out separate range
        requests every time ``read`` is invoked. This can lead to faster read
        operations, but only allows data to be read in its chronological order in
        chunks of ``chunk_size`` bytes (or less if it's the last chunk).

        The transfer starts at the current stream position and carries on until EOF.
        The chunks can be retrieved either by iterating over the generator this
        context manager provides or by repeatedly invoking ``read(chunk_size)`` (see
        example). If ``read`` is invoked with an argument of less than ``chunk_size``
        (and greater than 0), the whole chunk is consumed, but only the requested part
        of it is returned.

        Note that per ``HttpIO`` object only one chunked transfer can be active at
        the same time.

        Example::

            with HttpIO('https://example.test/test_file') as http_io:
                http_io.seek(1337)  # where the chunked transfer is supposed to start
                with http_io.chunked_transfer(1024) as chunked_transfer:
                    for chunk in chunked_transfer:
                        print(chunk)
                    # or
                    while True:
                        chunk = http_io.read(1024)  # or less
                        if not chunk:
                            break
                # left context, so random access is possible again
                http_io.seek(0)
                http_io.read(2048)
        """
        self._check_closed()

        if self._chunked_transfer is not None:
            raise ValueError('Another chunked transfer is already active')
        if chunk_size <= 0:
            raise ValueError(f'Chunk size must be greater than 0, got {chunk_size}')

        def chunk_wrapper(chunks_: Iterator[bytes]) -> Iterator[bytes]:
            """Iterator wrapper to make the ``HttpIO`` object keep track of the
            current stream position during chunked transfer.
            """
            for chunk in chunks_:
                self._pos += len(chunk)  # last chunk might be smaller than chunk size
                yield chunk

        response: Optional[requests.Response] = None
        try:
            if self._pos >= self._length:
                chunks = iter(())  # empty iterator
            else:
                # request up to EOF
                response = self._range_request(self._pos, self._length - 1, stream=True)
                chunks = response.iter_content(chunk_size)

            self._chunked_transfer = (chunks, chunk_size)
            yield chunk_wrapper(chunks)

        finally:
            self._chunked_transfer = None
            if response is not None:
                response.close()

    def read(self, size: int = -1) -> bytes:
        """Read and return up to ``size`` bytes.

        Returns an empty ``bytes`` object on EOF.
        """
        self._check_closed()

        if size == 0 or self._pos >= self._length:
            return b''
        if size < 0 or self._pos + size > self._length:
            size = self._length - self._pos

        if self._chunked_transfer is not None:
            chunks, chunk_size = self._chunked_transfer

            if size > chunk_size:
                raise ValueError(
                    f'Can only read in chunks of {chunk_size} bytes or less during '
                    f'chunked transfer ({size} bytes requested)'
                )
            # consume whole chunk
            data = next(chunks)
            new_pos = self._pos + len(data)

            # but return less if requested
            if size < chunk_size:
                data = data[:size]
        else:
            response = self._range_request(self._pos, self._pos + size - 1)
            data = response.content
            new_pos = self._pos + len(data)

        actual_size = len(data)
        if actual_size != size:
            raise ConnectionError(
                f'Server did not send the requested amount of bytes (expected {size}, '
                f'got {actual_size})'
            )

        self._pos = new_pos
        return data

    def readall(self) -> bytes:
        """Read until EOF."""
        return self.read()

    def seek(self, offset: int, whence: int = 0) -> int:
        """Change the stream position.

        Change the stream position to byte offset ``offset``. Argument ``offset`` is
        interpreted relative to the position indicated by ``whence``. Values for
        ``whence`` are ints:

        - 0 -- Start of stream (the default); offset should be zero or positive
        - 1 -- Current stream position; offset may be negative
        - 2 -- End of stream; offset is usually negative

        Returns an ``int`` indicating the new absolute position.
        """
        self._check_closed()
        self._check_chunked_transfer()

        if whence == SEEK_SET:
            if offset < 0:
                raise ValueError(f'Negative seek position {offset}')
            self._pos = offset
        elif whence == SEEK_CUR:
            self._pos = max(0, self._pos + offset)
        elif whence == SEEK_END:
            self._pos = max(0, self._length + offset)
        else:
            raise ValueError('Unsupported whence value, must be one of (0, 1, 2)')
        return self._pos

    def tell(self) -> int:
        """Return an ``int`` indicating the current stream position."""
        self._check_closed()
        return self._pos

    def close(self) -> None:
        """Close the IO object.

        This method has no effect if the file is already closed.
        """
        if not self.closed:
            self._session.close()
        super().close()


# --- Target path strategies


def tps_default(filepath_iso: PurePosixPath, target_paths: tuple[Path, ...]) -> Path:
    """Default strategy to determine the correct target path for a specific file
    present in an ISO image file.

    Always chooses the first target path of the target paths passed.
    """
    if len(target_paths) < 1:
        raise ValueError(
            f'Strategy requires at least 1 target path, got {target_paths}'
        )
    if not filepath_iso.is_absolute():
        raise ValueError(f'filepath_iso must be an absolute path, got {filepath_iso}')
    return target_paths[0]


def tps_win10_uefi(filepath_iso: PurePosixPath, target_paths: tuple[Path, ...]) -> Path:
    """Strategy to determine the correct target path for a specific file present in a
    Windows 10 or newer ISO image file if this ISO image file is used to create a
    bootable USB drive for a UEFI system.

    Chooses the second target path (NTFS partition) for the `sources` directory except
    `sources/boot.wim`.
    Chooses the first target path for everything else.

    For more information on why we copy the installation files of Windows 10+ images
    to multiple partitions, refer to the documentation of the ``drive`` package.
    """
    if len(target_paths) != 2:
        raise ValueError(
            f'Strategy requires exactly 2 target paths, got {target_paths}'
        )
    if not filepath_iso.is_absolute():
        raise ValueError(f'filepath_iso must be an absolute path, got {filepath_iso}')

    parts = filepath_iso.parts[1:]  # strip the '/' part
    part_count = len(parts)

    if part_count == 0 or (parts[0] == 'sources' and part_count == 1):
        raise ValueError(f'Got invalid filepath_iso {filepath_iso} for Windows 10+')

    # actual logic
    if parts[0] == 'sources' and not parts[1] == 'boot.wim':
        return target_paths[1]
    return target_paths[0]


# --- Extraction

PyCdlibFacade = Union[PyCdlibISO9660, PyCdlibJoliet, PyCdlibRockRidge, PyCdlibUDF]


class OpenResult(NamedTuple):
    """File-like object and PyCdlib object passed back to the main thread after
    opening the ISO image file.
    """

    source_file: BinaryIO
    iso: PyCdlib


class ExtractJob(NamedTuple):
    """Information required for a worker thread to extract a specific file from an
    ISO image file to a local directory.

    filepath_iso: Absolute path of the file to extract from the image.
    filepath_local: Desired path of the file on a local file system.
    """

    filepath_iso: PurePosixPath
    filepath_local: Path


def _get_facade_for_iso(iso: PyCdlib) -> PyCdlibFacade:
    """Return the pycdlib facade of ``iso`` which matches the most preferable ISO
    extension supported by ``iso``.
    """
    if iso.has_udf():
        # Prioritize UDF if it's available because it's required for modern
        # Windows ISOs (>= Vista).
        return iso.get_udf_facade()
    if iso.has_joliet():
        return iso.get_joliet_facade()
    if iso.has_rock_ridge():
        return iso.get_rock_ridge_facade()
    return iso.get_iso9660_facade()


def _pause_or_quit(quit_event: Event, resume_event: Event) -> bool:
    """If ``quit_event`` isn't set, wait as long as ``resume_event`` isn't set.

    Returns whether ``quit_event`` is set.
    """
    if not quit_event.is_set():
        resume_event.wait()
    return quit_event.is_set()


def _extract_file(
    job: ExtractJob,
    source_file: BinaryIO,
    iso_facade: PyCdlibFacade,
    progress: 'Synchronized[int]',  # multiprocessing.Value
    quit_event: Event,
    resume_event: Event,
) -> bool:
    """Extract a specific file from an ISO image.

    :param job: ExtractJob object containing source and destination path.
    :param source_file: IO handle for the ISO image.
    :param iso_facade: PyCdlib facade to use to extract the file from the image.
    :param progress: Value indicating how many bytes were already extracted in total.
    :param quit_event: Event indicating that the worker is to be stopped.
    :param resume_event: Event indicating that the worker is not to be paused.
    :returns: True if the according extraction job can be marked as done.
    """
    filepath_iso, filepath_local = job
    log.debug(f'Extracting {filepath_iso} -> {filepath_local}')

    # create parent directories if necessary
    dirpath_local = filepath_local.parent
    dirpath_local.mkdir(parents=True, exist_ok=True)

    # symlink handling
    record = iso_facade.get_record(str(filepath_iso))
    if record.is_symlink():
        if isinstance(iso_facade, PyCdlibRockRidge):
            target_bytes = record.rock_ridge.symlink_path()  # type: ignore[union-attr]
            target = target_bytes.decode('utf-8')
            filepath_local.symlink_to(target)
        else:
            log.warning(f'Skipping non-Rock Ridge symlink at {filepath_iso}')
        return True

    try:
        file_iso = iso_facade.open_file_from_iso(str(filepath_iso))
    except PyCdlibInvalidInput as e:
        # An El Torito boot catalog is also represented by a data file in an ISO
        # image (e.g. '/isolinux/boot.cat'). If we try to open such a data file using
        # open_file_from_iso, an exception with the message 'File has no data' is
        # raised because this edge case is not handled. As such a file serves no
        # purpose in the file system itself, we can safely skip its extraction.
        if str(e) == 'File has no data':
            log.debug(f'Skipping suspected boot catalog at {filepath_iso}')
            skipped_record = iso_facade.get_record(str(filepath_iso))
            with progress.get_lock():
                progress.value += skipped_record.get_data_length()
            return True
        else:
            raise

    # contexts active for the current extraction job
    with ExitStack() as job_stack:
        # open_file_from_iso succeeded
        job_stack.enter_context(file_iso)
        file_local = job_stack.enter_context(filepath_local.open('wb'))
        file_bytes_total: int = file_iso.length()
        file_bytes_left: int = file_bytes_total

        if isinstance(source_file, HttpIO):
            # use chunked transfer for larger files
            if file_bytes_total > CHUNK_SIZE_REMOTE_CHUNKED:
                # noinspection PyTypeChecker
                job_stack.enter_context(
                    source_file.chunked_transfer(CHUNK_SIZE_REMOTE_CHUNKED)
                )
                chunk_size = CHUNK_SIZE_REMOTE_CHUNKED
            else:
                chunk_size = CHUNK_SIZE_REMOTE
        else:
            chunk_size = CHUNK_SIZE_LOCAL

        # read and write chunk by chunk
        while not _pause_or_quit(quit_event, resume_event) and file_bytes_left > 0:
            bytes_expected = min(chunk_size, file_bytes_left)
            chunk = file_iso.read(bytes_expected)
            bytes_read = len(chunk)
            bytes_written = file_local.write(chunk)

            if bytes_read != bytes_expected:  # pragma: no cover
                raise OSError(
                    f'Did not read the expected amount of bytes (expected '
                    f'{bytes_expected} bytes, read {bytes_read} bytes)'
                )
            if bytes_written != bytes_read:  # pragma: no cover
                raise OSError(
                    f'Did not write the expected amount of bytes (read '
                    f'{bytes_expected} bytes, wrote {bytes_read} bytes)'
                )
            file_bytes_left -= bytes_written

            # sync with other threads
            with progress.get_lock():
                progress.value += bytes_written

        return file_bytes_left == 0  # False if _pause_or_quit returned True

    assert False  # see https://github.com/python/mypy/issues/7726, skipcq: BAN-B101


def _extract_worker(
    source: LocalSource | RemoteSource,
    open_result_queue: Queue[OpenResult],
    extract_queue: Queue[ExtractJob],
    exc_queue: Queue[BaseException],
    progress: 'Synchronized[int]',  # multiprocessing.Value
    extract_queue_ready: Event,
    quit_event: Event,
    resume_event: Event,
) -> None:
    """Function executed by worker threads spawned to extract files from an ISO image.

    All workers first open an IO handle for the desired ISO image and use that to
    instantiate a new ``PyCdlib`` object. IO handle and ``PyCdlib`` object are then
    added to ``open_result_queue`` exactly once for the main thread to iterate over
    the files present in the ISO image. After that, every file path is added to
    ``extract_queue`` by the main thread, so that the spawned workers (described by
    this function) can execute these "extraction jobs" by extracting the according
    files.

    :param exc_queue: Queue of exceptions raised by workers.
    :param progress: Value indicating how many bytes were already extracted in total.
    :param extract_queue_ready: Event indicating that all extraction jobs were enqueued.
    :param quit_event: Event indicating that the worker is to be stopped.
    :param resume_event: Event indicating that the worker is not to be paused.
    """
    try:
        # contexts active for the lifetime of the worker
        with ExitStack() as worker_stack:
            source_file: BinaryIO
            if isinstance(source, LocalSource):
                source_file = source.path.open(mode='rb')
            else:
                source_file = HttpIO(source.url)
            # noinspection PyTypeChecker
            worker_stack.enter_context(source_file)

            # open ISO
            iso = PyCdlib()
            iso.open_fp(source_file)  # this might take a while
            worker_stack.enter_context(closing(iso))

            try:
                open_result = OpenResult(source_file, iso)
                open_result_queue.put(open_result, block=False)
            except Full:
                pass  # main thread only needs (source_file, iso) once

            iso_facade = _get_facade_for_iso(iso)

            while not _pause_or_quit(quit_event, resume_event):
                try:
                    job = extract_queue.get(timeout=EXTRACT_QUEUE_TIMEOUT)
                except Empty:
                    # wait until all jobs are enqueued
                    if not extract_queue_ready.is_set():
                        continue
                    break  # no jobs left

                task_done = _extract_file(
                    job, source_file, iso_facade, progress, quit_event, resume_event
                )
                if task_done:
                    extract_queue.task_done()

    except BaseException as e:
        exc_queue.put(e)


class ExtractProgress(NamedTuple):
    """Information about the extraction progress.

    bytes_total: How many bytes to extract in total.
    bytes_done: How many bytes were already extracted.
    seconds_delta: How many seconds passed since the last progress update.
    bytes_delta: How many bytes were extracted since the last progress update.
    """

    bytes_total: int
    bytes_done: int
    seconds_delta: float
    bytes_delta: int

    @property
    def done_ratio(self) -> float:
        """Ratio of already extracted bytes to bytes to extract in total."""
        if self.bytes_total <= 0:
            return 1.0
        return self.bytes_done / self.bytes_total

    @property
    def bytes_per_second(self) -> Optional[float]:
        """Average extraction speed since the last progress update in bytes per
        second.

        ``None`` if the extraction speed can't be calculated reliably.
        """
        if self.seconds_delta <= 0:
            return None  # quite unlikely to happen
        return self.bytes_delta / self.seconds_delta

    @property
    def seconds_left(self) -> Optional[float]:
        """Estimated time required for extracting the remaining data in seconds.

        ``None`` if the estimated time left can't be calculated reliably.
        """
        if self.bytes_per_second is None or self.bytes_per_second <= 0:
            return None
        return (self.bytes_total - self.bytes_done) / self.bytes_per_second


def iso_size_contents(iso_facade: PyCdlibFacade) -> int:
    """Return the size of the ISO image contents which are visible to ``iso_facade``
    in bytes.
    """
    size = 0
    for dirpath_iso, _, filelist in iso_facade.walk('/'):
        for filename in filelist:
            filepath_iso_abs = PurePosixPath(dirpath_iso) / filename
            # pycdlib doesn't like pathlib paths
            record = iso_facade.get_record(str(filepath_iso_abs))
            if not record.is_symlink():
                size += record.get_data_length()
    return size


def _enqueue_jobs(
    iso_facade: PyCdlibFacade,
    extract_queue: Queue[ExtractJob],
    target_paths: tuple[Path, ...],
    target_path_strategy: Callable[[PurePosixPath, tuple[Path, ...]], Path],
) -> None:
    """Iterate over every file visible to ``iso_facade`` and add according
    ``ExtractJob`` objects to ``extract_queue``.

    The local file paths planned for the files to extract are determined by
    ``target_path_strategy``.
    """
    for dirpath_iso, _, filelist in iso_facade.walk('/'):
        dirpath_iso = PurePosixPath(dirpath_iso)  # absolute path to current dir
        dirpath_iso_rel = dirpath_iso.relative_to('/')

        for filename in filelist:
            filepath_iso = dirpath_iso / filename
            rootpath_local = target_path_strategy(filepath_iso, target_paths)

            # strip version number and semicolon from ISO 9660 local file name
            if isinstance(iso_facade, PyCdlibISO9660):
                filename_local = filename.split(';')[0]
            else:
                filename_local = filename

            filepath_local = rootpath_local / dirpath_iso_rel / filename_local
            extract_queue.put(ExtractJob(filepath_iso, filepath_local))


def _report_progress(
    futures: list[Future[None]],
    size: int,
    exc_queue: Queue[BaseException],
    progress: 'Synchronized[int]',  # multiprocessing.Value
    resume_event: Event,
) -> Generator[Optional[ExtractProgress], Optional[bool], None]:
    """Wait for all workers described by ``futures`` to finish and regularly yield
    progress updates regarding the extraction in the form of ``ExtractProgress``
    objects.

    If exceptions raised in worker threads are encountered via ``exc_queue``, they
    are re-raised.

    :param size: Size of ISO image contents to extract in total.
    :param exc_queue: Queue of exceptions raised by workers.
    :param progress: Value indicating how many bytes were already extracted in total.
    :param resume_event: Event indicating that the workers are not to be paused.
    """
    bytes_delta_start = progress.value  # measure bytes
    seconds_delta_start = time.perf_counter()  # measure time

    # wait for all worker threads to finish
    while not all(future.done() for future in futures):
        try:
            exc = exc_queue.get(timeout=EXC_QUEUE_TIMEOUT)
        except Empty:
            pass
        else:
            raise RuntimeError('Exception raised in worker thread') from exc

        extract_progress = None
        seconds_delta_end = time.perf_counter()
        seconds_delta = seconds_delta_end - seconds_delta_start

        # send a progress update at most every PROGRESS_UPDATE_MIN_GAP seconds
        if seconds_delta >= PROGRESS_UPDATE_MIN_GAP:
            bytes_delta_end = progress.value
            bytes_delta = bytes_delta_end - bytes_delta_start

            # only send an actual ExtractProgress (and not None) if at least one
            # byte was extracted or if it's enforced by PROGRESS_UPDATE_MAX_GAP
            if bytes_delta > 0 or seconds_delta >= PROGRESS_UPDATE_MAX_GAP:
                extract_progress = ExtractProgress(
                    size, bytes_delta_end, seconds_delta, bytes_delta
                )
                seconds_delta_start = seconds_delta_end
                bytes_delta_start = bytes_delta_end

        should_pause = yield extract_progress
        if should_pause:
            resume_event.clear()
            pause_start = time.perf_counter()
            while (yield None):
                time.sleep(PAUSE_TIMEOUT)
            seconds_delta_start += time.perf_counter() - pause_start
            resume_event.set()

    if progress.value != size:  # pragma: no cover
        log.warning(
            f'Size of extracted files does not match calculated size '
            f'(expected {size} bytes, got {progress.value} bytes)'
        )

    # last progress update
    progress.value = size
    seconds_delta = time.perf_counter() - seconds_delta_start
    bytes_delta = size - bytes_delta_start
    yield ExtractProgress(size, size, seconds_delta, bytes_delta)


def _extract(
    source: LocalSource | RemoteSource,
    target_paths: Path | tuple[Path, ...],
    target_path_strategy: Callable[[PurePosixPath, tuple[Path, ...]], Path],
) -> Generator[Optional[ExtractProgress], Optional[bool], None]:
    """Extract the contents of an ISO image file to local directories.

    See ``extract`` for complete documentation.
    """
    time_start = time.perf_counter()
    log.info('Preparing ISO image extraction ...')
    log.debug(f'Source: {source}')
    log.debug(f'Target path(s): {target_paths}')

    if isinstance(target_paths, Path):
        target_paths = (target_paths,)
    target_paths = tuple(path.resolve() for path in target_paths)

    for path in target_paths:
        if not path.is_dir():
            raise ValueError(f'{path} is not a directory')

    # prepare threading
    if isinstance(source, LocalSource):
        max_workers = MAX_WORKERS_LOCAL
    else:
        max_workers = MAX_WORKERS_REMOTE

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        open_result_queue: Queue[OpenResult] = Queue(maxsize=1)
        extract_queue: Queue[ExtractJob] = Queue()
        exc_queue: Queue[BaseException] = Queue()
        progress: 'Synchronized[int]' = Value('q', 0)  # zero bytes processed yet
        extract_queue_ready = Event()
        quit_event = Event()
        resume_event = Event()
        resume_event.set()
        futures = []

        try:
            # open IO handles separately to avoid side effects in extraction workers
            for _ in range(max_workers):
                future = executor.submit(
                    _extract_worker,
                    source,
                    open_result_queue,
                    extract_queue,
                    exc_queue,
                    progress,
                    extract_queue_ready,
                    quit_event,
                    resume_event,
                )
                futures.append(future)

            # get first open IO handle and PyCdlib object
            while True:
                try:
                    source_file, iso = open_result_queue.get(block=False)
                    break
                except Empty:
                    # open_result_queue stays empty forever if an exception is raised
                    # in every worker thread before enqueuing a result
                    try:
                        exc = exc_queue.get(timeout=EXC_QUEUE_TIMEOUT)
                    except Empty:
                        continue
                    else:
                        raise RuntimeError('Exception raised in worker thread') from exc

            open_time = time.perf_counter() - time_start
            log.debug(f'Opened ISO after {open_time:.4f} seconds')

            # select ISO extension
            iso_facade = _get_facade_for_iso(iso)
            iso_format_str = iso_facade.__class__.__name__.strip('PyCdlib')
            log.debug(f'Selected ISO format {iso_format_str!r}')

            if isinstance(iso_facade, PyCdlibISO9660):
                log.warning('Fallback to pure ISO 9660 format')

            # size checks
            size = iso_size_contents(iso_facade)  # ISO contents
            size_container = source_file.seek(0, 2)  # ISO file
            source_file.seek(0, 0)
            log.debug(f'ISO size: {size_container} bytes')
            log.debug(f'ISO size (contents): {size} bytes')

            space_available = sum(disk_usage(path).free for path in target_paths)
            log.debug(
                f'Total space available at target directories: {space_available} bytes'
            )
            if space_available < size + TARGET_PATHS_EXTRA_FREE_SPACE:
                raise ValueError(
                    'Not enough disk space available at target directories'
                )

            # add extraction jobs to queue
            _enqueue_jobs(iso_facade, extract_queue, target_paths, target_path_strategy)

            # start extraction
            log.info('Extracting files from ISO image ...')
            time_extraction_start = time.perf_counter()
            extract_queue_ready.set()

            yield from _report_progress(
                futures, size, exc_queue, progress, resume_event
            )
            wait(futures)

        finally:
            # Any unhandled exception raised during extraction wouldn't take effect if
            # the worker threads stayed busy with jobs or with waiting for
            # extract_queue_ready.
            log.info('Waiting for all workers to quit ...')
            quit_event.set()
            resume_event.set()  # order is important here!
            # make sure we don't miss any exception or cancellation of a future
            for future in as_completed(futures):
                future.result()

    extract_queue.join()
    extraction_time = time.perf_counter() - time_extraction_start

    log.info(f'Finished ISO image extraction after {extraction_time:.4f} seconds')
    if extraction_time > 0:
        log.info(
            f'Average extraction speed: {size / extraction_time:.4f} bytes per second'
        )


@contextmanager
def extract(
    source: LocalSource | RemoteSource,
    target_paths: Path | tuple[Path, ...],
    target_path_strategy: Callable[
        [PurePosixPath, tuple[Path, ...]], Path
    ] = tps_default,
) -> Iterator[Generator[Optional[ExtractProgress], Optional[bool], None]]:
    """Extract the contents of an ISO image file to local directories.

    This is a context manager returning a generator on ``__enter__``. This generator
    regularly yields ``ExtractProgress`` objects indicating the current extraction
    progress. Calling its ``close()`` method as well leaving the context causes the
    extraction to be aborted (see example). To pause the extraction, repeatedly pass
    ``True`` instead of ``None`` to the generator using its ``send()`` method until
    resumption is desired.

    Args:
        source: Source of the image file described by a LocalSource or a RemoteSource.

        target_paths: Paths to local directories to extract the contents of the image
            file to. Usually only one target path is required.
            The length of this tuple must match the length of the tuple the second
            parameter of the callable target_path_strategy accepts.
            Instead of passing a tuple of length 1, a single Path can be passed.

        target_path_strategy: Callable which determines which file (first parameter,
            file path relative to ISO root) is extracted to which local directory
            specified in target_paths (second parameter).
            Returns one of the target paths defined in the second parameter.
            Functions which can be used as a value for this parameter are exported by
            the image module and are named tps_*.

    Example::

        with extract(source, Path('/path/to/destination')) as extraction:
            for progress in extraction:
                if progress is not None:
                    print(f'{(progress.done_ratio * 100):.2f} percent done')
                if wanna_abort:
                    break  # leaving the context
    """
    gen = _extract(source, target_paths, target_path_strategy)
    try:
        yield gen
    finally:
        # _extract needs to stop the threads it spawns if an exception is raised
        # during extraction
        gen.close()
