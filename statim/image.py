"""Extraction of ISO image files available on the local file system or via HTTP."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from io import SEEK_CUR, SEEK_END, SEEK_SET, IOBase, UnsupportedOperation
from pathlib import Path, PurePosixPath
from queue import Empty, Full, Queue
from shutil import disk_usage
from threading import Event
from typing import BinaryIO, Callable, NamedTuple

import requests
from pycdlib import PyCdlib
from requests.adapters import HTTPAdapter, Retry

from .plan import LocalSource, RemoteSource

__all__ = ['extract', 'tps_default', 'tps_win10_uefi']


log = logging.getLogger(__name__)


HTTP_TIMEOUT = 10.0  # seconds
HTTP_RETRY_STRATEGY = Retry(
    total=5, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)  # retries
)

_MIB = 1024 * 1024
BLOCK_SIZE_LOCAL = 4 * _MIB
BLOCK_SIZE_REMOTE = 4 * _MIB
TARGET_PATHS_EXTRA_FREE_SPACE = 1 * _MIB  # safety buffer for additional files

MAX_WORKERS_LOCAL = 1
MAX_WORKERS_REMOTE = 16
EXTRACT_QUEUE_TIMEOUT = 1  # seconds
EXC_QUEUE_TIMEOUT = 0.05  # seconds


class HttpIO(IOBase, BinaryIO):
    """Access a file available via HTTP like a file-like object.

    Supported operations are seeking and reading. This is accomplished by utilizing
    the HTTP *Range* header.

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
            raise OSError('Server does not provide a \'Content-Length\' header')

        self._length = int(headers['content-length'])
        self._pos = 0

    def _check_closed(self) -> None:
        """Raise a ``ValueError`` if the file is closed."""
        if self.closed:
            raise ValueError('I/O operation on closed file')

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
        """URL this instance reads from."""
        return self._url

    def write(self, b: bytes) -> int:
        """Write the given buffer to the IO stream.

        Returns the number of bytes written, which may be less than the length of ``b``
        in bytes.
        """
        raise UnsupportedOperation(f'{self.__class__.__name__}.write() not supported')

    def readable(self) -> bool:
        """Return a ``bool`` indicating whether the object supports reading.

        True for all ``HttpIO`` instances.
        """
        return True

    def seekable(self) -> bool:
        """Return a ``bool`` indicating whether the object supports random access.

        True for all ``HttpIO`` instances.
        """
        return True

    def read(self, size: int = -1) -> bytes:
        """Read and return up to ``size`` bytes.

        Returns an empty ``bytes`` object on EOF.
        """
        self._check_closed()

        if size == 0 or self._pos >= self._length:
            return b''
        if size < 0 or self._pos + size > self._length:
            size = self._length - self._pos

        # range request
        first_byte_pos = self._pos
        last_byte_pos = self._pos + size - 1  # inclusive!
        headers = {'range': f'bytes={first_byte_pos}-{last_byte_pos}'}

        response = self._session.get(self._url, headers=headers, timeout=HTTP_TIMEOUT)
        response.raise_for_status()

        if response.status_code != 206 or 'content-range' not in response.headers:
            raise OSError('Server did not send a partial response')

        data = response.content
        actual_size = len(data)

        if actual_size != size:
            raise ConnectionError(
                f'Server did not send the requested amount of bytes (expected {size}, '
                f'got {actual_size})'
            )

        self._pos += actual_size
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
        self._checkClosed()
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

# noinspection PyUnusedLocal
def tps_default(filepath_iso: PurePosixPath, target_paths: tuple[Path, ...]) -> Path:
    """Default strategy to determine the correct target path for a specific file
    present in an ISO image file.

    Always chooses the first target path of the target paths passed.
    """
    if len(target_paths) < 1:
        raise ValueError(
            f'Strategy requires at least 1 target path, got {target_paths}'
        )
    if filepath_iso.is_absolute():
        raise ValueError(f'filepath_iso must be a relative path, got {filepath_iso}')
    return target_paths[0]


def tps_win10_uefi(filepath_iso: PurePosixPath, target_paths: tuple[Path, ...]) -> Path:
    """Strategy to determine the correct target path for a specific file present in a
    Windows 10 or newer ISO image file if this ISO image file is used to create a
    bootable USB drive for a UEFI system.

    Chooses the second target path (NTFS volume) for the *sources* folder except
    *sources/boot.wim*.
    Chooses the first target path for everything else.

    For more information on why we copy the installation files of Windows 10+ images
    to multiple volumes, refer to the documentation of the ``drive`` package.
    """
    if len(target_paths) != 2:
        raise ValueError(
            f'Strategy requires exactly 2 target paths, got {target_paths}'
        )
    if filepath_iso.is_absolute():
        raise ValueError(f'filepath_iso must be a relative path, got {filepath_iso}')

    parts = filepath_iso.parts
    part_count = len(parts)

    if part_count == 0 or (parts[0] == 'sources' and part_count == 1):
        raise ValueError(f'Got invalid filepath_iso {filepath_iso} for Windows 10+')

    # actual logic
    if filepath_iso.parts[0] == 'sources' and not filepath_iso.parts[1] == 'boot.wim':
        return target_paths[1]
    else:
        return target_paths[0]


# --- Extraction


class OpenResult(NamedTuple):
    """File-like object and PyCdlib object passed back to the main thread after
    opening the ISO image file.
    """

    source_file: BinaryIO
    iso: PyCdlib


class ExtractJob(NamedTuple):
    """Information required for a worker thread to extract a specific file from an
    ISO image file to a local directory.

    filepath_iso: Path of the file to extract, relative to the ISO root directory.
    filepath_local: Desired path of the file on the local file system.
    iso_path_type: Required for PyCdlib API.
    """

    filepath_iso: PurePosixPath
    filepath_local: Path
    iso_path_type: str


def _extract_worker(
    source: LocalSource | RemoteSource,
    open_result_queue: Queue[OpenResult],
    extract_queue: Queue[ExtractJob],
    exc_queue: Queue[BaseException],
    # progress: Value,
    # pause_event: Event,
    quit_event: Event,
) -> None:
    """Function executed by worker threads spawned to extract files from an ISO image.

    All workers first open an IO handle for the desired ISO image and use that to
    instantiate a new ``PyCdlib`` object. IO handle and ``PyCdlib`` object are then
    added to ``open_result_queue`` exactly once for the main thread to iterate over
    the files present in the ISO image. After that, every file path is added to
    ``extract_queue`` by the main thread, so that the spawned workers (described by
    this function) can execute these "extraction jobs" by extracting the according
    files.

    :param exc_queue: Queue of exceptions which were raised by this function.
    :param progress: Counter describing how many bytes were already extracted.
    :param pause_event: Event indicating that the worker threads are to be paused.
    :param quit_event: Event indicating that the worker threads are to be stopped.
    """
    try:
        source_file: BinaryIO
        if isinstance(source, LocalSource):
            blocksize = BLOCK_SIZE_LOCAL
            source_file = source.path.open(mode='rb')
        else:
            blocksize = BLOCK_SIZE_REMOTE
            source_file = HttpIO(source.url)

        with source_file:
            # open ISO
            iso = PyCdlib()
            iso.open_fp(source_file)  # this might take a while
            try:
                open_result = OpenResult(source_file, iso)
                open_result_queue.put(open_result, block=False)
            except Full:
                # main thread only needs (source_file, iso) once
                pass

            while not quit_event.is_set():
                try:
                    # wait a bit in case there are no jobs in the queue yet
                    job = extract_queue.get(timeout=EXTRACT_QUEUE_TIMEOUT)
                except Empty:
                    break

                filepath_iso, filepath_local, iso_path_type = job

                # create parent directories if necessary
                dirpath_local_ = filepath_local.parent
                dirpath_local_.mkdir(parents=True, exist_ok=True)

                # noinspection PyUnresolvedReferences
                filepath_iso_abs = '/' / filepath_iso
                log.debug(f'Extracting {filepath_iso_abs}')

                # TODO: progess, pause, abort
                iso.get_file_from_iso(
                    str(filepath_local),  # pycdlib doesn't like pathlib paths
                    **{iso_path_type: str(filepath_iso_abs)},
                    blocksize=blocksize,
                )
                extract_queue.task_done()

            iso.close()

    except BaseException as e:
        exc_queue.put(e)


def extract(
    source: LocalSource | RemoteSource,
    target_paths: Path | tuple[Path, ...],
    target_path_strategy: Callable[
        [PurePosixPath, tuple[Path, ...]], Path
    ] = tps_default,
) -> None:
    """Extract the contents of an ISO image file to local directories.

    Args:
        source: Source of the image file described by a LocalSource or a
            RemoteSource.

        target_paths: Paths to local directories to extract the contents of the image
            file to. Usually only one target path is required.
            The length of this tuple must match the length of the tuple the second
            parameter of the callable target_path_strategy accepts.
            Instead of passing a tuple of length 1, a single Path can also be passed.

        target_path_strategy: Callable which determines which file (first parameter,
            file path relative to ISO root) to extract to which local directory
            specified in target_paths (second parameter).
            Returns one of the target paths defined in the second parameter.
            Functions which can be used as a value for this parameter are exported by
            the image module and are named tps_*.
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
        # progress = Value('q', 0)  # zero bytes processed yet
        # pause_event = Event()
        quit_event = Event()

        # open IO handles separately to avoid side effects in extraction workers
        futures = []
        for _ in range(max_workers):
            future = executor.submit(
                _extract_worker,
                source,
                open_result_queue,
                extract_queue,
                exc_queue,
                # progress,
                # pause_event,
                quit_event,
            )
            futures.append(future)

        # get first open IO handle and PyCdlib object
        while True:
            try:
                source_file, iso = open_result_queue.get(block=False)
                break
            except Empty:
                # open_result_queue stays empty forever if an exception occurred in
                # all worker threads before enqueuing a result
                try:
                    exc = exc_queue.get(timeout=EXC_QUEUE_TIMEOUT)
                except Empty:
                    continue
                else:
                    raise RuntimeError('Exception raised in worker thread') from exc

        log.debug(f'Opened ISO after {(time.perf_counter() - time_start):.4f} seconds')

        # get file size
        size = source_file.seek(0, 2)
        source_file.seek(0, 0)
        log.debug(f'ISO size: {size} bytes')

        space_available = sum(disk_usage(path).free for path in target_paths)
        log.debug(
            f'Total space available at target directories: {space_available} bytes'
        )
        if space_available + TARGET_PATHS_EXTRA_FREE_SPACE < size:
            raise ValueError('Not enough disk space available at target directories')

        # detect supported ISO extensions
        if iso.has_udf():
            # prioritize UDF if it's available because it's required for modern
            # Windows ISOs (>= Vista)
            iso_path_type = 'udf_path'
        elif iso.has_joliet():
            iso_path_type = 'joliet_path'
        elif iso.has_rock_ridge():
            iso_path_type = 'rr_path'
        else:
            iso_path_type = 'iso_path'
            log.warning('Fallback to pure ISO 9660 format')

        log.debug(f'Selected ISO format with path type {iso_path_type!r}')

        # actual extraction
        time_extraction_start = time.perf_counter()
        log.info('Extracting files from ISO image ...')

        try:
            # The try block needs to start here because from here on jobs are added to
            # extract_queue and these jobs keep the worker threads busy which means
            # they won't quit until the queue is empty if an exception is raised here.

            for dirpath_iso, _, filelist in iso.walk(**{iso_path_type: '/'}):

                dirpath_iso = PurePosixPath(dirpath_iso).relative_to('/')
                # dirpath_iso: path to current dir (relative to iso root)
                # filelist: list of files in current dir

                for filename in filelist:
                    filepath_iso = dirpath_iso / filename
                    rootpath_local = target_path_strategy(filepath_iso, target_paths)
                    filepath_local = rootpath_local / filepath_iso

                    extract_queue.put(
                        ExtractJob(filepath_iso, filepath_local, iso_path_type)
                    )

            # waiting for all worker threads to finish
            while not all(future.done() for future in futures):
                try:
                    exc = exc_queue.get(timeout=EXC_QUEUE_TIMEOUT)
                except Empty:
                    continue
                else:
                    raise RuntimeError('Exception raised in worker thread') from exc

        except BaseException:
            # Any unhandled exception (incl. KeyboardInterrupt) raised inside the try
            # block above wouldn't take effect if the worker threads stayed busy with
            # jobs from extract_queue because the workers wouldn't quit in a timely
            # manner.
            log.warning(
                'An unhandled exception was raised. Waiting for all workers to quit ...'
            )
            quit_event.set()
            wait(futures)
            raise

        # make sure we don't miss any exception or cancellation of a future
        for future in as_completed(futures):
            future.result()

        extraction_time = time.perf_counter() - time_extraction_start
        log.debug(f'Finished extraction after {extraction_time:.4f} seconds')
        log.debug(
            f'Average extraction speed: {(size / extraction_time):.4f} bytes per second'
        )
        log.info('Finished ISO image extraction')
