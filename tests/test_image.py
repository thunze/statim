import os
import re
import time
from collections import namedtuple
from functools import partial
from io import SEEK_CUR, SEEK_END, SEEK_SET, BytesIO, UnsupportedOperation
from multiprocessing import Value
from pathlib import Path, PurePath, PurePosixPath
from random import randbytes
from shutil import rmtree
from tempfile import mkdtemp, mkstemp
from threading import Event
from typing import BinaryIO, Callable

import pytest
from pycdlib import PyCdlib
from pycdlib.facade import PyCdlibISO9660, PyCdlibJoliet, PyCdlibRockRidge, PyCdlibUDF
from pycdlib.pycdlibexception import PyCdlibInvalidInput
from requests import Request
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError
from requests_mock import Mocker

# noinspection PyProtectedMember
from statim.image import (
    CHUNK_SIZE_REMOTE_CHUNKED,
    ExtractJob,
    ExtractProgress,
    HttpIO,
    _extract_file,
    _get_facade_for_iso,
    _pause_or_quit,
    extract,
    iso_size_contents,
    tps_default,
    tps_win10_uefi,
)
from statim.plan import LocalSource, RemoteSource

TEST_URL = 'http://example.test'

# used for creation of files and directories on Rock Ridge images, only permission to
# read is important here
RR_DEFAULT_FILE_MODE = 0o100444


@pytest.fixture
def http_io_from_data(requests_mock: Mocker):
    """Fixture providing a function which returns an ``HttpIO`` object to be used for
    testing purposes by mocking responses to `HEAD` and `GET` requests.

    The data the mocked endpoint should provide is to be passed to the returned
    function.
    """

    def http_io_from_data_(content):
        """Prepare an ``HttpIO`` object for testing purposes by mocking responses to
        `HEAD` and `GET` requests and return it.

        :param content: The data the mocked endpoint should provide.
        """
        requests_mock.head(
            TEST_URL,
            status_code=200,
            headers={'accept-ranges': 'bytes', 'content-length': str(len(content))},
        )

        def content_callback(request_: Request, context):
            """Dynamically generate the body of a response to a `GET` (range) request.

            Only accepts range requests as sent by ``HttpIO.read``.
            """
            content_length = len(content)
            range_ = request_.headers.get('range', '')
            if not range_:
                context.status_code = 200
                context.headers['content-length'] = str(content_length)
                return content

            range_start, range_end = range_.split('=')[1].split('-')
            # HTTP range is inclusive
            range_start, range_end = int(range_start), int(range_end) + 1

            if range_start < 0 or range_end > content_length:
                context.status_code = 416  # Range Not Satisfiable
                context.headers['content-range'] = f'bytes */{content_length}'
                return b''

            partial_content = content[range_start:range_end]
            context.status_code = 206
            context.headers['content-length'] = str(range_end - range_start)
            context.headers[
                'content-range'
            ] = f'bytes {range_start}-{range_end - 1}/{content_length}'
            return partial_content

        requests_mock.get(
            TEST_URL, headers={'accept-ranges': 'bytes'}, content=content_callback
        )
        return HttpIO(TEST_URL)

    return http_io_from_data_


@pytest.fixture
def http_io(request, http_io_from_data):
    """Fixture preparing an ``HttpIO`` object for testing purposes by mocking
    responses to `HEAD` and `GET` requests.

    The data the mocked endpoint should provide can be set using indirect
    parametrization.

    Returns the prepared ``HttpIO`` object.
    """
    return http_io_from_data(request.param)


class TestHttpIO:
    """Tests for ``HttpIO``, a file-like wrapper for files available via HTTP."""

    TEST_HEADER_BASES = [
        {'accept-ranges': 'bytes', 'content-length': '16'},
        {'accept-ranges': 'bytes', 'content-length': '16', 'content-encoding': ''},
    ]
    TEST_DATA = [b'', b'test', b'The knots will tie you down! ' * 7, randbytes(443)]
    TEST_CHUNK_SIZES = [1, 2, 7, 100]

    # Fixtures

    @pytest.fixture
    def http_io_head(self, request, requests_mock: Mocker):
        """Fixture preparing an ``HttpIO`` object for testing purposes by mocking
        responses to `HEAD` requests only.

        This is useful if the test function itself wants to define how `GET` requests
        are handled.

        The data the mocked endpoint should pretend to provide can be set using
        indirect parametrization.

        Returns the prepared ``HttpIO`` object.
        """
        content = request.param
        requests_mock.head(
            TEST_URL,
            status_code=200,
            headers={'accept-ranges': 'bytes', 'content-length': str(len(content))},
        )
        return HttpIO(TEST_URL)

    # Tests

    @pytest.mark.parametrize('header_base', TEST_HEADER_BASES)
    @pytest.mark.parametrize('content_length', [0, 5, 1234])
    def test_init_success(self, header_base, content_length, requests_mock: Mocker):
        """Test initialization using acceptable `HEAD` responses."""
        requests_mock.head(
            TEST_URL,
            status_code=200,
            headers=header_base | {'content-length': str(content_length)},
        )
        http_io = HttpIO(TEST_URL)

        assert http_io._url == TEST_URL
        assert http_io._session.headers['accept-encoding'] == 'identity'
        assert http_io._length == content_length
        assert http_io._pos == 0

    @pytest.mark.parametrize('status_code', [400, 404, 500])
    def test_init_fail_status(self, status_code, requests_mock: Mocker):
        """Test initialization using `HEAD` responses with unacceptable HTTP status
        codes.
        """
        requests_mock.head(TEST_URL, status_code=status_code)
        with pytest.raises(HTTPError):
            HttpIO(TEST_URL)

    @pytest.mark.parametrize('header_base', TEST_HEADER_BASES)
    @pytest.mark.parametrize('accept_ranges', [None, '', 'none', 'None'])
    def test_init_fail_accept_ranges(
        self, header_base, accept_ranges, requests_mock: Mocker
    ):
        """Test initialization using `HEAD` responses without or with unacceptable
        values for the `Accept-Range` header.
        """
        headers = header_base.copy()
        if accept_ranges is None:
            del headers['accept-ranges']
        else:
            headers = header_base | {'accept-ranges': accept_ranges}
        requests_mock.head(TEST_URL, status_code=200, headers=headers)
        with pytest.raises(OSError, match='Server does not support range requests'):
            HttpIO(TEST_URL)

    @pytest.mark.parametrize('header_base', TEST_HEADER_BASES)
    @pytest.mark.parametrize('content_encoding', ['gzip', 'deflate, gzip'])
    def test_init_fail_content_encoding(
        self, header_base, content_encoding, requests_mock: Mocker
    ):
        """Test initialization using `HEAD` responses with unacceptable values for the
        `Content-Encoding` header.
        """
        requests_mock.head(
            TEST_URL,
            status_code=200,
            headers=header_base | {'content-encoding': content_encoding},
        )
        with pytest.raises(
            OSError, match='Range requests are not supported for encoded content'
        ):
            HttpIO(TEST_URL)

    @pytest.mark.parametrize('header_base', TEST_HEADER_BASES)
    def test_init_fail_content_length(self, header_base, requests_mock: Mocker):
        """Test initialization using `HEAD` responses without a `Content-Length`
        header.
        """
        headers = header_base.copy()
        del headers['content-length']
        requests_mock.head(TEST_URL, status_code=200, headers=headers)
        with pytest.raises(
            OSError, match='Server does not provide a \'Content-Length\' header'
        ):
            HttpIO(TEST_URL)

    @pytest.mark.parametrize('http_io', TEST_DATA, indirect=True)
    def test_enter(self, http_io: HttpIO):
        """Test context management protocol on ``HttpIO`` objects."""
        with http_io as f:
            assert type(f) is HttpIO
        # skipcq: PTC-W0062
        with pytest.raises(ValueError, match='I/O operation on closed file'):
            with http_io:
                pass

    @pytest.mark.parametrize('http_io', TEST_DATA, indirect=True)
    def test_properties(self, http_io: HttpIO):
        """Test that ``HttpIO`` objects are seekable, read-only binary streams.

        Also test that the printable representation of an ``HttpIO`` object contains
        the URL the object was initialized with.
        """
        assert http_io.mode == 'rb'
        assert http_io.readable()
        assert not http_io.writable()
        assert http_io.seekable()

        assert http_io.name == TEST_URL
        assert repr(http_io) == f'HttpIO(url={TEST_URL!r})'

    @pytest.mark.parametrize(
        ['http_io', 'data'], zip(TEST_DATA, TEST_DATA), indirect=['http_io']
    )
    def test_write(self, http_io: HttpIO, data):
        """Test that ``write`` always raises ``io.UnsupportedOperation``."""
        with pytest.raises(UnsupportedOperation):
            http_io.write(data)

    @pytest.mark.parametrize(
        ['http_io', 'data', 'pos', 'size'],
        [
            (data, data, pos, size)
            for data in TEST_DATA
            for pos in [0, len(data) // 2, len(data)]
            for size in [-2, -1, 0, len(data) // 5 + 1, len(data) // 2, len(data)]
        ],
        indirect=['http_io'],
    )
    def test_read_success(self, http_io: HttpIO, data, pos, size):
        """Test that ``read`` returns the requested bytes.

        Repeatedly reads ``size`` bytes from ``http_io`` until EOF is reached.

        :param pos: Stream position to seek to before reading.
        """
        http_io.seek(pos)
        read = None

        while read != b'':
            read = http_io.read(size)

            if size < 0:  # readall
                assert len(read) == len(data) - pos
                assert read == data[pos:]
            else:
                assert read == data[pos : pos + size]

                if pos + size > len(data):
                    assert len(read) == len(data) - pos
                else:
                    assert len(read) == size

            pos += len(read)
            assert http_io.tell() == pos

    @pytest.mark.parametrize(
        ['http_io_head', 'data'],
        # read on empty data returns early
        [(data, data) for data in TEST_DATA if len(data) > 0],
        indirect=['http_io_head'],
    )
    @pytest.mark.parametrize(
        ['status_code', 'content_range'], [(200, True), (200, False), (206, False)]
    )
    def test_read_fail_no_partial_response(
        self,
        http_io_head: HttpIO,
        data,
        status_code,
        content_range,
        requests_mock: Mocker,
    ):
        """Test that ``read`` raises ``OSError`` if it doesn't receive a partial
        response from the server.
        """
        headers = {'accept-ranges': 'bytes', 'content-length': str(len(data))}
        if content_range:
            # always send the full data range (doesn't really matter for this test)
            headers['content-range'] = f'bytes {0}-{len(data) - 1}/{len(data)}'

        requests_mock.get(
            TEST_URL, status_code=status_code, headers=headers, content=data
        )
        with pytest.raises(OSError, match='Server did not send a partial response'):
            http_io_head.read()

    @pytest.mark.parametrize(
        ['http_io_head', 'data'],
        # read on empty data returns early
        [(data, data) for data in TEST_DATA if len(data) > 0],
        indirect=['http_io_head'],
    )
    def test_read_fail_wrong_byte_count(
        self, http_io_head: HttpIO, data, requests_mock: Mocker
    ):
        """Test that ``read`` raises ``ConnectionError`` if it doesn't receive the
        requested amount of bytes as a response from the server.
        """
        requests_mock.get(
            TEST_URL,
            status_code=206,
            headers={
                'accept-ranges': 'bytes',
                'content-length': str(len(data)),
                'content-range': f'bytes {0}-{len(data) - 1}/{len(data)}',
            },
            # we request the complete file, we respond with the according headers,
            # but in reality we send back one byte less than expected
            content=data[: len(data) - 1],
        )
        with pytest.raises(
            ConnectionError,
            match=(
                r'Server did not send the requested amount of bytes \(expected \d+, '
                r'got \d+\)'
            ),
        ):
            http_io_head.read()

    @pytest.mark.parametrize(
        ['http_io', 'data', 'pos'],
        [
            (data, data, pos)
            for data in TEST_DATA
            for pos in [0, len(data) // 2, len(data)]
        ],
        indirect=['http_io'],
    )
    def test_readall(self, http_io: HttpIO, data, pos):
        """Test that ``readall`` returns all bytes until EOF.

        :param pos: Stream position to seek to before calling ``readall``.
        """
        http_io.seek(pos)
        readall = http_io.readall()
        assert readall == data[pos:]
        assert http_io.tell() == len(data)
        http_io.seek(pos)
        assert readall == http_io.read()

    @pytest.mark.parametrize(
        ['http_io', 'data'], zip(TEST_DATA, TEST_DATA), indirect=['http_io']
    )
    def test_seek_tell(self, http_io: HttpIO, data):
        """Test that ``seek`` correctly changes the stream position and that ``tell``
        returns it.
        """
        assert http_io.tell() == 0

        # invalid whence values
        with pytest.raises(
            ValueError,
            match=r'Unsupported whence value, must be one of \(0, 1, 2\)',  # regex!
        ):
            http_io.seek(0, -1)
        with pytest.raises(
            ValueError, match=r'Unsupported whence value, must be one of \(0, 1, 2\)'
        ):
            http_io.seek(0, 3)

        # SEEK_SET
        assert http_io.tell() == 0
        with pytest.raises(ValueError, match=r'Negative seek position -\d+'):
            http_io.seek(-123546, SEEK_SET)
        assert http_io.seek(0, SEEK_SET) == 0
        assert http_io.tell() == 0
        assert http_io.seek(1, SEEK_SET) == 1
        assert http_io.tell() == 1
        assert http_io.seek(len(data) // 2, SEEK_SET) == len(data) // 2
        assert http_io.tell() == len(data) // 2

        # SEEK_CUR
        assert http_io.seek(1, SEEK_CUR) == len(data) // 2 + 1
        assert http_io.seek(-1, SEEK_CUR) == len(data) // 2
        assert http_io.tell() == len(data) // 2

        assert http_io.seek(0, SEEK_SET) == 0  # reset
        assert http_io.seek(0, SEEK_CUR) == 0
        assert http_io.seek(-1, SEEK_CUR) == 0
        assert http_io.tell() == 0

        # SEEK_END
        assert http_io.seek(-len(data), SEEK_END) == 0
        assert http_io.tell() == 0
        assert http_io.seek(-len(data) - 10, SEEK_END) == 0
        assert http_io.tell() == 0
        assert http_io.seek(0, SEEK_END) == len(data)
        assert http_io.tell() == len(data)

        # new pos > len(data)
        assert http_io.seek(len(data) + 1, SEEK_SET) == len(data) + 1
        assert http_io.tell() == len(data) + 1

        assert http_io.seek(0, SEEK_SET) == 0  # reset
        assert http_io.tell() == 0

        if len(data) > 0:
            assert http_io.seek(len(data) - 1, SEEK_SET) == len(data) - 1
            assert http_io.tell() == len(data) - 1

            assert http_io.seek(-1, SEEK_END) == len(data) - 1
            assert http_io.seek(1, SEEK_CUR) == len(data)
            assert http_io.seek(10, SEEK_CUR) == len(data) + 10

        assert http_io.seek(0, SEEK_SET) == 0  # reset
        assert http_io.tell() == 0

    @pytest.mark.parametrize('http_io', TEST_DATA, indirect=True)
    @pytest.mark.parametrize(
        ['method', 'args'],
        [('read', ()), ('readall', ()), ('seek', (42,)), ('tell', ())],
    )
    def test_close(self, http_io: HttpIO, method, args):
        """Test that invoking certain methods on a closed ``HttpIO`` object raises
        ``ValueError``.
        """
        http_io.close()
        with pytest.raises(ValueError, match='I/O operation on closed file'):
            getattr(http_io, method)(*args)

    @pytest.mark.parametrize(
        ['http_io', 'data', 'pos'],
        [
            (data, data, pos)
            for data in TEST_DATA
            for pos in [0, len(data) // 2, len(data)]
        ],
        indirect=['http_io'],
    )
    @pytest.mark.parametrize('chunk_size', TEST_CHUNK_SIZES)
    def test_chunked_transfer_success(self, http_io: HttpIO, data, pos, chunk_size):
        """Test successful chunked transfer scenarios."""
        http_io.seek(pos)

        # using the iterator provided by the context manager
        with http_io.chunked_transfer(chunk_size) as chunks:
            chunk_list = []
            last_pos = http_io.tell()

            for chunk in chunks:
                chunk_list.append(chunk)
                assert chunk == data[last_pos : last_pos + chunk_size]
                assert http_io.tell() == min(last_pos + chunk_size, len(data))
                last_pos = http_io.tell()

            assert b''.join(chunk_list) == data[pos:]
            assert http_io.tell() == len(data)

        http_io.seek(pos)

        # using httpio.read -- read in chunks of chunk_size bytes
        with http_io.chunked_transfer(chunk_size):
            chunk_list = []
            last_pos = http_io.tell()

            while True:
                chunk = http_io.read(chunk_size)
                chunk_list.append(chunk)
                assert chunk == data[last_pos : last_pos + chunk_size]
                assert http_io.tell() == min(last_pos + chunk_size, len(data))

                if not chunk:
                    assert b''.join(chunk_list) == data[pos:]
                    assert http_io.tell() == len(data)
                    break

                last_pos = http_io.tell()

        http_io.seek(pos)

        # using httpio.read -- read less than chunk_size bytes
        with http_io.chunked_transfer(chunk_size):
            read_size_less = chunk_size - 1
            chunk = http_io.read(read_size_less)
            assert chunk == data[pos : pos + read_size_less]

            if read_size_less == 0:
                assert http_io.tell() == pos  # chunk not consumed
            else:
                # whole chunk consumed (!)
                assert http_io.tell() == min(pos + chunk_size, len(data))

        # back to usual IO
        assert http_io.seekable()
        http_io.seek(pos)
        read_size_less = chunk_size - 1
        assert http_io.read(read_size_less) == data[pos : pos + read_size_less]
        assert http_io.tell() == min(pos + read_size_less, len(data))

    @pytest.mark.parametrize('http_io', TEST_DATA, indirect=True)
    def test_chunked_transfer_fail_already_active(self, http_io: HttpIO):
        """Test that starting a new chunked transfer fails if another chunked
        transfer is already active.
        """
        # skipcq: PTC-W0062
        with http_io.chunked_transfer(1):
            # skipcq: PTC-W0062
            with pytest.raises(
                ValueError, match='Another chunked transfer is already active'
            ):
                with http_io.chunked_transfer(2):
                    pass

            assert not http_io.seekable()

    @pytest.mark.parametrize('http_io', TEST_DATA, indirect=True)
    @pytest.mark.parametrize('chunk_size', [-5, -1, 0])
    def test_chunked_transfer_fail_chunk_size(self, http_io: HttpIO, chunk_size):
        """Test that starting a new chunked transfer with a chunk size not greater
        than 0 fails.
        """
        # skipcq: PTC-W0062
        with pytest.raises(
            ValueError, match=r'Chunk size must be greater than 0, got -?\d+'
        ):
            with http_io.chunked_transfer(chunk_size):
                pass

    @pytest.mark.parametrize(
        ['http_io', 'data', 'pos'],
        [
            (data, data, pos)
            for data in TEST_DATA
            for pos in [0, len(data) // 2, len(data)]
        ],
        indirect=['http_io'],
    )
    @pytest.mark.parametrize('chunk_size', TEST_CHUNK_SIZES)
    def test_chunked_transfer_fail_read(self, http_io: HttpIO, data, pos, chunk_size):
        """Test that during chunked transfer trying to read more data than specified
        by ``chunk_size`` at once fails.
        """
        http_io.seek(pos)
        bytes_left = len(data) - pos
        exc_message = (
            rf'Can only read in chunks of {chunk_size} bytes or less during chunked '
            r'transfer \({} bytes requested\)'
        )

        with http_io.chunked_transfer(chunk_size):
            if bytes_left > chunk_size:

                for read_size in [chunk_size + 1, chunk_size + 20]:
                    with pytest.raises(
                        ValueError, match=exc_message.format(min(bytes_left, read_size))
                    ):
                        http_io.read(read_size)

                with pytest.raises(ValueError, match=exc_message.format(bytes_left)):
                    http_io.read(-1)
                with pytest.raises(ValueError, match=exc_message.format(bytes_left)):
                    http_io.readall()
                with pytest.raises(ValueError, match=exc_message.format(bytes_left)):
                    http_io.read()

                assert http_io.tell() == pos

    @pytest.mark.parametrize('http_io', TEST_DATA, indirect=True)
    @pytest.mark.parametrize(['offset', 'whence'], [(1, 0), (1, 1), (-1, 2)])
    def test_chunked_transfer_fail_seek(self, http_io: HttpIO, offset, whence):
        """Test that seeking fails during chunked transfer."""
        assert http_io.seekable()

        with http_io.chunked_transfer(1):
            assert not http_io.seekable()

            with pytest.raises(
                ValueError, match='Random-access operation during chunked transfer'
            ):
                http_io.seek(offset, whence)

            assert not http_io.seekable()
        assert http_io.seekable()

    @pytest.mark.parametrize('http_io', TEST_DATA, indirect=True)
    def test_chunked_transfer_fail_context_left(self, http_io: HttpIO):
        """Test that the iterator provided by ``chunked_transfer`` stops if its
        context is left.
        """
        with http_io.chunked_transfer(1) as chunks:
            pass
        with pytest.raises(StopIteration):
            next(chunks)


# --- Target path strategies


class TestTps:
    """Tests for target path strategies."""

    # Fixtures

    @pytest.fixture(scope='class')
    def target_paths(self, request):
        """Return a ``tuple`` of ``<arg>`` distinct ``Path`` objects."""
        return tuple(Path() for _ in range(request.param))

    # Tests

    @pytest.mark.parametrize(
        ['target_path_strategy', 'target_paths'],
        [(tps_default, 1), (tps_win10_uefi, 2)],
        indirect=['target_paths'],
    )
    def test_tps_fail_absolute(
        self, target_path_strategy, target_paths: tuple[Path, ...]
    ):
        """Test that all target path strategies raise ``ValueError`` if a relative
        path is passed as a value for ``filepath_iso``.
        """
        with pytest.raises(
            ValueError, match='filepath_iso must be an absolute path, got thing'
        ):
            target_path_strategy(PurePosixPath('thing'), target_paths)

    @pytest.mark.parametrize('target_paths', range(1, 4), indirect=True)
    def test_tps_default_success(self, target_paths: tuple[Path, ...]):
        """Test that the default target path strategy always returns the first target
        path of the target paths passed.
        """
        filepath_iso = PurePosixPath('/thing')
        assert tps_default(filepath_iso, target_paths) is target_paths[0]

    @staticmethod
    def test_tps_default_fail():
        """Test that the default target path strategy raises ``ValueError`` if less
        than one target path is passed.
        """
        with pytest.raises(
            ValueError, match='Strategy requires at least 1 target path, got ()'
        ):
            tps_default(PurePosixPath('/thing'), ())

    @pytest.mark.parametrize('target_paths', [2], indirect=['target_paths'])
    def test_tps_win10_uefi_success(self, target_paths: tuple[Path, ...]):
        """Test that the target path strategy ``tps_win10_uefi`` returns the correct
        target path of the target paths passed.
        """
        tp = target_paths
        assert tps_win10_uefi(PurePosixPath('/thing'), tp) is tp[0]
        assert tps_win10_uefi(PurePosixPath('/nested/thing'), tp) is tp[0]
        assert tps_win10_uefi(PurePosixPath('/nested/boot.wim'), tp) is tp[0]
        assert tps_win10_uefi(PurePosixPath('/sources/boot.wim'), tp) is tp[0]
        assert tps_win10_uefi(PurePosixPath('/sources/thing'), tp) is tp[1]

    @pytest.mark.parametrize('target_paths', [0, 1, 3], indirect=True)
    def test_tps_win10_uefi_fail_target_paths(self, target_paths: tuple[Path, ...]):
        """Test that the target path strategy ``tps_win10_uefi`` raises ``ValueError``
        if less or more than 2 target paths are passed.
        """
        with pytest.raises(
            ValueError,
            match=re.escape(
                f'Strategy requires exactly 2 target paths, got {target_paths}'
            ),
        ):
            tps_win10_uefi(PurePosixPath('/thing'), target_paths)

    @pytest.mark.parametrize(
        ['filepath_iso', 'target_paths'],
        [(PurePosixPath('/'), 2), (PurePosixPath('/sources'), 2)],
        indirect=['target_paths'],
    )
    def test_tps_win10_uefi_fail_filepath_iso(
        self, filepath_iso: PurePosixPath, target_paths: tuple[Path, ...]
    ):
        """Test that the target path strategy ``tps_win10_uefi`` raises ``ValueError``
        if certain values for ``filepath_iso`` are passed which are invalid for
        Windows 10+ images.
        """
        with pytest.raises(
            ValueError, match=f'Got invalid filepath_iso {filepath_iso} for Windows 10+'
        ):
            tps_win10_uefi(filepath_iso, target_paths)


# --- Extraction


@pytest.mark.parametrize('rock_ridge', [None, '1.09'])
@pytest.mark.parametrize('joliet', [None, 3])
@pytest.mark.parametrize('udf', [None, '2.60'])
def test__get_facade_for_iso(rock_ridge, joliet, udf):
    """Test that ``_get_facade_for_iso`` returns the most preferable of the available
    facades of a ``PyCdlib`` object.
    """
    iso = PyCdlib()
    iso.new(rock_ridge=rock_ridge, joliet=joliet, udf=udf)
    facade = _get_facade_for_iso(iso)

    if udf:
        assert type(facade) is PyCdlibUDF
    elif joliet:
        assert type(facade) is PyCdlibJoliet
    elif rock_ridge:
        assert type(facade) is PyCdlibRockRidge
    else:
        assert type(facade) is PyCdlibISO9660


def test__pause_or_quit():
    """Test that ``_pause_or_quit`` returns whether quitting is desired.

    Pausing is not tested here but by the tests for ``extract``.
    """
    quit_event = Event()
    resume_event = Event()
    resume_event.set()

    assert not _pause_or_quit(quit_event, resume_event)
    quit_event.set()
    assert _pause_or_quit(quit_event, resume_event)


@pytest.fixture
def tempdir():
    """Fixture providing a new temporary directory for testing purposes."""
    path = Path(mkdtemp())
    yield path
    rmtree(path)  # clean up


@pytest.fixture
def tempfile():
    """Fixture providing a new temporary file for testing purposes.

    The file is opened using the mode ``wb+`` to allow for reading and writing.
    """
    fd, path_str = mkstemp()
    os.close(fd)  # we use Path.open() instead
    path = Path(path_str)
    file = path.open('wb+')
    yield file
    file.close()
    path.unlink(missing_ok=True)  # clean up


@pytest.fixture
def local_io_from_data(tempfile):
    """Fixture providing a function which returns an IO handle for a local file to be
    used for testing purposes.

    The data the file should provide is to be passed to the returned function.
    """

    def local_io_from_data_(content):
        """Prepare a local file for testing purposes and return an IO handle for it.

        :param content: The data the file should provide.
        """
        tempfile.write(content)
        tempfile.seek(0)
        return tempfile

    return local_io_from_data_


@pytest.fixture
def io_from_iso(
    http_io_from_data: Callable[[bytes], HttpIO],
    local_io_from_data: Callable[[bytes], BinaryIO],
):
    """Fixture providing a function which masters an ISO image passed in the form of a
    ``PyCdlib`` object and returns an IO handle for the mastered image.
    """

    def io_from_iso_(iso: PyCdlib, remote: bool) -> BinaryIO:
        """Master ``iso`` and return an IO handle for the mastered image.

        An ``HttpIO`` object is returned if ``remote`` is true. Otherwise an
        ``io.BufferedReader`` object reading from a local file is returned.
        """
        buffer = BytesIO()
        iso.write_fp(buffer)
        buffer.seek(0)
        data = buffer.read()

        if remote:
            return http_io_from_data(data)
        else:
            return local_io_from_data(data)

    return io_from_iso_


@pytest.fixture
def source_from_iso(io_from_iso):
    """Fixture providing a function which masters an ISO image passed in the form of a
    ``PyCdlib`` object and returns a ``LocalSource`` or ``RemoteSource`` for the
    mastered image.
    """

    def source_from_iso_(iso: PyCdlib, remote) -> LocalSource | RemoteSource:
        """Master ``iso`` and return a ``LocalSource`` or ``RemoteSource`` for the
        mastered image.

        A ``RemoteSource`` is returned if ``remote`` is true. Otherwise a
        ``LocalSource`` is returned``.
        """
        file = io_from_iso(iso, remote)
        file.close()

        if remote:
            return RemoteSource.parse_obj({'type': 'remote', 'url': file.name})
        else:
            return LocalSource.parse_obj({'type': 'local', 'path': file.name})

    return source_from_iso_


@pytest.fixture
def iso_typical():
    """Fixture providing a function which returns an ISO image in the form of a
    ``PyCdlib`` object containing various files and directories.
    """

    def iso_typical_(*, rock_ridge=False, joliet=False, udf=False):
        """Return an ISO image containing various files and directories.

        :param rock_ridge: If the ISO should support the Rock Ridge ISO extension.
        :param joliet: If the ISO should support the Joliet ISO extension.
        :param udf: If the ISO should support the UDF bridge format.

        All files and directories created will appear in every selected
        extension-specific view.

        Returns a ``tuple`` of (``PyCdlib`` object, directory list, file list
        including data).
        """
        iso = PyCdlib()
        iso.new(
            rock_ridge='1.09' if rock_ridge else None,
            joliet=3 if joliet else None,
            udf='2.60' if udf else None,
        )

        def add_directory(path: PurePosixPath):
            """Add directory ``path_`` to ``iso``."""
            iso_path = str(path).upper()
            rr_name = path.parts[-1]
            iso.add_directory(
                iso_path=iso_path,
                rr_name=rr_name if rock_ridge else None,
                joliet_path=str(path) if joliet else None,
                udf_path=str(path) if udf else None,
                file_mode=RR_DEFAULT_FILE_MODE if rock_ridge else None,
            )

        def add_file(path: PurePosixPath, data: bytes):
            """Add file ``path`` with data ``data`` to ``iso``."""
            iso_path = str(path).upper() + ';1'
            rr_name = path.parts[-1]
            iso.add_fp(
                BytesIO(data),
                len(data),
                iso_path=iso_path,
                rr_name=rr_name if rock_ridge else None,
                joliet_path=str(path) if joliet else None,
                udf_path=str(path) if udf else None,
                file_mode=RR_DEFAULT_FILE_MODE if rock_ridge else None,
            )

        directories = ['/dir1', '/dir2', '/dir2/subdir1', '/dir3']
        files = [
            ('/file1.bin', b'test'),
            ('/dir1/file2.bin', b''),
            ('/dir2/file3.bin', randbytes(443)),
            ('/file.bin', randbytes(CHUNK_SIZE_REMOTE_CHUNKED * 3 + 17)),
            ('/dir2/subdir1/file4.bin', randbytes(443)),
        ]

        for path_ in directories:
            add_directory(PurePosixPath(path_))

        for path_, data_ in files:
            add_file(PurePosixPath(path_), data_)

        return iso, directories, files

    return iso_typical_


@pytest.mark.parametrize('rock_ridge', [False, True])
@pytest.mark.parametrize('joliet', [False, True])
@pytest.mark.parametrize('udf', [False, True])
@pytest.mark.parametrize('remote', [False, True])
def test__extract_file_typical(
    iso_typical, rock_ridge, joliet, udf, io_from_iso, remote, tempdir
):
    """Test that typical data files are extracted as expected."""
    # prepare iso
    iso, directories, files = iso_typical(rock_ridge=rock_ridge, joliet=joliet, udf=udf)

    # reopen iso
    with io_from_iso(iso, remote) as file:
        iso.close()
        iso_reopened = PyCdlib()
        iso_reopened.open_fp(file)

        # extract files
        progress = Value('q', 0)
        resume_event = Event()
        resume_event.set()

        for filepath_iso_str, data in files:
            filepath_iso = PurePosixPath(filepath_iso_str)
            filepath_local_tail = filepath_iso

            if not any((rock_ridge, joliet, udf)):  # pure ISO 9660
                filepath_iso = PurePosixPath(filepath_iso_str.upper() + ';1')
                filepath_local_tail = PurePosixPath(filepath_iso_str.upper())

            filepath_local = tempdir / filepath_local_tail.relative_to('/')

            done = _extract_file(
                ExtractJob(filepath_iso, filepath_local),
                file,
                _get_facade_for_iso(iso_reopened),
                progress,
                Event(),
                resume_event,
            )
            assert done
            assert filepath_local.read_bytes() == data

        assert progress.value == sum(len(file[1]) for file in files)
        iso_reopened.close()

    # check directories
    for directory in directories:
        directory_extracted = directory
        if not any((rock_ridge, joliet, udf)):  # pure ISO 9660
            directory_extracted = directory.upper()

        if any(file_.startswith(directory) for file_, data in files):
            assert (tempdir / Path(directory_extracted).relative_to('/')).is_dir()
        else:
            # empty directories are not extracted
            assert not (tempdir / Path(directory_extracted).relative_to('/')).exists()


@pytest.mark.parametrize('iso_new_args', [{'rock_ridge': '1.09'}, {'udf': '2.60'}])
@pytest.mark.parametrize('remote', [False, True])
def test__extract_file_symlink(iso_new_args, io_from_iso, remote, tempdir):
    """Test that symlinks are extracted (or skipped) as expected."""
    # prepare iso
    iso = PyCdlib()
    iso.new(**iso_new_args)
    iso_facade = _get_facade_for_iso(iso)

    # Rock Ridge facade takes extra arguments
    add_fp = iso_facade.add_fp
    add_directory = iso_facade.add_directory
    if isinstance(iso_facade, PyCdlibRockRidge):
        add_fp = partial(add_fp, file_mode=RR_DEFAULT_FILE_MODE)
        add_directory = partial(add_directory, file_mode=RR_DEFAULT_FILE_MODE)

    # add files and directories
    add_directory('/dir1')

    data = b'test'
    file_path = PurePosixPath('/dir1/file1.bin')
    add_fp(BytesIO(data), len(data), str(file_path))

    # use PurePath for symlink targets because on Windows, symlink targets must use
    # backward slashes instead of forward slashes to work with actually existing files
    symlinks = [
        (PurePosixPath('/link1'), PurePath(file_path.relative_to('/'))),
        (PurePosixPath('/dir1/link2'), PurePath(file_path.relative_to('/dir1'))),
        (PurePosixPath('/link3'), PurePath('path/does/not/exist')),
    ]
    symlink_paths = [path for path, _ in symlinks]
    for symlink_path, symlink_target in symlinks:
        iso_facade.add_symlink(str(symlink_path), str(symlink_target))

    # reopen iso
    with io_from_iso(iso, remote) as file:
        iso.close()
        iso_reopened = PyCdlib()
        iso_reopened.open_fp(file)

        # extract files
        progress = Value('q', 0)
        resume_event = Event()
        resume_event.set()

        for filepath_iso in symlink_paths + [file_path]:
            filepath_local = tempdir / filepath_iso.relative_to('/')
            done = _extract_file(
                ExtractJob(filepath_iso, filepath_local),
                file,
                _get_facade_for_iso(iso_reopened),
                progress,
                Event(),
                resume_event,
            )
            assert done

        assert progress.value == len(data)
        iso_reopened.close()

    # inspect extracted files
    assert not (tempdir / file_path.relative_to('/')).is_symlink()
    assert (tempdir / file_path.relative_to('/')).read_bytes() == b'test'

    if isinstance(iso_facade, PyCdlibRockRidge):
        assert (tempdir / 'link1').is_symlink()
        assert (tempdir / 'link1').read_bytes() == b'test'
        assert (tempdir / 'dir1/link2').is_symlink()
        assert (tempdir / 'dir1/link2').read_bytes() == b'test'
        assert (tempdir / 'link3').is_symlink()
        assert not (tempdir / 'link3').exists()  # exists follows symlinks
    else:  # UDF
        assert not (tempdir / 'link1').is_symlink()
        assert not (tempdir / 'link1').exists()
        assert not (tempdir / 'dir1/link2').is_symlink()
        assert not (tempdir / 'dir1/link2').exists()
        assert not (tempdir / 'link3').is_symlink()
        assert not (tempdir / 'link3').exists()


@pytest.mark.parametrize(
    'iso_new_args', [{}, {'rock_ridge': '1.09'}, {'joliet': 3}, {'udf': '2.60'}]
)
@pytest.mark.parametrize('remote', [False, True])
def test__extract_file_el_torito(iso_new_args, io_from_iso, remote, tempdir):
    """Test that an El Torito boot catalog is skipped during extraction.

    Also test that ``PyCdlibInvalidInput`` is still raised if it's because of a
    reason other than that the requested file is a boot catalog.
    """
    # prepare iso
    iso = PyCdlib()
    iso.new(**iso_new_args)
    iso_facade = _get_facade_for_iso(iso)

    # add boot file and boot catalog
    boot_file_path = PurePosixPath('/BOOT.BIN;1')
    boot_file_path_non_iso = PurePosixPath('/boot.bin')

    # boot file must be visible on pure ISO 9660
    if isinstance(iso_facade, PyCdlibRockRidge):
        iso.add_fp(BytesIO(b'duh'), 3, iso_path=str(boot_file_path), rr_name='boot.bin')
    else:
        iso.add_fp(BytesIO(b'duh'), 3, iso_path=str(boot_file_path))
    iso.add_eltorito(str(boot_file_path))

    if isinstance(iso_facade, PyCdlibISO9660):
        boot_cat_path = PurePosixPath('/BOOT.CAT;1')
    else:
        boot_cat_path = PurePosixPath('/boot.cat')
    boot_catalog_length = iso_facade.get_record(str(boot_cat_path)).get_data_length()

    # add dummy directory
    dirpath = PurePosixPath('/DIR1')
    if isinstance(iso_facade, PyCdlibRockRidge):
        # skipcq: PYL-E1123
        iso_facade.add_directory(str(dirpath), file_mode=RR_DEFAULT_FILE_MODE)
    else:
        iso_facade.add_directory(str(dirpath))

    # reopen iso
    with io_from_iso(iso, remote) as file:
        iso.close()
        iso_reopened = PyCdlib()
        iso_reopened.open_fp(file)

        # try to extract boot catalog
        progress = Value('q', 0)
        resume_event = Event()
        resume_event.set()

        filepath_local = tempdir / boot_cat_path.relative_to('/')
        done = _extract_file(
            ExtractJob(boot_cat_path, filepath_local),
            file,
            _get_facade_for_iso(iso_reopened),
            progress,
            Event(),
            resume_event,
        )
        assert done

        # trying to extract a directory should raise PyCdlibInvalidInput
        dirpath_local = tempdir / dirpath.relative_to('/')

        with pytest.raises(PyCdlibInvalidInput, match='Path to open must be a file'):
            _extract_file(
                ExtractJob(dirpath, dirpath_local),
                file,
                _get_facade_for_iso(iso_reopened),
                progress,
                Event(),
                resume_event,
            )
        assert progress.value == boot_catalog_length
        iso_reopened.close()

    # inspect extracted files
    assert not (tempdir / dirpath.relative_to('/')).exists()
    assert not (tempdir / boot_cat_path.relative_to('/')).exists()
    # not extracted
    assert not (tempdir / boot_file_path_non_iso.relative_to('/')).exists()


class TestExtractProgress:
    """Tests for the named tuple ``ExtractProgress`` and its properties which require
    calculations.
    """

    @staticmethod
    def test_done_ratio():
        """Test that the ``done_ratio`` property handles zero values and other
        unexpected values properly.
        """
        assert ExtractProgress(-1, 0, 0, 0).done_ratio == 1
        assert ExtractProgress(0, 0, 0, 0).done_ratio == 1
        assert ExtractProgress(0, 1, 0, 0).done_ratio == 1
        assert ExtractProgress(1, 0, 0, 0).done_ratio == 0
        assert ExtractProgress(1, 1, 0, 0).done_ratio == 1
        assert ExtractProgress(8, 2, 0, 0).done_ratio == 0.25

    @staticmethod
    def test_bytes_per_second():
        """Test that the ``bytes_per_second`` property handles zero values and other
        unexpected values properly.
        """
        assert ExtractProgress(0, 0, -1, 0).bytes_per_second is None
        assert ExtractProgress(0, 0, 0, 0).bytes_per_second is None
        assert ExtractProgress(0, 0, 0, 1).bytes_per_second is None
        assert ExtractProgress(0, 0, 1, 0).bytes_per_second == 0
        assert ExtractProgress(0, 0, 1, 1).bytes_per_second == 1
        assert ExtractProgress(0, 0, 2, 8).bytes_per_second == 4
        assert ExtractProgress(0, 0, 2, -8).bytes_per_second == -4
        assert ExtractProgress(0, 0, float('inf'), 1).bytes_per_second == 0

    @staticmethod
    def test_seconds_left():
        """Test that the ``seconds_left`` property handles zero values and other
        unexpected values properly.
        """
        assert ExtractProgress(0, 0, 0, 0).seconds_left is None
        assert ExtractProgress(0, 0, 0, 1).seconds_left is None
        assert ExtractProgress(0, 0, 1, 0).seconds_left is None
        assert ExtractProgress(0, 0, 1, 1).seconds_left == 0
        assert ExtractProgress(1, 0, 1, 1).seconds_left == 1
        assert ExtractProgress(1, 1, 1, 1).seconds_left == 0
        assert ExtractProgress(1, 1, 1, -1).seconds_left is None


@pytest.mark.parametrize('rock_ridge', [False, True])
@pytest.mark.parametrize('joliet', [False, True])
@pytest.mark.parametrize('udf', [False, True])
@pytest.mark.parametrize('remote', [False, True])
def test_iso_size_contents(iso_typical, rock_ridge, joliet, udf, io_from_iso, remote):
    """Test that ``iso_size_contents`` correctly calculates the size of ISO image
    contents.
    """
    # prepare ISO
    iso, _, files = iso_typical(rock_ridge=rock_ridge, joliet=joliet, udf=udf)
    iso_facade = _get_facade_for_iso(iso)
    size_expected = sum(len(file[1]) for file in files)

    # boot catalog should add 2048 bytes to the total size
    iso.add_eltorito(files[0][0].upper() + ';1')
    size_expected += 2048

    # symlinks shouldn't add to the total size
    if isinstance(iso_facade, (PyCdlibRockRidge, PyCdlibUDF)):
        symlink_target_abs = PurePosixPath(files[1][0])
        symlink_target_1 = symlink_target_abs.relative_to('/')
        symlink_target_2 = symlink_target_abs.relative_to('/dir1')

        # internal check, we want the symlink targets to actually exist on the image
        assert str(symlink_target_abs).startswith('/dir1/')

        symlinks = [
            (PurePosixPath('/link1'), PurePath(symlink_target_1)),
            (PurePosixPath('/dir1/link2'), PurePath(symlink_target_2)),
            (PurePosixPath('/link3'), PurePath('path/does/not/exist')),
        ]
        for symlink_path, symlink_target in symlinks:
            iso_facade.add_symlink(str(symlink_path), str(symlink_target))

    # reopen iso
    with io_from_iso(iso, remote) as file:
        iso.close()
        iso_reopened = PyCdlib()
        iso_reopened.open_fp(file)
        iso_facade_reopened = _get_facade_for_iso(iso_reopened)
        assert iso_size_contents(iso_facade_reopened) == size_expected


@pytest.mark.parametrize('rock_ridge', [False, True])
@pytest.mark.parametrize('joliet', [False, True])
@pytest.mark.parametrize('udf', [False, True])
@pytest.mark.parametrize('remote', [False, True])
@pytest.mark.parametrize('with_pause', [False, True])
def test_extract_success(
    mocker,
    iso_typical,
    rock_ridge,
    joliet,
    udf,
    source_from_iso,
    remote,
    tempdir,
    with_pause,
):
    """Test that a typical ISO image is extracted by ``extract`` as expected.

    :param with_pause: If the extraction should be paused and resumed once.
    """
    # prepare iso
    iso, directories, files = iso_typical(rock_ridge=rock_ridge, joliet=joliet, udf=udf)

    # get source for iso
    source = source_from_iso(iso, remote)
    iso.close()

    def check_progress(progress: ExtractProgress):
        """Test meaningfulness of an ``ExtractProgress`` object in the context of the
        current extraction.
        """
        assert progress.bytes_total == sum(len(data) for _, data in files)
        assert progress.done_ratio <= 1
        assert progress.bytes_per_second is None or progress.bytes_per_second >= 0
        assert progress.seconds_left is None or progress.seconds_left >= 0

    if with_pause:
        # extract with pause

        # Let the loop "while not all(future.done() for future in futures): ..." run
        # run for a while.
        # This is necessary because we want to guarantee that the extraction process
        # is actually paused at some point. This however is only possible if the first
        # extraction progress yielded is not the last extraction progress because the
        # last yield doesn't check which value is received via send().
        mocker.patch('concurrent.futures._base.Future.done', return_value=False)

        with extract(source, tempdir) as extraction:
            # first progress update
            progress_ = next(extraction)
            if progress_ is not None:
                check_progress(progress_)

            timer_start = time.perf_counter()
            paused = False

            while True:
                try:
                    # pause for two seconds
                    if timer_start <= time.perf_counter() < timer_start + 1:
                        paused = True
                        progress_ = extraction.send(True)
                    else:
                        # pause is over, let main thread break free from while loop
                        # with future.done() checks (see above)
                        mocker.stopall()
                        assert paused
                        progress_ = next(extraction)
                except StopIteration:
                    break

                if progress_ is not None:
                    check_progress(progress_)

            assert paused
            assert time.perf_counter() - timer_start >= 1
    else:
        # extract without pause
        with extract(source, tempdir) as extraction:
            for progress_ in extraction:
                if progress_ is not None:
                    check_progress(progress_)

    # check files and directories
    for directory in directories:
        directory_extracted = directory
        if not any((rock_ridge, joliet, udf)):  # pure ISO 9660
            directory_extracted = directory.upper()

        # check directory
        files_in_directory = [
            (file, data) for (file, data) in files if file.startswith(directory)
        ]
        dir_extracted_path = tempdir / Path(directory_extracted).relative_to('/')

        if files_in_directory:
            assert dir_extracted_path.is_dir()
        else:
            # empty directories are not extracted
            assert not dir_extracted_path.exists()

        # check files
        for file, data in files_in_directory:
            file_extracted = file
            if not any((rock_ridge, joliet, udf)):  # pure ISO 9660
                file_extracted = file.upper()

            file_extracted_path = tempdir / Path(file_extracted).relative_to('/')
            assert file_extracted_path.is_file()
            assert file_extracted_path.read_bytes() == data


@pytest.mark.parametrize('extra_update', [False, True])
@pytest.mark.parametrize('rock_ridge', [False, True])
@pytest.mark.parametrize('joliet', [False, True])
@pytest.mark.parametrize('udf', [False, True])
@pytest.mark.parametrize('remote', [False, True])
def test_extract_abort_generator(
    extra_update, iso_typical, rock_ridge, joliet, udf, source_from_iso, remote, tempdir
):
    """Test that the extraction of an ISO image can be aborted by closing the
    according generator.

    :param extra_update: Whether to wait for one progress update to arrive before
        aborting. (At least the last progress update will definitely happen.)
    """
    iso, _, _ = iso_typical(rock_ridge=rock_ridge, joliet=joliet, udf=udf)
    source = source_from_iso(iso, remote)
    iso.close()

    with extract(source, tempdir) as extraction:
        if extra_update:
            next(extraction)

        extraction.close()
        with pytest.raises(StopIteration):
            next(extraction)


@pytest.mark.parametrize('extra_update', [False, True])
@pytest.mark.parametrize('rock_ridge', [False, True])
@pytest.mark.parametrize('joliet', [False, True])
@pytest.mark.parametrize('udf', [False, True])
@pytest.mark.parametrize('remote', [False, True])
def test_extract_abort_context(
    extra_update, iso_typical, rock_ridge, joliet, udf, source_from_iso, remote, tempdir
):
    """Test that the extraction of an ISO image can be aborted by leaving the
    according context.

    :param extra_update: Whether to wait for one progress update to arrive before
        aborting. (At least the last progress update will definitely happen.)
    """
    iso, _, _ = iso_typical(rock_ridge=rock_ridge, joliet=joliet, udf=udf)
    source = source_from_iso(iso, remote)
    iso.close()

    with extract(source, tempdir) as extraction:
        if extra_update:
            next(extraction)

    with pytest.raises(StopIteration):
        next(extraction)


@pytest.mark.parametrize('remote', [False, True])
def test_extract_fail_target_paths(iso_typical, source_from_iso, remote, tempdir):
    """Test that extraction fails if one of the target paths is not a directory."""
    iso, _, _ = iso_typical()
    source = source_from_iso(iso, remote)
    iso.close()

    not_a_directory = tempdir / 'notadirectory'
    target_paths_list = [
        not_a_directory,
        (not_a_directory, tempdir),
        (tempdir, not_a_directory),
    ]

    for target_paths in target_paths_list:
        # skipcq: PTC-W0062
        with pytest.raises(
            ValueError,
            match=re.escape(f'{not_a_directory.resolve()} is not a directory'),
        ):
            with extract(source, target_paths) as extraction:
                next(extraction)


@pytest.mark.parametrize('remote', [False, True])
def test_extract_fail_source(remote, tempdir):
    """Test that extraction fails if the source passed is unavailable.

    This should lead to an exception being raised in the worker threads which then
    gets passed on to the main thread via the exception queue.
    """
    if remote:
        source = RemoteSource.parse_obj(
            {'type': 'remote', 'url': 'http://example.invalid'}
        )
        cause = RequestsConnectionError
    else:
        source = LocalSource.parse_obj(
            {'type': 'local', 'path': str(tempdir / 'notafile')}
        )
        cause = FileNotFoundError

    # skipcq: PTC-W0062
    with pytest.raises(
        RuntimeError, match='Exception raised in worker thread'
    ) as exc_info:
        with extract(source, tempdir) as extraction:
            for _ in extraction:  # skipcq: PTC-W0047
                pass

    # worker threads were aborted because an exception of type cause was raised
    assert type(exc_info.value.__cause__) is cause


@pytest.mark.parametrize('remote', [False, True])
def test_extract_fail_disk_space(mocker, iso_typical, source_from_iso, remote, tempdir):
    """Test that extraction fails if there is not enough disk space available.

    This should lead to an exception being raised in the main thread before
    extracting anything.
    """
    iso, _, _ = iso_typical()
    source = source_from_iso(iso, remote)
    iso.close()

    disk_usage_ret = namedtuple('usage', ['total', 'used', 'free'])
    # already imported by image module
    mocker.patch('statim.image.disk_usage', return_value=disk_usage_ret(1, 1, 0))

    # skipcq: PTC-W0062
    with pytest.raises(
        ValueError, match='Not enough disk space available at target directories'
    ):
        with extract(source, tempdir) as extraction:
            for _ in extraction:  # skipcq: PTC-W0047
                pass

    assert not list(tempdir.iterdir())  # nothing in tempdir


@pytest.mark.parametrize('remote', [False, True])
def test_extract_fail_worker(mocker, iso_typical, source_from_iso, remote, tempdir):
    """Test that extraction fails if one of the worker threads fails amidst extraction.

    This should lead to an exception being raised in a worker thread *after* the
    actual extraction started which then gets passed on to the main thread via the
    exception queue.
    """
    iso, _, _ = iso_typical()
    source = source_from_iso(iso, remote)
    iso.close()

    class VerySpecificException(Exception):
        """Exception type used for mocking ``_extract_file``."""

    # noinspection PyUnusedLocal
    def _extract_file_mock(job, source_file, iso_facade, progress, *args):
        """Function replacing ``_extract_file``.

        Raises ``VerySpecificException`` as soon as ``progress`` exceeds 0.
        If ``progress`` doesn't exceed 0, ``progress`` is increased by 1.
        """
        if progress.value > 0:
            raise VerySpecificException
        with progress.get_lock():
            progress.value += 1
        return True

    mocker.patch('statim.image._extract_file', wraps=_extract_file_mock)

    # skipcq: PTC-W0062
    with pytest.raises(
        RuntimeError, match='Exception raised in worker thread'
    ) as exc_info:
        with extract(source, tempdir) as extraction:
            for _ in extraction:  # skipcq: PTC-W0047
                pass

    # worker threads were aborted because VerySpecificException was raised
    assert type(exc_info.value.__cause__) is VerySpecificException
