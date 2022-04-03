import io
from random import randbytes

import pytest
from requests import Request
from requests.exceptions import HTTPError
from requests_mock import Mocker

# noinspection PyProtectedMember
from statim.image import HttpIO


class TestHttpIO:
    """Tests for ``HttpIO``, a file-like wrapper for files available via HTTP."""

    TEST_URL = 'mock://test.example'
    TEST_HEADER_BASES = [
        {'accept-ranges': 'bytes', 'content-length': '16'},
        {'accept-ranges': 'bytes', 'content-length': '16', 'content-encoding': ''},
    ]
    TEST_DATA = [b'', b'test', b'The knots will tie you down! ' * 7, randbytes(443)]

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
            self.TEST_URL,
            status_code=200,
            headers={'accept-ranges': 'bytes', 'content-length': str(len(content))},
        )
        return HttpIO(self.TEST_URL)

    @pytest.fixture
    def http_io(self, request, requests_mock: Mocker):
        """Fixture preparing an ``HttpIO`` object for testing purposes by mocking
        responses to `HEAD` and `GET` requests.

        The data the mocked endpoint should provide can be set using indirect
        parametrization.

        Returns the prepared ``HttpIO`` object.
        """
        content = request.param
        requests_mock.head(
            self.TEST_URL,
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
            self.TEST_URL, headers={'accept-ranges': 'bytes'}, content=content_callback
        )
        return HttpIO(self.TEST_URL)

    # Tests

    @pytest.mark.parametrize('header_base', TEST_HEADER_BASES)
    @pytest.mark.parametrize('content_length', [0, 5, 1234])
    def test_init_success(self, header_base, content_length, requests_mock: Mocker):
        """Test initialization using acceptable `HEAD` responses."""
        requests_mock.head(
            self.TEST_URL,
            status_code=200,
            headers=header_base | {'content-length': str(content_length)},
        )
        http_io = HttpIO(self.TEST_URL)

        assert http_io._url == self.TEST_URL
        assert http_io._session.headers['accept-encoding'] == 'identity'
        assert http_io._length == content_length
        assert http_io._pos == 0

    @pytest.mark.parametrize('status_code', [400, 404, 500])
    def test_init_fail_status(self, status_code, requests_mock: Mocker):
        """Test initialization using `HEAD` responses with unacceptable HTTP status
        codes.
        """
        requests_mock.head(self.TEST_URL, status_code=status_code)
        with pytest.raises(HTTPError):
            HttpIO(self.TEST_URL)

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
        requests_mock.head(self.TEST_URL, status_code=200, headers=headers)
        with pytest.raises(OSError, match='Server does not support range requests'):
            HttpIO(self.TEST_URL)

    @pytest.mark.parametrize('header_base', TEST_HEADER_BASES)
    @pytest.mark.parametrize('content_encoding', ['gzip', 'deflate, gzip'])
    def test_init_fail_content_encoding(
        self, header_base, content_encoding, requests_mock: Mocker
    ):
        """Test initialization using `HEAD` responses with unacceptable values for the
        `Content-Encoding` header.
        """
        requests_mock.head(
            self.TEST_URL,
            status_code=200,
            headers=header_base | {'content-encoding': content_encoding},
        )
        with pytest.raises(
            OSError, match='Range requests are not supported for encoded content'
        ):
            HttpIO(self.TEST_URL)

    @pytest.mark.parametrize('header_base', TEST_HEADER_BASES)
    def test_init_fail_content_length(self, header_base, requests_mock: Mocker):
        """Test initialization using `HEAD` responses without a `Content-Length`
        header.
        """
        headers = header_base.copy()
        del headers['content-length']
        requests_mock.head(
            self.TEST_URL,
            status_code=200,
            headers=headers,
        )
        with pytest.raises(
            OSError, match='Server does not provide a \'Content-Length\' header'
        ):
            HttpIO(self.TEST_URL)

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
        assert http_io.readable() is True
        assert http_io.writable() is False
        assert http_io.seekable() is True

        assert http_io.name == self.TEST_URL
        assert repr(http_io) == f'HttpIO(url={self.TEST_URL!r})'

    @pytest.mark.parametrize(
        ['http_io', 'data'], zip(TEST_DATA, TEST_DATA), indirect=['http_io']
    )
    def test_write(self, http_io: HttpIO, data):
        """Test that ``write`` always raises ``io.UnsupportedOperation``."""
        with pytest.raises(io.UnsupportedOperation):
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
            self.TEST_URL, status_code=status_code, headers=headers, content=data
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
            self.TEST_URL,
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
            http_io.seek(-123546, io.SEEK_SET)
        assert http_io.seek(0, io.SEEK_SET) == 0
        assert http_io.tell() == 0
        assert http_io.seek(1, io.SEEK_SET) == 1
        assert http_io.tell() == 1
        assert http_io.seek(len(data) // 2, io.SEEK_SET) == len(data) // 2
        assert http_io.tell() == len(data) // 2

        # SEEK_CUR
        assert http_io.seek(1, io.SEEK_CUR) == len(data) // 2 + 1
        assert http_io.seek(-1, io.SEEK_CUR) == len(data) // 2
        assert http_io.tell() == len(data) // 2

        assert http_io.seek(0, io.SEEK_SET) == 0  # reset
        assert http_io.seek(0, io.SEEK_CUR) == 0
        assert http_io.seek(-1, io.SEEK_CUR) == 0
        assert http_io.tell() == 0

        # SEEK_END
        assert http_io.seek(-len(data), io.SEEK_END) == 0
        assert http_io.tell() == 0
        assert http_io.seek(-len(data) - 10, io.SEEK_END) == 0
        assert http_io.tell() == 0
        assert http_io.seek(0, io.SEEK_END) == len(data)
        assert http_io.tell() == len(data)

        # new pos > len(data)
        assert http_io.seek(len(data) + 1, io.SEEK_SET) == len(data) + 1
        assert http_io.tell() == len(data) + 1

        assert http_io.seek(0, io.SEEK_SET) == 0  # reset
        assert http_io.tell() == 0

        if len(data) > 0:
            assert http_io.seek(len(data) - 1, io.SEEK_SET) == len(data) - 1
            assert http_io.tell() == len(data) - 1

            assert http_io.seek(-1, io.SEEK_END) == len(data) - 1
            assert http_io.seek(1, io.SEEK_CUR) == len(data)
            assert http_io.seek(10, io.SEEK_CUR) == len(data) + 10

        assert http_io.seek(0, io.SEEK_SET) == 0  # reset
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
