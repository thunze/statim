import pytest
from requests.exceptions import HTTPError
from requests_mock import Mocker

# noinspection PyProtectedMember
from statim.image import HttpIO


class TestHttpIO:
    """Tests for ``HttpIO``, a file-like wrapper for files available via HTTP."""

    TEST_URL = 'mock://test.example'
    TEST_HEADER_BASES = [
        {'accept-ranges': 'bytes', 'content-length': '10'},
        {'accept-ranges': 'bytes', 'content-length': '10', 'content-encoding': ''},
    ]

    @pytest.mark.parametrize('header_base', TEST_HEADER_BASES)
    @pytest.mark.parametrize('content_length', [0, 5, 1234])
    def test_init_success(self, header_base, content_length, requests_mock: Mocker):
        """Test initialization using acceptable responses."""
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
        """Test initialization using unacceptable HTTP status codes."""
        requests_mock.head(self.TEST_URL, status_code=status_code)
        with pytest.raises(HTTPError):
            HttpIO(self.TEST_URL)

    @pytest.mark.parametrize('header_base', TEST_HEADER_BASES)
    @pytest.mark.parametrize('accept_ranges', [None, '', 'none', 'None'])
    def test_init_fail_accept_ranges(
        self, header_base, accept_ranges, requests_mock: Mocker
    ):
        """Test initialization using responses without or with unacceptable values
        for the `Accept-Range` HTTP header.
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
        """Test initialization using responses with unacceptable values for the
        `Content-Encoding` HTTP header.
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
        """Test initialization using responses which don't define a `Content-Length`
        HTTP header.
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
