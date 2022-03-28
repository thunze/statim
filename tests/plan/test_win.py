import re

import pytest

from statim.plan.win import USERNAME_REGEX


@pytest.fixture(scope='module')
def username_regex():
    """Only compile regex once to save time."""
    return re.compile(USERNAME_REGEX)


@pytest.mark.parametrize(
    'username',
    [
        'Admin',
        'Chuck',
        'Chuck.',
        'Chuck Norris',
        'Chuck N()rr1z',
        'Chuck Nörriz',
        'CONny',
        'accOmmodation',
        'COM',
        'COM10',
        'LPT0',
        '123',
        '\'',
        '-',
        '!',
        '§',
        '$&',
        '{',
        '}',
        '(',
        ')^',
        '👉👈',
    ],
)
def test_username_regex_success(username, username_regex):
    """Test Windows username regex against valid usernames."""
    assert username_regex.match(username) is not None


@pytest.mark.parametrize(
    'username',
    [
        '',
        'aux',
        'cOn',
        'NUL',
        'COM1',
        'COM9',
        'LpT',
        'LPT5',
        'conin$',
        'conOUT$',
        'batch',
        'dialup',
        'TrustedInstaller',
        '?ayo',
        'what?',
        '/',
        '\\',
        '[',
        ']',
        ':',
        ';',
        '|',
        '=',
        ',',
        '+',
        '*',
        '?',
        '<',
        '>',
        '"',
        '@',
        '%',
        '$%',
        '.',
        '.blah',
        '.Blah.',
    ],
)
def test_username_regex_fail(username, username_regex):
    """Test Windows username regex against invalid usernames."""
    assert username_regex.match(username) is None
