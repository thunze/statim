"""Classes and constants used across the Windows modules."""

from ._base import QuietEnum

__all__ = ['Language', 'Locale', 'Timezone', 'USERNAME_REGEX']


class Language(str, QuietEnum):
    """Languages to choose from when fetching a Windows installation image from
    Microsoft (see ``win*.AutoSource``).
    """

    en_us = 'en-US'


class Locale(str, QuietEnum):
    """Locales to choose from when installing Windows (see ``win*.Unattend``).

    https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-lcid/70feba9f-294e-491e-b6eb-56532684c37f
    """

    en_us = 'en-US'
    de_de = 'de-DE'


class Timezone(str, QuietEnum):
    """Time zones to choose from when installing Windows (see ``win*.Unattend``).

    See ``tzutil /l`` on Windows for a list of time zones.
    """

    w_europe_standard_time = "W. Europe Standard Time"


USERNAME_REGEX = (
    r'(?i)^(?!(aux|con|nul|prn|com[1-9]|lpt|lpt[1-9]?|batch|dialup|proxy'
    r'|defaultaccount|defaultuser0|public|trustedinstaller|wdagutilityaccount)$)'
    r'(?!.*[/\\\[\]:;|=,+*?<>"%@].*$)(?!\..*\.?$).+'
)
"""Eliminates usernames forbidden on Windows."""
