"""Classes and constants used across the Windows modules."""

from enum import Enum

__all__ = ['Bitness', 'Language', 'Locale', 'Timezone', 'USERNAME_REGEX']


class Bitness(Enum):
    """Processor bitnesses supported by Windows."""

    thirtytwo = 32
    sixtyfour = 64


class Language(str, Enum):
    """Languages to choose from when fetching a Windows installation image from
    Microsoft (see ``win*.AutoSource``).
    """

    en_us = "en-US"


class Locale(str, Enum):
    """Locales to choose from when installing Windows (see ``win*.Unattend``).

    See https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-lcid/70feba9f-294e-491e-b6eb-56532684c37f.  # noqa
    """

    en_US = 'en-US'


# locales are valid values for timezones, too
Timezone = Locale


USERNAME_REGEX = (
    r'^(?i)(?!(aux|con|nul|prn|com[1-9]|lpt|lpt[1-9]?'
    r'|batch|dialup|proxy|defaultaccount|defaultuser0|public|trustedinstaller'
    r'|wdagutilityaccount)$)(?!.*[/\\\[\]:;|=,+*?<>"%@].*$)(?!\..*\.?$).+ '
)
