"""Windows 10 ``Plan`` class and related classes."""

from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import Field

from ._base import (
    OS,
    OS_FIELD_KWARGS,
    SOURCE_FIELD_KWARGS,
    UNATTEND_FIELD_KWARGS,
    BaseModel,
    LocalSource,
    PlanWithUnattendSupport,
    RemoteSource,
)
from .win import USERNAME_REGEX, Language, Locale, Timezone


class Version(str, Enum):
    """Windows 10 Versions."""

    v_21h2 = '21h2'


class Edition(str, Enum):
    """Windows 10 Editions."""

    home = 'Home'
    home_n = 'Home N'
    home_single_language = 'Home Single Language'
    education = 'Education'
    education_n = 'Education N'
    pro = 'Pro'
    pro_n = 'Pro N'
    pro_education = 'Pro Education'
    pro_education_n = 'Pro Education N'
    pro_for_workstations = 'Pro for Workstations'
    pro_n_for_workstations = 'Pro N for Workstations'


class AutoSource(BaseModel):
    """Windows 10 installation image automatically fetched from Microsoft servers.

    Resolves to a ``RemoteSource`` when ``resolve_source`` is called.
    """

    type: Literal['auto']
    language: Language = Field(
        ...,
        title='Language',
        description='Language of the Windows 10 installation image to download.',
    )


class Unattend(BaseModel):
    """Additional information required for the preparation of an unattended Windows
    10 installation.
    """

    edition: Edition = Field(
        ...,
        title='Edition',
        description=(
            'Edition of Windows 10 to install. Be aware that the configured source '
            'must support the selected edition.'
        ),
    )

    locale: Locale = Field(
        ...,
        title='Locale',
        description=(
            'Locale to set during the installation of Windows 10. This value '
            'determines the keyboard layout and format settings (e.g. date format). '
            'This value also determines the timezone setting if it is not set '
            'explicitly.'
        ),
    )

    timezone: Timezone = Field(
        None,
        title='Timezone',
        description=(
            'Timezone to set during installation of Windows 10. Defaults to the '
            'default timezone for the specified locale.'
        ),
    )

    # see https://github.com/samuelcolvin/pydantic/issues/975
    username: Annotated[
        str,
        Field(
            title='Username',
            description=(
                'Name of the local account to create when installing Windows 10. This '
                'will also be the name of the user folder of the account '
                '("C:\\Users\\<user>").'
            ),
            min_length=1,
            max_length=63,
            regex=USERNAME_REGEX,
            strip_whitespace=True,
        ),
    ]


class Plan(PlanWithUnattendSupport):
    """Windows 10 ``Plan`` class."""

    os: Literal[OS.win10] = Field(
        ..., title=OS_FIELD_KWARGS.title, description=OS_FIELD_KWARGS.description
    )

    version: Version = Field(
        Version.v_21h2,
        title='Version',
        description='Version of Windows 10 to install. Defaults to the latest final '
        'version of Windows 10. Be aware that the configured source must '
        'support the selected version.',
    )

    thirtytwo_bit: bool = Field(
        False,
        alias='32_bit',
        title='Bitness',
        description='If a 32-bit version of Windows is to be installed. Be aware that '
        'the configured source must support the selected bitness.',
    )

    source: LocalSource | RemoteSource | AutoSource = Field(
        ...,
        discriminator='type',
        additionalProperties=False,
        title=SOURCE_FIELD_KWARGS.title,
        description=SOURCE_FIELD_KWARGS.description,
    )

    unattend: Optional[Unattend] = Field(
        None,
        title=UNATTEND_FIELD_KWARGS.title,
        description=UNATTEND_FIELD_KWARGS.description,
    )

    def resolve_source(self) -> LocalSource | RemoteSource:
        """If an ``AutoSource`` was selected, fetch an according download URL from
        Microsoft and return a ``RemoteSource`` with this URL.
        """
        if isinstance(self.source, AutoSource):
            raise NotImplementedError
        return self.source

    def prepare_unattended_installation(self) -> None:
        """Prepare the drive to install Windows 10 in an unattended fashion if desired.

        If the ``unattend`` field is set, this adds an according ``AutoUnattend.xml``
        file to the drive which acts as an answer file to the Windows 10 installer.
        """
        raise NotImplementedError
