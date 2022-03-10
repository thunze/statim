"""Windows 10 ``Plan`` class and related classes."""

from enum import Enum
from typing import Literal, Optional

from pydantic import Field

from ._base import OS, BaseModel, LocalSource, PlanWithUnattendSupport, RemoteSource
from .win import Bitness, Language, Locale, Timezone


class Version(str, Enum):
    """Windows 10 Versions."""

    v_21h2 = '21h2'


class Edition(str, Enum):
    """Windows 10 Editions."""

    home = "Home"
    home_n = "Home N"
    home_single_language = "Home Single Language"
    education = "Education"
    education_n = "Education N"
    pro = "Pro"
    pro_n = "Pro N"
    pro_education = "Pro Education"
    pro_education_n = "Pro Education N"
    pro_for_workstations = "Pro for Workstations"
    pro_n_for_workstations = "Pro N for Workstations"


class AutoSource(BaseModel):
    """Windows 10 installation image automatically fetched from Microsoft servers.

    Resolves to a ``RemoteSource`` when ``resolve_source`` is called.
    """

    type: Literal['auto']
    language: Language


class Unattend(BaseModel):
    """Additional information required for the preparation of an unattended Windows
    10 installation.
    """

    edition: Edition
    locale: Locale
    timezone: Timezone
    username: str


class Plan(PlanWithUnattendSupport):
    """Windows 10 ``Plan`` class."""

    os: Literal[OS.win10]
    version: Version
    bitness: Bitness
    source: LocalSource | RemoteSource | AutoSource = Field(
        ..., discriminator='type', additionalProperties=False
    )
    unattend: Optional[Unattend]

    def resolve_source(self) -> LocalSource | RemoteSource:
        if isinstance(self.source, AutoSource):
            raise NotImplementedError
        return self.source

    def prepare_unattended_installation(self) -> None:
        raise NotImplementedError
