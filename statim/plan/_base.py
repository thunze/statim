"""Base classes and related classes used across the OS-specific modules."""

from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import AnyHttpUrl
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra, Field

__all__ = [
    'OS',
    'OS_FIELD_KWARGS',
    'SCHEMA_EXTRA_DEFAULT',
    'SOURCE_FIELD_KWARGS',
    'UNATTEND_FIELD_KWARGS',
    'BaseModel',
    'LocalSource',
    'Plan',
    'PlanWithUnattendSupport',
    'QuietEnum',
    'RemoteSource',
]


class OS(str, Enum):
    """Enum of all currently supported operating systems."""

    win10 = 'win10'


SCHEMA_EXTRA_DEFAULT = {'additionalProperties': False}


class BaseModel(PydanticBaseModel):
    """Base class of every pydantic model defined in this package. Only used to
    define the default configuration for pydantic models.
    """

    class Config:
        """Pydantic model configuration."""

        allow_mutation = False
        schema_extra = SCHEMA_EXTRA_DEFAULT
        extra = Extra.forbid


class QuietEnum(Enum):
    """``Enum`` which doesn't expose its ``__name__`` and ``__doc__`` if used as a
    field type in a pydantic model and its schema is exported.

    If the JSON schema of a pydantic field with an ``Enum`` type is created,
    two definitions are made in the schema: One defining the field itself with all
    its kwargs passed to the ``Field`` function and -- included using *allOf* -- the
    schema of the ``Enum`` type. The *title* and *description* values of the ``Enum``
    type schema however are always created using ``__name__`` and ``__doc__`` of the
    ``Enum`` which we want to avoid. We can safely remove these values from the
    schema because the schema of the regarding pydantic field (which kind of acts as
    a wrapper in this case) most likely already has *title* and *description* values
    defined, so auto-completion tools can just use them instead.
    """

    @classmethod
    def __modify_schema__(cls, field_schema: dict[str, Any]) -> None:
        """Remove the *title* and *description* attributes from the schema."""
        del field_schema['title']
        del field_schema['description']


class LocalSource(BaseModel):
    """OS installation image source accessible via a local path."""

    type: Literal['local']

    path: Path = Field(
        ...,
        title='Path',
        description=(
            'Local path to an installation image for the desired operating system. '
            'The referenced image should correspond to additional settings eventually '
            'set in this configuration. For example: "/path/to/image.iso".'
        ),
    )

    class Config:
        """Pydantic model configuration."""

        title = 'Local Source'
        schema_extra = SCHEMA_EXTRA_DEFAULT | {
            'description': (
                'Local source of an installation image for the desired operating '
                'system.'
            )
        }


class RemoteSource(BaseModel):
    """OS installation image source accessible via HTTP."""

    type: Literal['remote']

    url: AnyHttpUrl = Field(
        ...,
        title='URL',
        description=(
            'HTTP URL of an installation image for the desired operating system. Not '
            'constrained to public TLDs or domain names in general. The referenced '
            'image must correspond to additional settings eventually set in this '
            'configuration.'
        ),
    )

    class Config:
        """Pydantic model configuration."""

        title = 'Remote Source'
        schema_extra = SCHEMA_EXTRA_DEFAULT | {
            'description': (
                'HTTP source of an installation image for the desired operating '
                'system.'
            )
        }


class Plan(BaseModel, ABC):
    """Abstract base class of all OS-specific plans."""

    # dummy field for linters, correctly overridden by every subclass
    os: OS

    uefi: bool = Field(
        ...,
        title='UEFI Support',
        description=(
            'If set to true, the resulting drive will be bootable from a UEFI BIOS '
            'instead of a legacy BIOS.'
        ),
    )

    @abstractmethod
    def resolve_source(self) -> LocalSource | RemoteSource:
        """Abstract method to return either a ``LocalSource`` or a ``RemoteSource``.

        This method is introduced because OS-specific modules can define their own
        source types, but every source must eventually boil down to either a
        ``LocalSource`` instance or a ``RemoteSource`` instance.
        """

    def pre_copy(self) -> None:
        """Method invoked *before* writing the required OS installation files to the
        drive. Empty by default.
        """

    def post_copy(self) -> None:
        """Method invoked *after* writing the required OS installation files,
        including possibly required files for an unattended installation, to the
        drive. Empty by default.
        """


class PlanWithUnattendSupport(Plan):
    """Abstract base class of all OS-specific plans supporting unattended
    installation
    """

    # dummy field for linters, correctly overridden by every subclass
    unattend: Optional[Any]

    @abstractmethod
    def prepare_unattended_installation(self) -> None:
        """Abstract method to prepare the drive to install the desired OS in an
        unattended fashion (for example by adding an answer file to the drive).

        This method is invoked after copying the required OS installation files from
        the image to the drive but *before* running ``post_copy``.
        """


_FieldMeta = namedtuple('_FieldMeta', ['title', 'description'])
"""Keyword arguments to pass to pydantic's ``Field`` function when used for the
definition of an overridden field of an OS-specific ``Plan``.
"""

OS_FIELD_KWARGS = _FieldMeta(
    title='Operating System',
    description='Operating system whose installation files to write to the drive.',
)

SOURCE_FIELD_KWARGS = _FieldMeta(
    title='Source',
    description='Source of an installation image for the desired operating system.',
)

UNATTEND_FIELD_KWARGS = _FieldMeta(
    title='Unattended Installation',
    description='Configuration of an unattended installation of the operating system.',
)
