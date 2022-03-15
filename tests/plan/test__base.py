from enum import Enum

import pytest
from pydantic import schema_of

# noinspection PyProtectedMember
from statim.plan._base import QuietEnum


class UsualTestEnum(Enum):
    """A docstring."""

    something = 3


class QuietTestEnum(QuietEnum):
    """Another docstring."""

    very = 4


class QuietTestStrEnum(str, QuietEnum):
    """Yet another docstring."""

    special = 5


@pytest.mark.parametrize('enum', [UsualTestEnum, QuietTestEnum, QuietTestStrEnum])
def test_quiet_enum(enum):
    """Test that ``QuietEnum`` subclasses exported by pydantic have their *title* and
    *description* values removed and usual ``Enum`` subclasses have not.
    """
    assert issubclass(enum, Enum)

    # schema_of creates a pydantic model wrapper around the enum, so we need to unpack
    # the enum definition first
    wrapped_schema = schema_of(enum)
    schema = wrapped_schema['definitions'][enum.__name__]

    if issubclass(enum, QuietEnum):
        assert 'title' not in schema
        assert 'description' not in schema
    else:
        assert schema['title'] == enum.__name__
        assert schema['description'] == enum.__doc__
