"""Wrapper class and functions to provide the complete data model."""

from typing import Any

from pydantic import schema_json_of

from . import win10
from ._base import BaseModel, Plan

__all__ = ['from_dict', 'from_str', 'json_schema']


JSON_INDENT = 4


class _PlanWrapper(BaseModel):
    """Discriminated union of all ``Plan`` subclasses."""

    # commenting this out until we support more than one OS
    #
    # __root__: Union[win10.Plan, ...] = Field(
    #     ..., discriminator='os', additionalProperties=False
    # )
    __root__: win10.Plan


def from_dict(dict_: dict[str, Any]) -> Plan:
    """Parse an arbitrary ``Plan`` in the form of a ``dict``.

    Example::

      plan.from_dict({
          'os': OS.win10,
          'uefi': True,
          'source': {
              'type': 'local',
              'path': '/path/to/windows_10.iso'
          },
          ...
      })
    """
    return _PlanWrapper.parse_obj(dict_).__root__


def from_str(str_: str) -> Plan:
    """Parse an arbitrary ``Plan`` in the form of a ``str`` containing JSON data."""
    return _PlanWrapper.parse_raw(str_, content_type='json').__root__


def json_schema() -> str:
    """Return the JSON schema of the discriminated union of all ``Plan`` subclasses."""
    return schema_json_of(_PlanWrapper, indent=JSON_INDENT)
