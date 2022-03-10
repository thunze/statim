"""Configuration classes describing how statim should create a desired bootable drive.

This package provides ``Plan`` subclasses for all available operating systems and
related classes and functions. A ``Plan`` subclass defines which parameters can be
set to create a bootable drive for the purpose of installing an operating system. It
also defines OS-specific methods required to create a bootable drive supporting
specific functionality like unattended installation.

Plans are parsed using the ``from_dict`` or the ``from_str`` function. For example::

  plan.from_dict({
      'os': OS.win10,
      'uefi': True,
      'source': {
          'type': 'local',
          'path': '/path/to/windows_10.iso'
      },
      ...
  })

The according JSON schema can be exported using the ``json_schema`` function.
"""

# from ._base import OS, Plan, PlanWithUnattendSupport
# from ._wrapper import *
