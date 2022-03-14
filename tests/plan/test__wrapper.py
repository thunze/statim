import json

import pydantic
import pytest

from statim.plan import (
    OS,
    Plan,
    PlanWithUnattendSupport,
    from_dict,
    from_str,
    json_schema,
    win10,
)


@pytest.mark.parametrize(
    'model',
    [
        {
            'os': OS.win10,
            'uefi': True,
            'version': win10.Version.v_21h2,
            'source': {'type': 'auto', 'language': win10.Language.en_us},
            'unattend': {
                'edition': win10.Edition.pro,
                'locale': win10.Locale.en_us,
                'timezone': win10.Timezone.w_europe_standard_time,
                'username': 'chuck',
            },
        }
    ],
)
def test_parsing_success(model):
    """Test parsing of valid models. Especially check if the discriminated union of
    ``Plan`` subclasses works as expected.
    """
    plan = from_dict(model)
    assert plan == from_str(json.dumps(model))
    assert isinstance(plan, Plan)
    assert isinstance(plan, PlanWithUnattendSupport)
    assert isinstance(plan, win10.Plan)
    assert plan.os == OS.win10
    assert plan.os == 'win10'


@pytest.mark.parametrize(
    'model',
    [
        {},  # empty
        {'os': OS.win10},  # not enough fields
        {'os': 'win10'},
        {'os': 'not_an_os'},
        {'os': OS.win10, 'not_a_field': 'whaddup'},  # extra field
        {
            'os': OS.win10,
            'uefi': True,
            'source': {
                'type': 'auto',
                'language': win10.Language.en_us,
                'not_a_field': 'whaddup',  # extra nested field
            },
        },
    ],
)
def test_parsing_fail(model):
    """Test parsing of a few important invalid models."""
    with pytest.raises(pydantic.ValidationError):
        from_dict(model)


def test_json_schema():
    """Check if the JSON schema is exported correctly and has a custom title."""
    as_dict = json.loads(json_schema())
    assert as_dict['title'] != 'ParsingModel[_PlanWrapper]'
