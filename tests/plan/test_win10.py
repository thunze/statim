from statim.plan.win import Locale, Timezone
from statim.plan.win10 import Edition, Unattend


def test_timezone_eq_locale_if_none():
    """Test that ``timezone`` is inferred from ``locale`` if ``timezone`` is not set."""
    other = {'edition': Edition.home, 'username': 'chuck'}

    # no timezone set
    unattend = Unattend(locale=Locale.en_us, **other)
    assert unattend.timezone == unattend.locale
    assert unattend.timezone is not None

    # timezone set, no change expected
    unattend = Unattend(
        locale=Locale.en_us, timezone=Timezone.w_europe_standard_time, **other
    )
    assert unattend.timezone == Timezone.w_europe_standard_time

    unattend = Unattend(locale=Locale.en_us, timezone=Locale.de_de, **other)
    assert unattend.timezone == Locale.de_de
