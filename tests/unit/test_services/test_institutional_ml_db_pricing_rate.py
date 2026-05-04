"""
Tests pinning the explicit-`risk_free_rate` requirement on
services.institutional_ml_db's BSM-style pricing methods.

P-4 removed the `0.03` defaults from `_resolve_row_implied_volatility` and
`_calendar_spread_market_value_from_snapshot`. Callers must now pass the rate
explicitly (typically resolved via services.pricing_rates).
"""
import unittest
from datetime import datetime
from inspect import signature

from services.institutional_ml_db import InstitutionalMLDatabase


class TestRiskFreeRateIsExplicitlyRequired(unittest.TestCase):
    def test_resolve_row_implied_volatility_requires_keyword_rate(self):
        """`_resolve_row_implied_volatility(row, S)` (no rate) must raise TypeError.

        After P-4 the parameter has no default and is keyword-only.
        """
        db = InstitutionalMLDatabase.__new__(InstitutionalMLDatabase)
        with self.assertRaises(TypeError):
            db._resolve_row_implied_volatility(  # type: ignore[call-arg]
                row=None,
                underlying_price=100.0,
            )

    def test_calendar_spread_market_value_requires_rate(self):
        """`_calendar_spread_market_value_from_snapshot` requires risk_free_rate."""
        db = InstitutionalMLDatabase.__new__(InstitutionalMLDatabase)
        with self.assertRaises(TypeError):
            db._calendar_spread_market_value_from_snapshot(  # type: ignore[call-arg]
                underlying_price=100.0,
                strike=100.0,
                as_of_date=datetime(2026, 4, 20),
                short_expiry="2026-04-25",
                long_expiry="2026-05-23",
                front_iv=0.45,
                back_iv=0.30,
            )

    def test_resolve_row_implied_volatility_signature_has_no_default(self):
        sig = signature(InstitutionalMLDatabase._resolve_row_implied_volatility)
        param = sig.parameters["risk_free_rate"]
        self.assertIs(
            param.default,
            param.empty,
            msg="risk_free_rate must be a required parameter (no default)",
        )

    def test_calendar_spread_signature_has_no_default(self):
        sig = signature(InstitutionalMLDatabase._calendar_spread_market_value_from_snapshot)
        param = sig.parameters["risk_free_rate"]
        self.assertIs(
            param.default,
            param.empty,
            msg="risk_free_rate must be a required parameter (no default)",
        )


class TestCalendarSpreadAcceptsExplicitRate(unittest.TestCase):
    """Sanity: the function still produces a finite price when the rate is supplied."""

    def test_returns_finite_value_with_explicit_rate(self):
        db = InstitutionalMLDatabase.__new__(InstitutionalMLDatabase)
        value = db._calendar_spread_market_value_from_snapshot(
            underlying_price=100.0,
            strike=100.0,
            as_of_date=datetime(2026, 4, 20),
            short_expiry="2026-04-25",
            long_expiry="2026-05-23",
            front_iv=0.45,
            back_iv=0.30,
            risk_free_rate=0.045,
        )
        # Calendar (long back, short near) at ATM is positive when back IV >= front IV
        # contributes — but here front IV is higher (event premium), so the calendar
        # is still net-positive (back vega dominates over remaining time). Just
        # require a finite, non-zero number — exact magnitude is exercised by the
        # snapshot integration tests.
        self.assertTrue(value == value, msg="value should not be NaN")  # NaN-safe check
        self.assertGreater(value, 0.0)


if __name__ == "__main__":
    unittest.main()
