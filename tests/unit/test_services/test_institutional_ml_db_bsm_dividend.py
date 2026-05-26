"""Tests for the dividend-yield extension of the BSM pricer and IV solver in
services.institutional_ml_db.

The pricer follows Merton (1973):

    d1 = (ln(S/K) + (r - q + σ²/2)·T) / (σ·√T)
    Call = S·exp(-q·T)·N(d1) − K·exp(-r·T)·N(d2)
    Put  = K·exp(-r·T)·N(-d2) − S·exp(-q·T)·N(-d1)

These tests pin three contracts:
1. q=0 default reproduces legacy zero-dividend behavior exactly.
2. q > 0 reduces call prices and raises put prices (the right direction).
3. Put-call parity holds under the dividend-aware pricer:
       C − P = S·exp(-q·T) − K·exp(-r·T)
4. The IV solver round-trips: pricing at σ then inverting at the resulting
   price recovers σ within tolerance, both with and without q.
"""
import math
import unittest

from services.institutional_ml_db import InstitutionalMLDatabase


# Reference inputs used across tests.
S = 100.0
K = 100.0
T = 30.0 / 365.0
R = 0.045
SIGMA = 0.30


class TestDividendYieldBackwardCompat(unittest.TestCase):
    """q=0 must reproduce legacy zero-dividend behavior to bit-equality."""

    def test_call_q0_matches_legacy_explicit_zero(self):
        legacy_default = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, True
        )
        explicit_zero = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, True, dividend_yield=0.0
        )
        self.assertAlmostEqual(legacy_default, explicit_zero, places=12)

    def test_put_q0_matches_legacy_explicit_zero(self):
        legacy_default = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, False
        )
        explicit_zero = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, False, dividend_yield=0.0
        )
        self.assertAlmostEqual(legacy_default, explicit_zero, places=12)


class TestDividendDirection(unittest.TestCase):
    """A positive dividend yield must reduce the call value and raise the put
    value (the standard direction — dividends help the put holder, hurt the
    call holder)."""

    def test_call_decreases_with_dividend(self):
        call_no_div = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, True, dividend_yield=0.0
        )
        call_with_div = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, True, dividend_yield=0.04
        )
        self.assertLess(call_with_div, call_no_div)

    def test_put_increases_with_dividend(self):
        put_no_div = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, False, dividend_yield=0.0
        )
        put_with_div = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, False, dividend_yield=0.04
        )
        self.assertGreater(put_with_div, put_no_div)


class TestPutCallParity(unittest.TestCase):
    """Generalised put-call parity under continuous dividend yield:
           C − P = S·exp(-q·T) − K·exp(-r·T)
    Must hold for any (q, r, σ, T) inside the model's domain."""

    def _assert_parity(self, *, q: float, places: int = 8) -> None:
        call = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, True, dividend_yield=q
        )
        put = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, False, dividend_yield=q
        )
        expected = S * math.exp(-q * T) - K * math.exp(-R * T)
        self.assertAlmostEqual(call - put, expected, places=places)

    def test_parity_q0(self):
        self._assert_parity(q=0.0)

    def test_parity_q1_pct(self):
        self._assert_parity(q=0.01)

    def test_parity_high_yield(self):
        # KO-like 3% yield — large enough that the q=0 model breaks parity
        # by a few cents at ATM.
        self._assert_parity(q=0.03)


class TestImpliedVolRoundtrip(unittest.TestCase):
    """Pricing at σ then inverting at the resulting price must recover σ."""

    def test_roundtrip_no_dividend(self):
        price = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, True
        )
        recovered = InstitutionalMLDatabase._implied_volatility_from_price(
            price, S, K, T, R, True
        )
        self.assertAlmostEqual(recovered, SIGMA, places=4)

    def test_roundtrip_with_dividend_call(self):
        price = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, True, dividend_yield=0.03
        )
        recovered = InstitutionalMLDatabase._implied_volatility_from_price(
            price, S, K, T, R, True, dividend_yield=0.03
        )
        self.assertAlmostEqual(recovered, SIGMA, places=4)

    def test_roundtrip_with_dividend_put(self):
        price = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, False, dividend_yield=0.03
        )
        recovered = InstitutionalMLDatabase._implied_volatility_from_price(
            price, S, K, T, R, False, dividend_yield=0.03
        )
        self.assertAlmostEqual(recovered, SIGMA, places=4)

    def test_iv_with_q_differs_from_iv_without_q_for_same_price(self):
        """Same market price, different q assumptions → different IV. If the
        IV solver ignored q, this would falsely yield the same number."""
        price = InstitutionalMLDatabase._black_scholes_price(
            S, K, T, R, SIGMA, True, dividend_yield=0.04
        )
        iv_ignoring_div = InstitutionalMLDatabase._implied_volatility_from_price(
            price, S, K, T, R, True, dividend_yield=0.0
        )
        iv_aware_of_div = InstitutionalMLDatabase._implied_volatility_from_price(
            price, S, K, T, R, True, dividend_yield=0.04
        )
        # Dividend-ignoring IV understates the true vol (the price looks
        # "cheap" without the dividend discount); aware IV recovers it.
        self.assertAlmostEqual(iv_aware_of_div, SIGMA, places=4)
        self.assertLess(iv_ignoring_div, iv_aware_of_div - 0.01)


class TestCalendarSpreadWithDividend(unittest.TestCase):
    """The calendar-spread valuation accepts a dividend_yield kwarg and
    propagates it to both BSM legs."""

    def test_dividend_yield_argument_is_accepted_with_default_zero(self):
        from datetime import datetime

        db = InstitutionalMLDatabase.__new__(InstitutionalMLDatabase)
        value_default = db._calendar_spread_market_value_from_snapshot(
            underlying_price=100.0,
            strike=100.0,
            as_of_date=datetime(2026, 4, 20),
            short_expiry="2026-04-25",
            long_expiry="2026-05-23",
            front_iv=0.45,
            back_iv=0.30,
            risk_free_rate=0.045,
        )
        value_explicit_zero = db._calendar_spread_market_value_from_snapshot(
            underlying_price=100.0,
            strike=100.0,
            as_of_date=datetime(2026, 4, 20),
            short_expiry="2026-04-25",
            long_expiry="2026-05-23",
            front_iv=0.45,
            back_iv=0.30,
            risk_free_rate=0.045,
            dividend_yield=0.0,
        )
        self.assertAlmostEqual(value_default, value_explicit_zero, places=10)

    def test_calendar_value_differs_with_dividend(self):
        """For a high-yield name (3%), the calendar spread value shifts
        materially because both legs are discounted by exp(-qT) at different
        T's. The difference depends on direction of the term structure."""
        from datetime import datetime

        db = InstitutionalMLDatabase.__new__(InstitutionalMLDatabase)
        value_no_div = db._calendar_spread_market_value_from_snapshot(
            underlying_price=100.0,
            strike=100.0,
            as_of_date=datetime(2026, 4, 20),
            short_expiry="2026-04-25",
            long_expiry="2026-05-23",
            front_iv=0.45,
            back_iv=0.30,
            risk_free_rate=0.045,
            dividend_yield=0.0,
        )
        value_with_div = db._calendar_spread_market_value_from_snapshot(
            underlying_price=100.0,
            strike=100.0,
            as_of_date=datetime(2026, 4, 20),
            short_expiry="2026-04-25",
            long_expiry="2026-05-23",
            front_iv=0.45,
            back_iv=0.30,
            risk_free_rate=0.045,
            dividend_yield=0.03,
        )
        # Both finite, but not equal — confirms q flows through to BSM.
        self.assertTrue(value_no_div == value_no_div)  # not NaN
        self.assertTrue(value_with_div == value_with_div)
        self.assertNotAlmostEqual(value_no_div, value_with_div, places=2)


if __name__ == "__main__":
    unittest.main()
