import os
import tempfile
import unittest
from types import SimpleNamespace

os.environ["HOME"] = tempfile.gettempdir()

from services.institutional_ml_db import InstitutionalMLDatabase


class TestInstitutionalSignalScoring(unittest.TestCase):
    def _build_db(self) -> InstitutionalMLDatabase:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        return InstitutionalMLDatabase(db_path=f"{tmp_dir.name}/inst_signal_scoring.db")

    @staticmethod
    def _base_row(**overrides):
        defaults = dict(
            iv30_rv30_ratio=1.28,
            price_momentum_5d=0.012,
            rsi_14=51.0,
            bb_position=0.52,
            volume_ratio_10d=1.15,
            vix_level=20.0,
            vol_term_structure_slope=0.10,
            volume=6_000_000,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_setup_score_prefers_target_event_timing(self):
        db = self._build_db()

        row = self._base_row()
        near_score = db._score_setup_quality(
            row=row,
            days_to_earnings=6,
            target_entry_dte=6,
            entry_dte_band=6,
        )
        far_score = db._score_setup_quality(
            row=row,
            days_to_earnings=24,
            target_entry_dte=6,
            entry_dte_band=6,
        )

        self.assertGreater(near_score, far_score)

    def test_setup_score_penalizes_crowded_flow(self):
        db = self._build_db()

        balanced = self._base_row(volume_ratio_10d=1.2)
        crowded = self._base_row(volume_ratio_10d=4.8)

        balanced_score = db._score_setup_quality(
            row=balanced,
            days_to_earnings=6,
            target_entry_dte=6,
            entry_dte_band=6,
        )
        crowded_score = db._score_setup_quality(
            row=crowded,
            days_to_earnings=6,
            target_entry_dte=6,
            entry_dte_band=6,
        )

        self.assertGreater(balanced_score, crowded_score)

    def test_rank_candidate_responds_to_crush_quality(self):
        db = self._build_db()

        setup_score = 0.66
        strong_context = {
            "confidence": 0.86,
            "signal_strength": 0.81,
            "magnitude": 0.23,
            "edge_score": 0.11,
        }
        weak_context = {
            "confidence": 0.28,
            "signal_strength": 0.22,
            "magnitude": 0.05,
            "edge_score": 0.01,
        }

        strong_rank = db._rank_candidate_for_alpha(
            setup_score=setup_score,
            crush_context=strong_context,
            days_to_earnings=6,
            target_entry_dte=6,
            entry_dte_band=6,
        )
        weak_rank = db._rank_candidate_for_alpha(
            setup_score=setup_score,
            crush_context=weak_context,
            days_to_earnings=6,
            target_entry_dte=6,
            entry_dte_band=6,
        )

        self.assertGreater(strong_rank, weak_rank)

    def test_setup_score_handles_missing_nullable_fields(self):
        db = self._build_db()

        sparse_row = self._base_row(
            vol_term_structure_slope=None,
            volume=None,
            rsi_14=None,
            bb_position=None,
        )
        score = db._score_setup_quality(
            row=sparse_row,
            days_to_earnings=6,
            target_entry_dte=6,
            entry_dte_band=6,
        )
        self.assertTrue(0.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main()
