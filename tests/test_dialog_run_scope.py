"""Cover the single-dialog slot lifecycle across full runs and fragment reruns.

Regression coverage for the production incident where every dialog stopped opening
after the first one. `_dialog_opened_this_run` was reset only at top-level `app.py`
scope, so a `@st.fragment` rerun never cleared it, and dialogs using the default
``on_dismiss="ignore"`` do not rerun on close either. The flag latched ``True`` and
silently swallowed every later dialog.
"""

import unittest
from unittest.mock import patch

from nhl import ui_state
from nhl.ui_state import (
    DIALOG_OPENED_THIS_RUN_SESSION_KEY,
    begin_dialog_run,
    begin_script_run,
    dialog_slot_available,
    mark_dialog_opened_this_run,
)


class DialogRunScopeTests(unittest.TestCase):
    """Verify the slot is released per rerun without breaking single-dialog arbitration."""

    def setUp(self):
        """Patch Streamlit session state with a plain dict.

        Args:
            None.

        Returns:
            None.
        """
        self.session_state = {}
        patcher = patch.object(ui_state.st, "session_state", self.session_state, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_full_run_releases_the_slot(self):
        """A new script run always clears a slot left reserved by the previous run.

        Args:
            None.

        Returns:
            None.
        """
        mark_dialog_opened_this_run()
        self.assertFalse(dialog_slot_available())

        begin_script_run()

        self.assertTrue(dialog_slot_available())

    def test_fragment_rerun_releases_a_latched_slot(self):
        """The reported bug: a fragment reruns and must not inherit a stale reservation.

        Args:
            None.

        Returns:
            None.
        """
        begin_script_run()
        begin_dialog_run("chart")          # first execution under this run token
        mark_dialog_opened_this_run()      # a dialog opened
        self.assertFalse(dialog_slot_available())

        # No new script run: the fragment reruns on its own (dialog dismissed with
        # on_dismiss="ignore", or the user clicked again).
        begin_dialog_run("chart")

        self.assertTrue(dialog_slot_available())

    def test_later_fragment_does_not_steal_an_earlier_reservation(self):
        """On a full run, fragments execute in sequence and must not clobber each other.

        Resetting unconditionally in every fragment would let two dialogs open in one
        run, which Streamlit does not allow.

        Args:
            None.

        Returns:
            None.
        """
        begin_script_run()

        begin_dialog_run("chart")
        mark_dialog_opened_this_run()      # the chart took the slot

        begin_dialog_run("detail_tabs")    # runs later in the same script run
        self.assertFalse(dialog_slot_available())

        begin_dialog_run("predictions")
        self.assertFalse(dialog_slot_available())

    def test_slot_is_free_again_on_the_next_full_run(self):
        """Reservations never survive into the following script run.

        Args:
            None.

        Returns:
            None.
        """
        begin_script_run()
        begin_dialog_run("chart")
        mark_dialog_opened_this_run()

        begin_script_run()
        begin_dialog_run("chart")

        self.assertTrue(dialog_slot_available())

    def test_repeated_fragment_reruns_stay_usable(self):
        """Open/close cycles inside one fragment must keep working indefinitely.

        Args:
            None.

        Returns:
            None.
        """
        begin_script_run()
        begin_dialog_run("detail_tabs")

        for _ in range(5):
            self.assertTrue(dialog_slot_available())
            mark_dialog_opened_this_run()
            self.assertFalse(dialog_slot_available())
            begin_dialog_run("detail_tabs")

        self.assertTrue(dialog_slot_available())

    def test_begin_script_run_clears_the_flag_directly(self):
        """The public helper owns the key; nothing should set it by hand.

        Args:
            None.

        Returns:
            None.
        """
        self.session_state[DIALOG_OPENED_THIS_RUN_SESSION_KEY] = True

        begin_script_run()

        self.assertIs(self.session_state[DIALOG_OPENED_THIS_RUN_SESSION_KEY], False)


class BlockedClickIsNotConsumedTests(unittest.TestCase):
    """A click blocked by a busy slot must stay retriable, not be silently destroyed."""

    def test_identity_card_click_retries_after_the_slot_frees(self):
        """The identity path burned its nonce before the gate, losing the click.

        Args:
            None.

        Returns:
            None.
        """
        import nhl.comparison as comparison_module

        session_state = {"_dialog_opened_this_run": True}
        trigger = "player:8478402|nonce-1"

        with patch.object(
            comparison_module, "show_player_identity_details"
        ) as mock_show, patch.object(
            comparison_module.st, "session_state", session_state, create=True
        ):
            blocked = comparison_module._show_identity_card_from_trigger(trigger)
            self.assertFalse(blocked)
            mock_show.assert_not_called()
            # The nonce must NOT have been recorded as handled.
            self.assertIsNone(session_state.get("_last_identity_card_trigger_nonce"))

            # Slot frees on the next fragment rerun; the same click now works.
            session_state["_dialog_opened_this_run"] = False
            retried = comparison_module._show_identity_card_from_trigger(trigger)

        self.assertTrue(retried)
        mock_show.assert_called_once_with(8478402)
        self.assertEqual(session_state["_last_identity_card_trigger_nonce"], "nonce-1")

    def test_identity_card_click_still_dedupes_within_a_run(self):
        """Replaying the same nonce after it was handled must not reopen the dialog.

        Args:
            None.

        Returns:
            None.
        """
        import nhl.comparison as comparison_module

        session_state = {"_dialog_opened_this_run": False}
        trigger = "player:8478402|nonce-1"

        with patch.object(
            comparison_module, "show_player_identity_details"
        ) as mock_show, patch.object(
            comparison_module.st, "session_state", session_state, create=True
        ):
            first = comparison_module._show_identity_card_from_trigger(trigger)
            session_state["_dialog_opened_this_run"] = False
            duplicate = comparison_module._show_identity_card_from_trigger(trigger)

        self.assertTrue(first)
        self.assertFalse(duplicate)
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
