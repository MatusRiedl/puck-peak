"""Session-state helpers for chart and dialog rerun orchestration."""

from __future__ import annotations

import streamlit as st


DIALOG_OPENED_THIS_RUN_SESSION_KEY = "_dialog_opened_this_run"
DIALOG_RUN_TOKEN_SESSION_KEY = "_dialog_run_token"
DIALOG_SCOPE_TOKEN_KEY_PREFIX = "_dialog_scope_token_"


def _get_session_state():
    """Return Streamlit's session state proxy or a patched test double."""
    return getattr(st, "session_state", None)


def session_state_get(key: str, default=None):
    """Read one session-state value from either a mapping or attribute proxy."""
    session_state = _get_session_state()
    if session_state is None:
        return default

    if hasattr(session_state, "get"):
        try:
            return session_state.get(key, default)
        except Exception:
            pass

    return getattr(session_state, key, default)


def session_state_set(key: str, value) -> None:
    """Write one session-state value to either a mapping or attribute proxy."""
    session_state = _get_session_state()
    if session_state is None:
        return

    try:
        session_state[key] = value
        return
    except Exception:
        pass

    try:
        setattr(session_state, key, value)
    except Exception:
        pass


def session_state_pop(key: str, default=None):
    """Pop one session-state value from either a mapping or attribute proxy."""
    session_state = _get_session_state()
    if session_state is None:
        return default

    if hasattr(session_state, "pop"):
        try:
            return session_state.pop(key, default)
        except Exception:
            pass

    value = getattr(session_state, key, default)
    if hasattr(session_state, key):
        try:
            delattr(session_state, key)
        except Exception:
            pass
    return value


def begin_script_run() -> None:
    """Release the dialog slot and open a new run token. Call once from `app.py`.

    The token lets each fragment tell a fresh full script run (where this function
    already cleared the slot) apart from a fragment-scoped rerun (where `app.py` never
    executed and the fragment must clear the slot itself).
    """
    session_state_set(DIALOG_OPENED_THIS_RUN_SESSION_KEY, False)
    try:
        token = int(session_state_get(DIALOG_RUN_TOKEN_SESSION_KEY, 0) or 0)
    except (TypeError, ValueError):
        token = 0
    session_state_set(DIALOG_RUN_TOKEN_SESSION_KEY, token + 1)


def begin_dialog_run(scope: str) -> None:
    """Release the single-dialog slot when *scope* is rerunning on its own.

    A `@st.fragment` rerun never re-executes top-level `app.py`, so when the reset
    lived only there the flag latched `True` after the first dialog and silently
    swallowed every dialog afterwards — dialogs using the default
    ``on_dismiss="ignore"`` do not rerun on close either, so nothing cleared it.

    Resetting unconditionally would be wrong in the other direction: on a full run all
    fragments execute in sequence, and a later one would clear a reservation an earlier
    one had already taken, letting two dialogs open in a single run. So the slot is
    released only when this scope has already run under the current run token, which is
    exactly the fragment-scoped-rerun case.

    Args:
        scope: Stable identifier for the calling fragment.
    """
    token = session_state_get(DIALOG_RUN_TOKEN_SESSION_KEY, 0)
    scope_key = f"{DIALOG_SCOPE_TOKEN_KEY_PREFIX}{scope}"

    if session_state_get(scope_key, None) == token:
        session_state_set(DIALOG_OPENED_THIS_RUN_SESSION_KEY, False)
    else:
        session_state_set(scope_key, token)


def is_dialog_opened_this_run() -> bool:
    """Return whether a dialog has already been opened in the current rerun."""
    return bool(session_state_get(DIALOG_OPENED_THIS_RUN_SESSION_KEY, False))


def mark_dialog_opened_this_run() -> None:
    """Reserve the single-dialog slot for the current rerun.

    Call this immediately after opening an `st.dialog`. Streamlit allows only one dialog
    per run, so the reservation stops a second call site opening another one. The slot is
    released by `begin_script_run()` / `begin_dialog_run()`, never by hand.
    """
    session_state_set(DIALOG_OPENED_THIS_RUN_SESSION_KEY, True)


def dialog_slot_available() -> bool:
    """Return whether another dialog may be opened during this rerun."""
    return not is_dialog_opened_this_run()
