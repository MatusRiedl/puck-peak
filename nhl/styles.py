"""
nhl.styles — stylesheet delivery and UI asset helpers for the Puck Peak page.

The bulk of the stylesheet lives in `assets/puckpeak.css` and is handed to
Streamlit's media file manager, which serves it from `/media/<sha224>.css` with
a real `text/css` content type, gzip, and a content-addressed URL the browser
can cache forever. The page only carries a one-line `@import` plus a small
critical block that stops the Streamlit chrome flashing into view before the
imported sheet lands. Also holds a favicon path resolver so app.py can keep page
chrome configuration simple.

Why not paste the CSS into the page: it used to be a 77 KB Python string sent as
an `st.markdown` delta, alongside a 101 KB base64 data URI of the header logo.
Streamlit's ForwardMsg cache deduplicates both from the second rerun of a
connection onward, but every cold load and every websocket reconnect paid the
full ~180 KB uncompressed. Served from `/media/` it is ~12 KB gzipped once, then
cached.

Why NOT Streamlit's static file serving (`/app/static`): `AppStaticFileHandler`
forces `Content-Type: text/plain` plus `X-Content-Type-Options: nosniff` for any
extension outside `SAFE_APP_STATIC_FILE_EXTENSIONS`, and `.css` is not on that
list. Browsers refuse a `text/plain` stylesheet in standards mode, for `@import`
and `<link>` alike, and no rename fixes it. `MediaFileHandler` has no such
override — it derives the type from the suffix via `mimetypes`.
"""

from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Asset bytes — read once at import, not per rerun
# ---------------------------------------------------------------------------

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _read_asset(name: str) -> bytes:
    """Return the bytes of an asset file, or empty bytes when it is missing.

    Args:
        name: File name inside the repository `assets/` folder.

    Returns:
        bytes: File contents, or `b""` if the file cannot be read.
    """
    try:
        return (_ASSETS_DIR / name).read_bytes()
    except OSError:
        return b""


_CSS_BYTES = _read_asset("puckpeak.css")
_BB_LOGO_BYTES = _read_asset("BB.png")

_CRITICAL_CSS = """
        /* Critical subset — inlined so the Streamlit chrome and the page geometry
           never flash before the imported sheet arrives. Every rule here is also in
           assets/puckpeak.css; test_every_critical_inline_rule_also_exists_in_the_
           external_sheet guards the two against drifting apart. */
        .stDeployButton,
        [data-testid="stDeployButton"],
        #MainMenu > button[kind="header"],
        header [data-testid="stToolbarActionButton"],
        .stAppDeployButton,
        footer,
        #manage-app-button,
        [data-testid="manage-app-button"],
        [data-testid="stStatusWidget"],
        [data-testid="stDecoration"],
        .viewerBadge_container__r5tak,
        ._profileContainer_gzau3_53,
        ._container_gzau3_1 {
            display: none !important;
            visibility: hidden !important;
        }
        .block-container { padding-top: 3.85rem !important; padding-bottom: 0rem !important; padding-left: 2rem !important; padding-right: 2rem !important; }
        [data-testid="stSidebar"] .sidebar-brand__image {
            display: block;
            width: 100%;
            max-width: 100%;
            height: auto;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 2rem !important;
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
        }
"""
"""Minimal CSS inlined every run to avoid a flash of unstyled chrome."""


# ---------------------------------------------------------------------------
# Media URLs
# ---------------------------------------------------------------------------

def get_app_css_path() -> Path:
    """Return the absolute path to the application stylesheet.

    Returns:
        Path: Absolute path to `assets/puckpeak.css`.
    """
    return _ASSETS_DIR / "puckpeak.css"


def get_app_css_text() -> str:
    """Return the full application stylesheet as text.

    Reads from disk on each call so tests and tooling always see the current
    file. `inject_css()` does not use this — it works from the bytes cached at
    import.

    Returns:
        str: Contents of `assets/puckpeak.css`, or an empty string if missing.
    """
    try:
        return get_app_css_path().read_text(encoding="utf-8")
    except OSError:
        return ""


def _media_url(data: bytes, mimetype: str, coordinates: str) -> str | None:
    """Register bytes with Streamlit's media manager and return their URL.

    Must be called on every full script run rather than memoized: the script
    runner calls `media_file_mgr.clear_session_refs()` at the start of each full
    run and `remove_orphaned_files()` at the end, so a file nobody re-registered
    is collected. Fragment reruns deliberately skip the clear, so a fragment-only
    rerun does not need to re-register anything. The call is idempotent and the
    URL is content-addressed, so re-registering the same bytes is free and always
    yields the same URL.

    Args:
        data: Raw file bytes to serve.
        mimetype: Content type used for both the URL suffix and the response.
        coordinates: Stable identifier for this file within the session.

    Returns:
        str | None: A `/media/<hash>.<ext>?v=1` URL, or None when no Streamlit
            runtime is available (bare mode, unit tests, `python app.py`).
    """
    if not data:
        return None

    try:
        from streamlit.runtime import exists as runtime_exists, get_instance as get_runtime

        if not runtime_exists():
            return None
        # ?v=1 is what flips Tornado's StaticFileHandler into its 10-year
        # Cache-Control mode. Safe here because the path is a content hash.
        return f"{get_runtime().media_file_mgr.add(data, mimetype, coordinates)}?v=1"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_favicon_path() -> Path:
    """Return the absolute path to the custom site favicon asset.

    Args:
        None.

    Returns:
        Path: Absolute path to the favicon SVG file in the repository assets folder.
    """
    return _ASSETS_DIR / "favicon.svg"


def inject_css() -> None:
    """Pull in the Puck Peak stylesheet and inline the critical chrome rules.

    Emits one `<style>` element containing an `@import` of the media-served
    sheet followed by the critical block. `@import` must be the first rule in the
    block, so the critical rules sit after it and win ties against their copies
    in the external sheet.

    Falls back to inlining the whole stylesheet when no runtime is available, so
    bare-mode runs and unit tests still get correct styling.

    `<style>` rather than `<link>` on purpose: `st.html()` sanitizes with
    DOMPurify's html profile, whose allowlist contains `style` but not `link`, so
    a `<link>` tag would be silently dropped if this injection is ever moved to
    `st.html`. Keep this on `st.markdown` regardless — `st.html` with style-only
    content routes to the event container, which has no layout footprint, and the
    `.block-container` top padding is tuned around this element occupying a flex
    gap.

    Must be called once per app run, after st.set_page_config().
    """
    url = _media_url(_CSS_BYTES, "text/css", "puckpeak-stylesheet")
    body = _CRITICAL_CSS if url is None else f'@import url("{url}");\n{_CRITICAL_CSS}'
    if url is None:
        body = _CSS_BYTES.decode("utf-8")

    st.markdown(f"<style>{body}</style>", unsafe_allow_html=True)


def inject_header_bb_logo() -> None:
    """Paint the BB logo into the middle of the Streamlit top header bar.

    The geometry stays inline because a 5rem negative margin on the header is a
    layout offset that must not wait on the external sheet; only the image itself
    moves to a media URL. Must be called once per app run, after
    st.set_page_config().
    """
    url = _media_url(_BB_LOGO_BYTES, "image/png", "puckpeak-header-logo")

    # The geometry is emitted whether or not the image resolves: the -5rem header
    # margin is a layout offset, and dropping this injection entirely would also
    # cost the page one of its three style elements (see inject_mobile_dropdown_fix).
    image_rule = (
        ""
        if not url
        else f"""
        [data-testid="stHeader"]::after {{
            content: "";
            display: block;
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 270px;
            height: 48px;
            background-image: url('{url}');
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            pointer-events: none;
        }}"""
    )

    st.markdown(
        f"""
    <style>
        /* CENTER — BB logo pinned to the middle of the Streamlit top header bar */
        [data-testid="stHeader"] {{
            position: relative;
            margin-bottom: -5rem;  /* ← negative value pulls page content up (e.g. -1rem) */
        }}{image_rule}
    </style>
    """,
        unsafe_allow_html=True,
    )


def inject_mobile_dropdown_fix() -> None:
    """Inject the CSS-only mobile dropdown fix after page config is set.

    Kept as its own inline injection: it is ~1 KB, well under the 10 KB
    ForwardMsg cache threshold, and `app.py` depends on there being exactly three
    top-level `st.markdown` style injections — each one occupies a flex gap in the
    main block container, and `.block-container { padding-top }` is tuned around
    that count.
    """
    mobile_css = """
    <style>
        /* Disable search input in dropdowns on touch devices (mobile/tablet)
           to prevent on-screen keyboard from opening when tapping dropdowns */
        @media (pointer: coarse) {
            /* Target the input inside Streamlit selectbox/multiselect dropdowns */
            div[data-baseweb="select"] input,
            div[data-baseweb="popover"] input,
            div[data-baseweb="select"] [role="combobox"] input {
                pointer-events: none !important;
                caret-color: transparent !important;
                -webkit-user-select: none !important;
                user-select: none !important;
            }

            /* Ensure the dropdown container remains fully clickable */
            div[data-baseweb="select"] {
                cursor: pointer !important;
            }
        }

        /* Additional targeting for iOS Safari and older mobile browsers */
        @media (hover: none) and (pointer: coarse) {
            [role="combobox"] input,
            [role="listbox"] input {
                pointer-events: none !important;
                caret-color: transparent !important;
            }
        }
    </style>
    """
    st.markdown(mobile_css, unsafe_allow_html=True)
