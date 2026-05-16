"""
nhl.styles — CSS injection and UI asset helpers for the Puck Peak page.

Contains the CSS injection helpers plus a small favicon path resolver so app.py
can keep page chrome configuration simple and robust across local and deployed
environments.
"""

import base64
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Private CSS block
# ---------------------------------------------------------------------------

_CSS = """
    <style>
        /* GLOBAL — header: hide Deploy button and Manage App widget */
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

        /* DESKTOP — main content area: top/bottom/left/right edge padding */
        .block-container { padding-top: 3.85rem !important; padding-bottom: 0rem !important; padding-left: 2rem !important; padding-right: 2rem !important; }

        /* DESKTOP — FAQ button: negative margin-top pulls it up without touching the collapse button */
        div.element-container:has(.faq-btn-anchor) {
            margin-top: -3rem !important;  /* ← adjust to move FAQ button up/down */
        }

        /* DESKTOP — sidebar logo container: centers the brand image horizontally */
        [data-testid="stSidebar"] .sidebar-brand {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            margin: rem 0 0.65rem 0;  /* ← first value = top margin (e.g. -1rem pulls logo up) */
        }

        /* DESKTOP — sidebar logo image: full-width with glow drop-shadow */
        [data-testid="stSidebar"] .sidebar-brand__image {
            display: block;
            width: 100%;
            max-width: 100%;
            height: auto;
            margin: 0;
            filter: drop-shadow(0 8px 18px rgba(43, 113, 199, 0.16)) drop-shadow(0 6px 20px rgba(255, 255, 255, 0.22));
        }

        /* DESKTOP — trims the gap below the sidebar logo wrapper */
        div.element-container:has(.sidebar-brand) {
            margin-bottom: 0.55rem !important;
        }

        /* DESKTOP — removes default top margin on all expander widgets */
        [data-testid="stExpander"] {
            margin-top: 0 !important;
        }

        /* MOBILE — tighter block-container padding (top/left/right); lock sidebar to vertical-only scroll */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 2rem !important;
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            section[data-testid="stSidebar"],
            section[data-testid="stSidebar"] > div:first-child {
                overflow-x: hidden !important;
                touch-action: pan-y !important;
            }
        }

        /* DESKTOP — makes all Streamlit buttons fill their container width */
        .stButton button { width: 100%; }

        /* DESKTOP — Ko-fi support button: pill shape, amber gradient, layout */
        [data-testid="stSidebar"] .sidebar-support-link {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 0.65rem;
            width: 100%;
            margin: 0.72rem 0 0.48rem 0;
            padding: 0.58rem 0.82rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 244, 231, 0.16);
            background: linear-gradient(135deg, #9d6535 0%, #bf7a3f 100%);
            color: #ffffff !important;
            text-decoration: none !important;
            box-shadow: 0 6px 14px rgba(88, 49, 24, 0.16);
            transition: transform 0.16s ease, box-shadow 0.16s ease, filter 0.16s ease;
        }

        /* DESKTOP — Ko-fi button hover: slight lift + brightness */
        [data-testid="stSidebar"] .sidebar-support-link:hover {
            transform: translateY(-1px);
            filter: brightness(1.02);
            box-shadow: 0 8px 18px rgba(88, 49, 24, 0.2);
        }

        /* DESKTOP — Ko-fi button keyboard focus ring */
        [data-testid="stSidebar"] .sidebar-support-link:focus,
        [data-testid="stSidebar"] .sidebar-support-link:focus-visible {
            outline: 2px solid rgba(255, 255, 255, 0.8);
            outline-offset: 2px;
        }

        /* DESKTOP — Ko-fi button emoji badge: circular, fixed size */
        [data-testid="stSidebar"] .sidebar-support-link__emoji {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
            width: 1.65rem;
            height: 1.65rem;
            border-radius: 999px;
            background: rgba(255, 248, 240, 0.18);
            font-size: 0.95rem;
            line-height: 1;
            font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
        }

        /* DESKTOP — Ko-fi button text column: stacks label + sublabel */
        [data-testid="stSidebar"] .sidebar-support-link__text {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            min-width: 0;
        }

        /* DESKTOP — Ko-fi button main label: bold, slightly larger */
        [data-testid="stSidebar"] .sidebar-support-link__label {
            font-weight: 700;
            font-size: 0.9rem;
            line-height: 1.08;
        }

        /* DESKTOP — Ko-fi button sublabel: smaller, dimmer */
        [data-testid="stSidebar"] .sidebar-support-link__sublabel {
            margin-top: 0.08rem;
            font-size: 0.73rem;
            line-height: 1.12;
            color: rgba(255, 247, 240, 0.86);
        }

        /* DESKTOP — sidebar player row: shrinks add/remove buttons to auto-width, floats right */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] div.stButton button {
            width: auto !important;
            min-width: 0 !important;
            padding: 0.2rem 0.6rem !important;
            float: right;
        }

        /* DESKTOP — sidebar X (remove) button: transparent, white icon, fixed 24x32 px */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button[kind="secondary"][data-testid="stBaseButton-secondary"] {
            background-color: transparent !important;
            border: none !important;
            color: white !important;
            padding: 0 !important;
            min-width: 24px !important;
            width: 24px !important;
            height: 32px !important;
            font-size: 18px !important;
            line-height: 32px !important;
            margin-left: -8px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        /* DESKTOP — sidebar X button hover: subtle tint + blue icon */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button[kind="secondary"][data-testid="stBaseButton-secondary"]:hover {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: #2596be !important;
        }

        /* DESKTOP — sidebar player row columns: equal height, content centered vertically */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            align-items: stretch !important;
            gap: 0 !important;
        }

        /* DESKTOP — sidebar row: each column is a flex container for vertical centering */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            display: flex !important;
            align-items: center !important;
        }

        /* DESKTOP — sidebar row: inner vertical block stretches full width */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] {
            width: 100% !important;
            justify-content: center !important;
        }

        /* DESKTOP — sidebar row: zero margins/padding on inner element-containers */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .element-container {
            margin: 0 !important;
            padding: 0 !important;
        }

        /* DESKTOP — sidebar: tighten vertical spacing on hr, element-container, h3, widget labels */
        [data-testid="stSidebar"] .stMarkdown hr {
            margin-top: 4px !important;
            margin-bottom: 4px !important;
        }
        [data-testid="stSidebar"] .element-container {
            margin-bottom: 2px !important;
        }
        [data-testid="stSidebar"] h3 {
            margin-top: 4px !important;
            margin-bottom: 4px !important;
        }
        [data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
            margin-bottom: 0.18rem !important;
        }

        /* DESKTOP — sidebar: removes top margin above Global Search widget label */
        [data-testid="stSidebar"] .element-container:has(> div > div > label[data-testid="stWidgetLabel"]:nth-child(1)) {
            margin-top: 0 !important;
        }
        /* DESKTOP — sidebar: removes top margin/padding on first text input after a divider */
        [data-testid="stSidebar"] hr + .element-container .stTextInput label {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* DESKTOP — sidebar: normalizes first Team dropdown (size, radius, padding) to match search */
        [data-testid="stSidebar"] hr + .element-container .stSelectbox label {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        [data-testid="stSidebar"] hr + .element-container [data-baseweb="select"] > div {
            min-height: 3.25rem !important;
            border-radius: 0.75rem !important;
            padding-left: 0.95rem !important;
            padding-right: 2.75rem !important;
            align-items: center !important;
        }
        [data-testid="stSidebar"] hr + .element-container [data-baseweb="select"] > div > div:first-child {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        [data-testid="stSidebar"] hr + .element-container [data-baseweb="select"] * {
            font-size: 15px !important;
            line-height: 1.3 !important;
        }
        [data-testid="stSidebar"] hr + .element-container [data-baseweb="select"] svg {
            width: 18px !important;
            height: 18px !important;
        }
        /* DESKTOP — sidebar: dims selected value text in dropdowns to a muted grey */
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div > div:first-child,
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div > div:first-child * {
            color: rgba(250, 250, 250, 0.55) !important;
        }

        /* DESKTOP — expander summary: compact padding, centered bold title */
        [data-testid="stExpander"] details summary {
            padding-top: 0.4rem !important;
            padding-bottom: 0.4rem !important;
            justify-content: center !important;
        }
        [data-testid="stExpander"] details summary p {
            font-size: 1.08rem !important;
            font-weight: 700 !important;
            color: rgba(255, 255, 255, 0.80) !important;
            letter-spacing: 0.01em !important;
        }
        [data-testid="stExpander"] details > div {
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
        }
        [data-testid="stExpander"] .element-container {
            margin-bottom: 0 !important;
        }
        [data-testid="stExpander"] [data-testid="stHorizontalBlock"] {
            gap: 0.5rem !important;
            row-gap: 0.25rem !important;
        }
        [data-testid="stExpander"] .stRadio > label {
            margin-bottom: 0.1rem !important;
        }
        [data-testid="stExpander"] [data-testid="stToggle"] {
            margin-bottom: 0 !important;
        }
        [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
            gap: 0.25rem !important;
        }

        /* DESKTOP — Metric Selections popover: fixed 640px wide so all labels fit */
        [data-testid="stPopoverBody"] {
            width: 640px !important;
            min-width: 640px !important;
        }
        /* MOBILE — Metric Selections popover: stretches to 94% of viewport width */
        @media (max-width: 768px) {
            [data-testid="stPopoverBody"] {
                width: 94vw !important;
                min-width: 0 !important;
            }
        }

        /* DESKTOP — controls toolbar: muted pill row for unavailable metric options */
        .controls-toolbar-muted {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.35rem;
            margin: 0.25rem 0 0.1rem 0;
        }
        .controls-toolbar-muted__label {
            color: #7f8aa3;
            font-size: 0.76rem;
            font-weight: 600;
        }
        .controls-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 1.8rem;
            padding: 0.12rem 0.62rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            line-height: 1;
            white-space: nowrap;
        }
        .controls-pill--disabled {
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: rgba(30, 41, 59, 0.45);
            color: #7f8aa3;
        }

        /* DESKTOP — sidebar player name: single line, ellipsis on overflow */
        .player-name {
            font-size: 15px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            line-height: 32px !important;
        }

        /* DESKTOP — sidebar row: vertically centers the markdown element inside its column */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stMarkdown"] {
            display: flex !important;
            align-items: center !important;
            margin: 0 !important;
        }

        /* DESKTOP — sidebar row: right-aligns and vertically centers the button column */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stButton"],
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-end !important;
            margin: 0 !important;
        }

        /* DESKTOP — blue-btn-anchor: styles the next button as solid blue */
        div.element-container:has(.blue-btn-anchor) + div.element-container button {
            background-color: #2b71c7 !important;
            border-color: #2b71c7 !important;
            color: white !important;
        }
        div.element-container:has(.blue-btn-anchor) + div.element-container button:hover {
            background-color: #1a569d !important;
            border-color: #1a569d !important;
        }

        /* DESKTOP — faq-btn-anchor: styles the next button as a ghost blue button */
        div.element-container:has(.faq-btn-anchor) + div.element-container button {
            background: rgba(43, 113, 199, 0.16) !important;
            border: 1px solid rgba(103, 168, 255, 0.28) !important;
            color: rgba(230, 241, 255, 0.95) !important;
            box-shadow: inset 0 0 0 1px rgba(43, 113, 199, 0.05) !important;
        }
        div.element-container:has(.faq-btn-anchor) + div.element-container button:hover {
            background: rgba(43, 113, 199, 0.24) !important;
            border-color: rgba(124, 184, 255, 0.4) !important;
            color: #ffffff !important;
        }

        /* DESKTOP — live game cards: shell, link overlay, card base styles */
        .live-game-card-shell {
            position: relative;
        }
        .live-game-card-link {
            position: absolute;
            inset: 0;
            z-index: 2;
            display: block;
            border-radius: 10px;
            color: inherit !important;
            text-decoration: none !important;
        }
        .live-game-card-link:hover,
        .live-game-card-link:focus,
        .live-game-card-link:focus-visible {
            color: inherit !important;
            text-decoration: none !important;
            outline: none;
        }
        .live-game-card {
            position: relative;
            z-index: 1;
            background:
                linear-gradient(
                    105deg,
                    var(--lgc-away-tint, transparent) 0%,
                    rgba(255, 255, 255, 0.018) 38%,
                    rgba(255, 255, 255, 0.018) 62%,
                    var(--lgc-home-tint, transparent) 100%
                );
            box-shadow: inset 0 0 80px var(--lgc-inset-glow, transparent);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 10px;
            margin-bottom: 0.55rem;
            padding: 0.55rem 0.65rem 0.45rem;
            box-sizing: border-box;
            cursor: pointer;
            pointer-events: none;
            transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
        }
        .live-game-card-shell:hover .live-game-card,
        .live-game-card-shell:focus-within .live-game-card,
        .live-game-card-link:focus + .live-game-card,
        .live-game-card-link:focus-visible + .live-game-card,
        .live-game-card:hover,
        .live-game-card:focus-within,
        .live-game-card:focus {
            transform: translateY(-3px);
            box-shadow:
                0 10px 22px rgba(0, 0, 0, 0.26),
                inset 0 0 80px var(--lgc-inset-glow, transparent);
            border-color: rgba(255, 255, 255, 0.16);
            outline: none;
        }
        .lgc-header {
            position: relative;
            display: block;
            margin-bottom: 0.34rem;
        }
        .lgc-header__main {
            min-width: 0;
            padding-right: 0;
        }
        .lgc-matchup {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            font-size: 1rem;
            margin-bottom: 0.15rem;
        }
        .lgc-detail {
            color: #8c8c8c;
            font-size: 0.9rem;
            line-height: 1.2;
            min-width: 0;
        }
        .lgc-meta {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            justify-content: flex-start;
            gap: 0.14rem;
            min-width: 220px;
            max-width: min(280px, calc(100vw - 4rem));
            text-align: left;
        }
        .lgc-meta-popover {
            position: absolute;
            top: 0.1rem;
            right: 0;
            z-index: 4;
            opacity: 0;
            visibility: hidden;
            transform: translateY(6px);
            pointer-events: none;
            transition: opacity 140ms ease, visibility 140ms ease, transform 140ms ease;
        }
        .live-game-card-shell:hover .lgc-meta-popover,
        .live-game-card-shell:focus-within .lgc-meta-popover,
        .live-game-card-link:focus + .live-game-card .lgc-meta-popover,
        .live-game-card-link:focus-visible + .live-game-card .lgc-meta-popover,
        .live-game-card:hover .lgc-meta-popover,
        .live-game-card:focus-within .lgc-meta-popover,
        .live-game-card:focus .lgc-meta-popover {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }
        .lgc-meta-popover .lgc-meta {
            padding: 0.48rem 0.58rem;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 10px;
            background: rgba(8, 13, 21, 0.96);
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.34);
            backdrop-filter: blur(10px);
        }
        .lgc-meta-popover .live-games-probability__meta {
            font-size: 0.82rem;
            font-weight: 600;
            color: #c3c3c3;
        }
        .lgc-prob-section {
            /* probability labels + bar embedded below the header block */
        }
        .live-games-probability--muted {
            color: #8c8c8c;
            font-size: 0.9rem;
        }

        .live-games-probability__labels {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.93rem;
            margin-bottom: 0.28rem;
        }
        .live-games-probability__label {
            color: rgba(255, 255, 255, var(--label-opacity, 0.92));
            text-shadow: 0 0 18px var(--label-glow, rgba(0, 0, 0, 0));
            transition: color 120ms ease, text-shadow 120ms ease, opacity 120ms ease;
        }
        .live-games-probability__label strong {
            font-size: 1.02rem;
            letter-spacing: -0.01em;
        }
        .live-games-probability__label--leading {
            color: rgba(255, 255, 255, 0.99);
        }
        .live-games-probability__label--trailing {
            color: rgba(255, 255, 255, 0.8);
        }
        .live-games-probability__bar {
            display: flex;
            position: relative;
            isolation: isolate;
            width: 100%;
            height: 8px;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.045);
            margin-bottom: 0.38rem;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04), inset 0 8px 18px rgba(255, 255, 255, 0.03);
        }
        .live-games-probability__bar::before,
        .live-games-probability__bar::after {
            content: "";
            position: absolute;
            top: -10px;
            bottom: -10px;
            pointer-events: none;
            filter: blur(10px);
            opacity: 0.95;
            z-index: 0;
        }
        .live-games-probability__bar::before {
            left: 0;
            width: var(--away-glow-width, 0%);
            background: linear-gradient(90deg, var(--away-bar-glow, rgba(0, 0, 0, 0)), rgba(0, 0, 0, 0) 88%);
        }
        .live-games-probability__bar::after {
            right: 0;
            width: var(--home-glow-width, 0%);
            background: linear-gradient(270deg, var(--home-bar-glow, rgba(0, 0, 0, 0)), rgba(0, 0, 0, 0) 88%);
        }
        .live-games-probability__segment {
            display: block;
            position: relative;
            z-index: 1;
            height: 100%;
            min-width: 0;
            background:
                linear-gradient(
                    180deg,
                    rgba(255, 255, 255, var(--segment-sheen, 0.08)) 0%,
                    var(--segment-color, rgba(255, 255, 255, 0.55)) 45%,
                    var(--segment-color, rgba(255, 255, 255, 0.55)) 100%
                );
            opacity: var(--segment-opacity, 1);
            filter: saturate(var(--segment-saturation, 1)) brightness(var(--segment-brightness, 1));
            transition: opacity 120ms ease, filter 120ms ease, box-shadow 120ms ease;
        }
        .live-games-probability__segment--leading {
            box-shadow: inset 0 0 12px rgba(255, 255, 255, 0.16), 0 0 14px var(--segment-glow, rgba(0, 0, 0, 0));
        }
        .live-games-probability__segment--trailing {
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.16);
        }
        .live-games-probability__segment--tied {
            box-shadow: inset 0 0 10px rgba(255, 255, 255, 0.12);
        }
        .live-games-probability__divider {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 6px;
            transform: translateX(-50%);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.4);
            pointer-events: none;
            z-index: 2;
        }
        .live-games-probability__meta {
            color: #b1b1b1;
            font-size: 0.75rem;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        .live-games-probability__meta--playoff {
            margin-top: 0.18rem;
            padding-top: 0.18rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }
        div.element-container:has(.live-game-card) {
            margin-bottom: 0.55rem !important;
        }

        /* TABLET (≤1200px) — live game card hover tooltip: narrows max-width */
        @media (max-width: 1200px) {
            .lgc-meta {
                max-width: min(250px, calc(100vw - 4rem));
            }
        }

        /* MOBILE — live game cards: disables hover lift; shows stat tooltip inline instead of floating */
        @media (hover: none), (max-width: 768px) {
            .live-game-card {
                transform: none !important;
            }
            .lgc-meta-popover {
                position: static;
                opacity: 1;
                visibility: visible;
                transform: none;
                pointer-events: auto;
                margin-top: 0.35rem;
            }
            .lgc-meta-popover .lgc-meta {
                min-width: 0;
                max-width: none;
                padding: 0;
                border: 0;
                border-radius: 0;
                background: transparent;
                box-shadow: none;
                backdrop-filter: none;
            }
            div:has(> #controls-dropdowns) + div [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
            }
            div:has(> #controls-dropdowns) + div [data-testid="column"] {
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }
        }

        /* DESKTOP — main chart toolbar: layout, title, and share button base styles */
        div.element-container:has(.nhl-chart-toolbar) {
            margin: 0 !important;
            line-height: 0 !important;
        }
        div.element-container:has(.nhl-chart-toolbar) + div.element-container {
            margin-top: 0 !important;
        }
        .nhl-chart-toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            min-height: 40px !important;
            margin: 0 0 0.18rem 0;
        }
        .nhl-chart-toolbar__title {
            color: rgba(255, 255, 255, 0.90);
            font-size: 1rem;
            font-weight: 400;
            line-height: 1.2;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .nhl-chart-share-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
            padding: 0.35rem 0.7rem;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.72);
            color: #dbe4f0;
            font-size: 0.8rem;
            font-weight: 600;
            line-height: 1;
            cursor: pointer;
            transition: border-color 0.18s ease, color 0.18s ease, background 0.18s ease;
        }
        .nhl-chart-share-btn:hover {
            color: #ffffff;
            border-color: rgba(255, 255, 255, 0.28);
            background: rgba(30, 41, 59, 0.88);
        }
        .nhl-chart-share-btn.is-copied {
            color: #4ade80;
            border-color: rgba(74, 222, 128, 0.45);
        }
        .nhl-chart-share-btn svg {
            width: 15px;
            height: 15px;
            display: block;
        }
        /* TABLET (≤900px) — chart toolbar: tighter gaps, smaller title and share button font */
        @media (max-width: 900px) {
            .nhl-chart-toolbar {
                gap: 0.5rem;
            }
            .nhl-chart-toolbar__title {
                font-size: 0.90rem;
            }
            .nhl-chart-share-btn {
                padding: 0.30rem 0.60rem;
                font-size: 0.70rem;
            }
        }
        /* MOBILE — chart toolbar: even tighter sizing; top margin pushes toolbar below Streamlit header */
        @media (max-width: 768px) {
            .nhl-chart-toolbar {
                gap: 0.4rem;
                min-height: 32px !important;
                margin: 3.75rem 0 0.1rem 0;
            }
            .nhl-chart-toolbar__title {
                font-size: 0.84rem;
                line-height: 1.15;
            }
            .nhl-chart-share-btn {
                gap: 0.28rem;
                padding: 0.24rem 0.52rem;
                font-size: 0.7rem;
            }
            .nhl-chart-share-btn svg {
                width: 13px;
                height: 13px;
            }
        }

        /* DESKTOP — comparison panel: matches column gap between player cards to prediction gap */
        [data-baseweb="tab-panel"] [data-testid="stHorizontalBlock"]:has(.comparison-player-card) {
            column-gap: 1.1rem !important;
        }

        /* DESKTOP — comparison panel cards: base padding and typography */
        .comparison-card {
            padding: 0.5rem 0.25rem;
        }
        .comparison-card b {
            font-size: 18px;
        }
        .comparison-card small {
            color: #aaa;
            font-size: 12px;
        }
        .comparison-player-card {
            display: flex;
            align-items: flex-start;
            gap: 0.9rem;
            margin: 0 0 1.1rem 0;
            padding: 0.9rem;
            border: 1px solid rgba(70, 84, 122, 0.5);
            border-radius: 18px;
            background: linear-gradient(
                160deg,
                var(--pc-color-tint, transparent) 0%,
                rgba(12, 18, 33, 0.92) 40%,
                rgba(9, 13, 24, 0.98) 100%
            );
            box-shadow:
                inset 0 0 80px var(--pc-inset-glow, transparent),
                0 12px 24px rgba(0, 0, 0, 0.16);
            transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
        }
        .comparison-card-shell {
            display: block;
        }
        .comparison-card-shell--clickable {
            display: block;
            cursor: pointer;
            outline: none;
        }
        .comparison-card-shell--clickable .comparison-player-card {
            cursor: pointer;
        }
        .comparison-card-shell--clickable:hover .comparison-player-card,
        .comparison-card-shell--clickable:focus .comparison-player-card,
        .comparison-card-shell--clickable:focus-visible .comparison-player-card,
        .comparison-card-shell--clickable:focus-within .comparison-player-card {
            transform: translateY(-3px);
            border-color: rgba(255, 255, 255, 0.16);
            box-shadow:
                inset 0 0 80px var(--pc-inset-glow, transparent),
                0 14px 30px rgba(0, 0, 0, 0.24);
        }
        .comparison-card-shell--clickable:focus,
        .comparison-card-shell--clickable:focus-visible {
            outline: none;
        }
        .comparison-player-card--no-image {
            padding-left: 1rem;
        }
        .comparison-player-card__media {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            flex: 0 0 38%;
            max-width: 180px;
        }
        .comparison-player-card__media--player {
            position: relative;
            align-items: center;
            justify-content: flex-end;
            flex: 0 0 20%;
            min-width: 72px;
            min-height: 86px;
            max-width: 104px;
            padding: 0.15rem 0.2rem 0;
            isolation: isolate;
            overflow: visible;
        }
        .comparison-player-card__media--player::before {
            content: "";
            position: absolute;
            inset: 14% 8% 22%;
            border-radius: 50%;
            background: radial-gradient(
                circle at 50% 40%,
                rgba(255, 255, 255, 0.14) 0%,
                rgba(116, 148, 220, 0.18) 28%,
                rgba(39, 54, 92, 0) 74%
            );
            filter: blur(13px);
            z-index: 0;
            pointer-events: none;
        }
        .comparison-player-card__media--player::after {
            content: "";
            position: absolute;
            left: 16%;
            right: 16%;
            bottom: 0.45rem;
            height: 1.2rem;
            border-radius: 999px;
            background: radial-gradient(circle at 50% 50%, rgba(0, 0, 0, 0.38) 0%, rgba(0, 0, 0, 0) 72%);
            filter: blur(6px);
            z-index: 0;
            pointer-events: none;
        }
        .comparison-player-card__image {
            display: block;
            width: 100%;
            aspect-ratio: 4 / 3;
            object-fit: cover;
            border-radius: 14px;
        }
        .comparison-player-card__image--player-cutout {
            position: relative;
            z-index: 1;
            width: auto;
            max-width: 100%;
            height: 96px;
            aspect-ratio: auto;
            object-fit: contain;
            border-radius: 0;
            filter: drop-shadow(0 12px 14px rgba(0, 0, 0, 0.32)) drop-shadow(0 3px 6px rgba(0, 0, 0, 0.16));
            transform: translateY(5px) scale(1.16);
            transform-origin: center bottom;
        }
        .comparison-player-card__body {
            flex: 1 1 auto;
            min-width: 0;
        }
        .comparison-card-stats {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 0.2rem 0.9rem;
            margin: 0.18rem 0 0 0;
            font-size: 0.92rem;
        }
        .comparison-card-context-row {
            margin-top: 0.18rem;
            line-height: 1.2;
        }
        .comparison-card-stats__item {
            display: inline-flex;
            align-items: baseline;
            min-width: 0;
            white-space: nowrap;
        }
        .comparison-card-stats__label {
            color: #f4f6fb;
            font-weight: 700;
        }
        .comparison-card-stats__value {
            color: #f4f6fb;
            font-weight: 600;
        }
        .comparison-player-card--team {
            align-items: center;
        }
        .comparison-player-card__media--team {
            flex: 0 0 112px;
            max-width: 112px;
            align-items: center;
            justify-content: center;
        }
        .comparison-player-card__image--team-logo {
            width: 100%;
            aspect-ratio: 1 / 1;
            object-fit: contain;
            padding: 0.4rem;
        }
        .comparison-panel-heading {
            margin: 0 0 0.22rem 0;
            color: #f4f6fb;
            font-size: 0.98rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .comparison-panel-heading--rail-title {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            text-align: center;
            font-size: 1.08rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            line-height: 1.1;
            color: rgba(255, 255, 255, 0.80);
        }
        .comparison-panel-heading--predictions {
            margin: 0 auto 0.42rem;
            padding: 0.18rem 0 0.16rem;
        }
        .stanley-cup-board-shell {
            margin: 0.2rem 0 0.85rem 0;
        }
        .stanley-cup-board-meta {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.55rem 1rem;
            margin: 0 0 0.55rem 0;
            color: rgba(229, 237, 249, 0.84);
            font-size: 0.82rem;
            line-height: 1.3;
        }
        .stanley-cup-board-summary {
            margin: 0 0 0.75rem 0;
            padding: 0.6rem 0.8rem;
            border: 1px solid rgba(71, 85, 105, 0.42);
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.78) 0%, rgba(17, 24, 39, 0.94) 100%);
            color: rgba(226, 232, 240, 0.88);
            font-size: 0.84rem;
            line-height: 1.35;
        }
        .stanley-cup-board-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.95rem;
        }
        .stanley-cup-division-window {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(70, 84, 122, 0.52);
            border-radius: 18px;
            background: linear-gradient(
                160deg,
                var(--division-accent-soft, rgba(96, 165, 250, 0.08)) 0%,
                rgba(12, 18, 33, 0.94) 34%,
                rgba(8, 12, 22, 0.98) 100%
            );
            box-shadow:
                inset 0 0 72px var(--division-accent-soft, transparent),
                0 12px 24px rgba(0, 0, 0, 0.18);
        }
        .stanley-cup-division-window::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: linear-gradient(90deg, var(--division-accent, #60a5fa) 0%, var(--division-accent-glow, rgba(96, 165, 250, 0.28)) 100%);
        }
        .stanley-cup-division-header {
            padding: 0.9rem 1rem 0.7rem 1rem;
        }
        .stanley-cup-division-kicker {
            margin: 0 0 0.18rem 0;
            color: rgba(191, 219, 254, 0.74);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .stanley-cup-division-heading {
            color: #f8fafc;
            font-size: 1.18rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .stanley-cup-table-head,
        .stanley-cup-row {
            display: grid;
            grid-template-columns: minmax(0, 1.85fr) repeat(5, minmax(32px, 0.42fr));
            gap: 0.35rem;
            align-items: center;
        }
        .stanley-cup-table-head {
            padding: 0 1rem 0.5rem 1rem;
            color: rgba(203, 213, 225, 0.76);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .stanley-cup-table-head__team {
            text-align: left;
        }
        .stanley-cup-row {
            margin: 0 0.55rem;
            padding: 0.74rem 0.45rem;
            border-top: 1px solid rgba(71, 85, 105, 0.34);
            color: #e5edf9;
            transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
        }
        .stanley-cup-row-shell {
            display: block;
            cursor: pointer;
            outline: none;
        }
        .stanley-cup-row:hover {
            background: rgba(255, 255, 255, 0.03);
        }
        .stanley-cup-row-shell:hover .stanley-cup-row,
        .stanley-cup-row-shell:focus .stanley-cup-row,
        .stanley-cup-row-shell:focus-visible .stanley-cup-row,
        .stanley-cup-row-shell:focus-within .stanley-cup-row {
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(148, 163, 184, 0.42);
            transform: translateY(-2px);
        }
        .stanley-cup-row--favorite {
            border-color: var(--favorite-accent, #22c55e);
            border-radius: 12px;
            background: linear-gradient(
                135deg,
                var(--favorite-accent-soft, rgba(34, 197, 94, 0.16)) 0%,
                rgba(15, 23, 42, 0.92) 100%
            );
            box-shadow:
                inset 0 0 44px var(--favorite-accent-soft, rgba(34, 197, 94, 0.12)),
                0 0 0 1px var(--favorite-accent, #22c55e),
                0 12px 24px rgba(0, 0, 0, 0.16);
            transform: translateY(-1px);
        }
        .stanley-cup-row-team {
            display: flex;
            align-items: center;
            gap: 0.62rem;
            min-width: 0;
        }
        .stanley-cup-team-logo {
            width: 26px;
            height: 26px;
            object-fit: contain;
            flex: 0 0 26px;
        }
        .stanley-cup-row-team-name {
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 0.96rem;
            font-weight: 700;
        }
        .stanley-cup-row-badge {
            margin-left: auto;
            padding: 0.18rem 0.46rem;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            color: #f8fafc;
            font-size: 0.67rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .stanley-cup-row-value {
            color: rgba(241, 245, 249, 0.9);
            font-size: 0.92rem;
            font-weight: 600;
            text-align: center;
        }
        .stanley-cup-row-value--pts {
            color: #ffffff;
            font-weight: 800;
        }
        .stanley-cup-favorite-button-anchor {
            display: inline-flex;
            align-items: center;
            margin-right: 0.45rem;
        }
        .stanley-cup-favorite-button-anchor::before {
            content: "Cup pick";
            display: inline-flex;
            align-items: center;
            padding: 0.18rem 0.46rem;
            border: 1px solid rgba(250, 204, 21, 0.34);
            border-radius: 999px;
            background: rgba(250, 204, 21, 0.12);
            color: rgba(254, 240, 138, 0.98);
            font-size: 0.66rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .comparison-panel-heading--season {
            margin: 0 auto 0.2rem;
            padding: 0.18rem 0 0.08rem;
        }
        .comparison-trace-toggle-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.22rem 0 0 0;
        }
        .comparison-trace-toggle {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.34rem 0.72rem;
            border: 1px solid rgba(96, 165, 250, 0.20);
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.62);
            color: #e5edf9;
            font-size: 0.78rem;
            font-weight: 600;
            line-height: 1;
            cursor: pointer;
            transition: border-color 0.18s ease, background 0.18s ease, color 0.18s ease, opacity 0.18s ease;
        }
        .comparison-trace-toggle:hover {
            border-color: rgba(148, 163, 184, 0.34);
            background: rgba(30, 41, 59, 0.86);
        }
        .comparison-trace-toggle.is-inactive {
            opacity: 0.56;
            background: rgba(15, 23, 42, 0.28);
        }
        .comparison-trace-toggle--icon-only {
            justify-content: center;
            gap: 0;
            min-width: 2.45rem;
            padding: 0.34rem 0.62rem;
        }
        .comparison-trace-toggle__line {
            position: relative;
            width: 18px;
            height: 0;
            border-top: 3px solid var(--trace-toggle-color, #4caf50);
            border-radius: 999px;
            flex: 0 0 auto;
        }
        .comparison-trace-toggle__line::after {
            content: "";
            position: absolute;
            top: -5px;
            left: 50%;
            width: 8px;
            height: 8px;
            transform: translateX(-50%);
            border-radius: 999px;
            background: var(--trace-toggle-color, #4caf50);
            box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.12);
        }
        .comparison-trace-toggle__label {
            white-space: nowrap;
        }
        .comparison-trace-toggle--compact {
            padding: 0.28rem 0.64rem;
            font-size: 0.74rem;
        }
        /* MOBILE — comparison player card: disables hover lift on touch devices */
        @media (hover: none), (max-width: 768px) {
            .comparison-card-shell--clickable .comparison-player-card {
                transform: none !important;
            }
        }

        /* DESKTOP — season filter anchor: collapses the zero-height anchor div spacing */
        div:has(> #comparison-season-filter) {
            margin: 0 !important;
            line-height: 0 !important;
        }
        div:has(> #comparison-season-filter) + div {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        div:has(> #comparison-season-filter) + div .stSelectbox label {
            display: none !important;
        }
        div:has(> #comparison-season-filter) + div .stSelectbox {
            margin-bottom: 0 !important;
        }
        div:has(> #comparison-season-filter) + div .stSelectbox [data-baseweb="select"] > div > div:first-child,
        div:has(> #comparison-season-filter) + div .stSelectbox [data-baseweb="select"] > div > div:first-child * {
            font-weight: 700 !important;
            font-size: 1.08rem !important;
            color: rgba(255, 255, 255, 0.80) !important;
            letter-spacing: 0.01em !important;
        }
        /* DESKTOP — controls panel anchor: collapses anchor div; pulls Metric Selections button up */
        div:has(> #comparison-controls-panel) {
            margin: 0 !important;
            line-height: 0 !important;
        }
        div:has(> #comparison-controls-panel) + div {
            margin-top: -0.55rem !important;
            margin-bottom: 0 !important;
        }
        div:has(> #comparison-controls-panel) + div [data-testid="stExpander"] {
            margin-bottom: 0 !important;
        }
        div:has(> #comparison-controls-panel) + div [data-testid="stExpander"] details summary {
            padding-top: 0.3rem !important;
            padding-bottom: 0.3rem !important;
        }
        /* MOBILE — Metric Selections: pulls the whole stacked column up toward the chart */
        @media (max-width: 768px) {
            [data-testid="stColumn"]:has(#comparison-controls-panel) {
                margin-top: -1.5rem !important;
            }
        }
        /* DESKTOP — Whole career / Metric Selections row: negative pull closes gap below chart */
        [data-testid="stHorizontalBlock"]:has(#comparison-season-filter) {
            margin-top: -2.4rem !important;
        }
        /* MOBILE — Whole career / Metric Selections row: stronger pull + removes side padding */
        @media (max-width: 768px) {
            [data-testid="stHorizontalBlock"]:has(#comparison-season-filter) {
                margin-top: -2.4rem !important;
                padding-left: 0rem !important;
                padding-right: 0rem !important;
            }
        }
        /* ══════════════════════════════════════════════════════════════════════════
           INTERMEDIATE BREAKPOINT BANDS  (between mobile ≤768px and desktop ≥1401px)
           ──────────────────────────────────────────────────────────────────────────
           Three independently-tunable bands. Edit ONLY the property values; do NOT
           change the min-width/max-width numbers (that shifts the band boundaries).

           KNOB REFERENCE — which CSS property controls what on screen:
           ┌──────────────────────────────────────────────────────────────────────┐
           │ controls-row margin-top   │ gap between chart bottom edge and the   │
           │                           │ "Whole career / Metric Selections" row. │
           │                           │ More negative → row moves UP (less gap).│
           ├──────────────────────────────────────────────────────────────────────┤
           │ controls-panel margin-top │ Metric Selections button vertical offset │
           │                           │ relative to the controls-row pull.       │
           ├──────────────────────────────────────────────────────────────────────┤
           │ detail-layout margin-top  │ pulls the Overview/Standings block up.  │
           │                           │ More negative → tabs section moves UP.  │
           ├──────────────────────────────────────────────────────────────────────┤
           │ tabs anchor margin-top    │ gap above the invisible tab-anchor div  │
           │                           │ (sits just above the tab-pill bar).     │
           ├──────────────────────────────────────────────────────────────────────┤
           │ tabs + div margin-top     │ gap above the visible tab-pill bar row. │
           ├──────────────────────────────────────────────────────────────────────┤
           │ right-rail margin-top     │ top offset of the predictions panel.    │
           │                           │ Only visible in STACKED layout          │
           │                           │ (Bands B & C); ignored when columns are │
           │                           │ still side-by-side (Band A).            │
           └──────────────────────────────────────────────────────────────────────┘

           BANDS:
             Band A │ 1281–1400px │ sidebar overlay │ columns SIDE-BY-SIDE
             Band B │ 1025–1280px │ sidebar overlay │ columns STACKED
             Band C │  769–1024px │ sidebar overlay │ columns STACKED
           ════════════════════════════════════════════════════════════════════════ */

        /* BAND A (1281–1400px) — sidebar is a fixed overlay, main columns still SIDE-BY-SIDE.
           Chart is in the left column; predictions panel is in the right column.
           right-rail margin-top has no visible effect here (columns are not stacked). */
        @media (min-width: 1281px) and (max-width: 1400px) {
            /* controls-row: pulls "Whole career / Metric Selections" row up under the chart */
            [data-testid="stHorizontalBlock"]:has(#comparison-season-filter) {
                margin-top: -2.4rem !important;
            }
            /* controls-panel: Metric Selections button fine-offset within the controls row */
            div:has(> #comparison-controls-panel) + div {
                margin-top: 0 !important;
            }
            /* detail-layout: pulls Overview / Current Standings section up toward chart bottom */
            div.element-container:has(#comparison-detail-layout) {
                margin-top: -3.7rem !important;
            }
            /* tabs anchor: gap above the invisible tab-anchor div (above the pill row) */
            div.element-container:has(#comparison-tabs) {
                margin-top: -0.6rem !important;
            }
            /* tabs + div: gap above the actual visible tab-pill bar */
            div.element-container:has(#comparison-tabs) + div.element-container {
                margin-top: -0.45rem !important;
            }
            /* right-rail: predictions panel offset (columns side-by-side — matches desktop pull) */
            div.element-container:has(#comparison-right-rail) + div {
                margin-top: -2.4rem !important;
            }
        }

        /* BAND B (1025–1280px) — sidebar overlay, columns now STACKED (chart spans full width).
           Chart is full-width; controls row, tabs, and predictions panel stack below it.
           right-rail margin-top now matters: positive = space above the predictions heading. */
        @media (min-width: 1025px) and (max-width: 1280px) {
            /* controls-row: pulls "Whole career / Metric Selections" row up under the chart */
            [data-testid="stHorizontalBlock"]:has(#comparison-season-filter) {
                margin-top: -2.4rem !important;
            }
            /* controls-panel: Metric Selections button fine-offset within the controls row */
            div:has(> #comparison-controls-panel) + div {
                margin-top: 0 !important;
            }
            /* detail-layout: pulls Overview / Current Standings section up toward chart bottom */
            div.element-container:has(#comparison-detail-layout) {
                margin-top: -3.7rem !important;
            }
            /* tabs anchor: gap above the invisible tab-anchor div (above the pill row) */
            div.element-container:has(#comparison-tabs) {
                margin-top: -0.6rem !important;
            }
            /* tabs + div: gap above the actual visible tab-pill bar */
            div.element-container:has(#comparison-tabs) + div.element-container {
                margin-top: -0.45rem !important;
            }
            /* right-rail: predictions panel top offset (stacked — predictions sit below chart) */
            div.element-container:has(#comparison-right-rail) + div {
                margin-top: 1rem !important;
            }
        }

        /* BAND C (769–1024px) — sidebar overlay, columns STACKED, narrower viewport.
           Structurally identical to Band B but narrower; tune these values independently
           if Band B and Band C need to diverge. */
        @media (min-width: 769px) and (max-width: 1024px) {
            /* controls-row: pulls "Whole career / Metric Selections" row up under the chart */
            [data-testid="stHorizontalBlock"]:has(#comparison-season-filter) {
                margin-top: -2.4rem !important;
            }
            /* controls-panel: Metric Selections button fine-offset within the controls row */
            div:has(> #comparison-controls-panel) + div {
                margin-top: 0 !important;
            }
            /* detail-layout: pulls Overview / Current Standings section up toward chart bottom */
            div.element-container:has(#comparison-detail-layout) {
                margin-top: -3.7rem !important;
            }
            /* tabs anchor: gap above the invisible tab-anchor div (above the pill row) */
            div.element-container:has(#comparison-tabs) {
                margin-top: -0.6rem !important;
            }
            /* tabs + div: gap above the actual visible tab-pill bar */
            div.element-container:has(#comparison-tabs) + div.element-container {
                margin-top: -0.45rem !important;
            }
            /* right-rail: predictions panel top offset (stacked — predictions sit below chart) */
            div.element-container:has(#comparison-right-rail) + div {
                margin-top: 1rem !important;
            }
        }
        /* DESKTOP — predictions panel anchor: collapses zero-height anchor spacing */
        div.element-container:has(#comparison-predictions-panel) {
            margin: 0 !important;
            line-height: 0 !important;
        }
        div.element-container:has(#comparison-predictions-panel) + div.element-container {
            margin-top: 0 !important;
        }
        /* DESKTOP — right-rail anchor: collapses anchor, pulls predictions panel to top of column */
        div.element-container:has(#comparison-right-rail) {
            margin: 0 !important;
            line-height: 0 !important;
        }
        div.element-container:has(#comparison-right-rail) + div {
            margin-top: -2.4rem !important;
        }
        /* DESKTOP — main Plotly chart anchor: collapses anchor + removes bottom gap under chart */
        div.element-container:has(#comparison-main-plotly) {
            margin: 0 !important;
            line-height: 0 !important;
        }
        div.element-container:has(#comparison-main-plotly) + div.element-container {
            margin-top: 0 !important;
            margin-bottom: -2.15rem !important;
            line-height: 0 !important;
        }
        div.element-container:has(#comparison-main-plotly) + div.element-container [data-testid="stPlotlyChart"] {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            line-height: normal !important;
        }
        /* DESKTOP — main Plotly chart: border, radius, overflow, transition */
        div[data-testid="stPlotlyChart"] {
            border: 1px solid rgba(70, 84, 122, 0.5);
            border-radius: 14px;
            overflow: hidden;
            transition: box-shadow 0.3s ease, background 0.3s ease;
        }
        /* DESKTOP — detail layout anchor: negative top margin pulls detail section up under chart */
        div.element-container:has(#comparison-detail-layout) {
            margin: -3.7rem 0 0 0 !important;
            line-height: 0 !important;
        }
        div.element-container:has(#comparison-detail-layout) + div.element-container {
            margin-top: 0 !important;
        }

        /* DESKTOP — Overview / Current Standings tab row: styles, spacing, pill tab buttons */
        div.element-container:has(#comparison-tabs) {
            margin: -0.6rem 0 0 0 !important;
            line-height: 0 !important;
        }
        div.element-container:has(#comparison-tabs) + div.element-container {
            margin-top: -0.45rem !important;
        }
        div.element-container:has(#comparison-tabs) + div.element-container [data-testid="stTabs"] {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        div.element-container:has(#comparison-tabs) + div.element-container [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.35rem !important;
            flex-wrap: wrap !important;
            margin-bottom: 0.22rem !important;
            min-height: 40px !important;
            align-items: center !important;
            padding-top: 0 !important;
        }
        div.element-container:has(#comparison-tabs) + div.element-container [data-testid="stTabs"] [data-baseweb="tab-border"] {
            display: none !important;
        }
        div.element-container:has(#comparison-tabs) + div.element-container [data-testid="stTabs"] button[role="tab"] {
            margin: 0 !important;
            border: 1px solid #2a2a2a !important;
            border-radius: 999px !important;
            background: rgba(17, 24, 39, 0.7) !important;
            padding: 4px 10px !important;
            min-height: 0 !important;
            height: auto !important;
        }
        div.element-container:has(#comparison-tabs) + div.element-container [data-testid="stTabs"] button[role="tab"] p {
            margin: 0 !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            color: #d9d9d9 !important;
        }
        div.element-container:has(#comparison-tabs) + div.element-container [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            border-color: #2596be !important;
            background: rgba(37, 150, 190, 0.14) !important;
        }
        div.element-container:has(#comparison-tabs) + div.element-container [data-testid="stTabs"] [data-baseweb="tab-panel"] {
            padding-top: 0.1rem !important;
        }
        /* MOBILE — tabs section: stronger pull on detail-layout anchor; tighter tab spacing + padding */
        @media (max-width: 768px) {
            div.element-container:has(#comparison-detail-layout) {
                margin-top: -4.1rem !important;  /* adjust to move Overview/Current Standings up/down */
            }
            div.element-container:has(#comparison-tabs) {
                margin-top: -0.12rem !important;  /* adjust gap above the tab anchor itself */
            }
            div.element-container:has(#comparison-tabs) + div.element-container {
                margin-top: -0.2rem !important;  /* adjust gap above the tab bar */
            }
            div.element-container:has(#comparison-tabs) + div.element-container [data-testid="stTabs"] button[role="tab"] {
                padding: 3px 8px !important;  /* adjust inner padding of each tab pill */
            }
            /* right-rail: "Next matches prediction" top offset on mobile.
               Negative = pulls panel UP (less space above heading); positive = pushes DOWN. */
            div.element-container:has(#comparison-right-rail) + div {
                margin-top: -3rem !important;  /* ← adjust to move "Next matches prediction" up/down */
            }
        }

        /* DESKTOP — main layout anchor: collapses anchor + pulls chart/panel row up */
        div:has(> #main-chart-layout) {
            margin: 0 !important;
            line-height: 0 !important;
        }
        div:has(> #main-chart-layout) + div {
            margin-top: -0.45rem !important;
        }
        /* MOBILE — main layout: cancels desktop negative pull so toolbar clears Streamlit header */
        @media (max-width: 768px) {
            div:has(> #main-chart-layout) + div {
                margin-top: 0rem !important;
            }
        }
        div:has(> #main-chart-layout) + div [data-testid="stHorizontalBlock"] {
            align-items: flex-start !important;
        }
        /* TABLET (≤1280px) — main layout: chart and comparison panel stack vertically */
        @media screen and (max-width: 1280px) {
            div:has(> #main-chart-layout) + div [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
            }
            div:has(> #main-chart-layout) + div [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                min-width: 100% !important;
                width: 100% !important;
                flex: 1 1 100% !important;
            }
        }

        /* MOBILE — full responsive block: stacks all columns, resizes cards, standings grid */
        @media screen and (max-width: 768px) {
            .main .block-container {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            .main [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
            }
            .main [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                min-width: 100% !important;
                width: 100% !important;
            }
            .comparison-player-card {
                flex-direction: column;
                gap: 0.7rem;
                padding: 0.8rem;
                margin-bottom: 0.55rem;
            }
            div.element-container:has(.live-game-card) {
                margin-bottom: 1.1rem !important;
            }
            .comparison-player-card__media {
                flex-basis: auto;
                max-width: 100%;
            }
            .comparison-player-card__media--player {
                width: min(100%, 118px);
                min-height: 90px;
                margin: 0 auto;
            }
            .comparison-player-card__image--player-cutout {
                height: 100px;
                transform: translateY(6px) scale(1.14);
            }
            .comparison-card-stats {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .comparison-player-card__media--team {
                flex-basis: auto;
                max-width: 124px;
                margin: 0 auto;
            }
            .comparison-player-card__image--team-logo {
                padding: 0.3rem;
            }
            .stanley-cup-board-grid {
                grid-template-columns: 1fr;
                gap: 0.8rem;
            }
            .stanley-cup-division-header {
                padding: 0.8rem 0.85rem 0.62rem 0.85rem;
            }
            .stanley-cup-table-head,
            .stanley-cup-row {
                grid-template-columns: minmax(0, 1.7fr) repeat(5, minmax(28px, 0.42fr));
                gap: 0.28rem;
            }
            .stanley-cup-table-head {
                padding: 0 0.85rem 0.44rem 0.85rem;
                font-size: 0.69rem;
            }
            .stanley-cup-row {
                margin: 0 0.38rem;
                padding: 0.66rem 0.36rem;
            }
            .stanley-cup-team-logo {
                width: 22px;
                height: 22px;
                flex-basis: 22px;
            }
            .stanley-cup-row-team-name {
                font-size: 0.88rem;
            }
            .stanley-cup-row-badge {
                display: none;
            }
            .stanley-cup-board-meta {
                font-size: 0.76rem;
            }
            .comparison-trace-toggle {
                gap: 0.45rem;
                padding: 0.3rem 0.64rem;
                font-size: 0.74rem;
            }
            .comparison-trace-toggle--icon-only {
                min-width: 2.2rem;
                padding: 0.3rem 0.52rem;
            }
            .comparison-trace-toggle--compact {
                padding: 0.26rem 0.56rem;
                font-size: 0.7rem;
            }
        }

        /* DESKTOP — Plotly modebar: always visible, pinned top-right, flat style */
        .js-plotly-plot .plotly .modebar {
            opacity: 1 !important;
            top: 8px !important;
            right: 8px !important;
            left: auto !important;
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            padding: 0 !important;
            min-height: 30px !important;
            line-height: 1 !important;
            overflow: visible !important;
            display: flex !important;
            align-items: center !important;
        }
        .js-plotly-plot .plotly .modebar-btn::before,
        .js-plotly-plot .plotly .modebar-btn::after {
            display: none !important;
            content: none !important;
        }
        .js-plotly-plot .plotly .modebar-group {
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            overflow-y: visible !important;
            padding: 0 !important;
            line-height: 1 !important;
            display: flex !important;
            align-items: center !important;
        }
        .js-plotly-plot .plotly .modebar-btn {
            padding: 6px 8px !important;
            min-height: 30px !important;
            line-height: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .js-plotly-plot .plotly .modebar-btn svg {
            width: 18px !important;
            height: 18px !important;
        }
        /* MOBILE — Plotly modebar: smaller icons and tighter padding */
        @media (max-width: 768px) {
            .js-plotly-plot .plotly .modebar {
                top: 4px !important;
                right: 4px !important;
            }
            .js-plotly-plot .plotly .modebar-btn {
                padding: 3px 5px !important;
                min-height: 22px !important;
            }
            .js-plotly-plot .plotly .modebar-btn svg {
                width: 14px !important;
                height: 14px !important;
            }
        }

        /* TABLET/FOLDABLE (≤1400px or narrow aspect ratio) — sidebar becomes fixed overlay to prevent layout reflow */
        @media screen and (max-width: 1400px), screen and (max-width: 1600px) and (max-aspect-ratio: 11/10) {
            :root {
                --pp-overlay-sidebar-top: calc(env(safe-area-inset-top, 0px) + 3.75rem);
            }
            .comparison-player-card__media--player {
                display: none !important;
            }
            [data-testid="stAppViewContainer"] {
                overflow-x: clip !important;
            }
            section[data-testid="stSidebar"] {
                position: fixed !important;
                inset: var(--pp-overlay-sidebar-top) auto 0 0 !important;
                height: calc(100dvh - var(--pp-overlay-sidebar-top)) !important;
                z-index: 1002 !important;
            }
            section[data-testid="stSidebar"] > div:first-child {
                height: calc(100dvh - var(--pp-overlay-sidebar-top)) !important;
                box-shadow: 18px 0 36px rgba(8, 12, 22, 0.42) !important;
            }
            section[data-testid="stMain"] {
                width: 100% !important;
                max-width: 100% !important;
            }
            [data-testid="collapsedControl"] {
                position: fixed !important;
                top: calc(var(--pp-overlay-sidebar-top) + 0.35rem) !important;
                left: 0.75rem !important;
                z-index: 1003 !important;
            }
            [data-testid="stSidebarCollapseButton"] {
                z-index: 1003 !important;
            }
        }

        /* DESKTOP — sidebar toggle button: always visible (Streamlit hides it by default on hover) */
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="collapsedControl"] {
            opacity: 1 !important;
            visibility: visible !important;
        }
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="collapsedControl"] button {
            min-width: 36px;
            min-height: 36px;
        }

        /* ── Custom animated progress bar for cache spinners ───────────── */
        /* Hide the default "Running function_name()" text */
        [data-testid="stSpinner"] .stMarkdown p {
            display: none !important;
        }
        /* Replace with animated progress bar */
        [data-testid="stSpinner"] {
            position: relative !important;
            width: 100% !important;
            max-width: 400px !important;
            margin: 1rem auto !important;
        }
        [data-testid="stSpinner"]::before {
            content: '';
            display: block;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg,
                #2b71c7 0%,
                #2596be 50%,
                #2b71c7 100%);
            background-size: 200% 100%;
            border-radius: 2px;
            animation: progress-sweep 2s ease-in-out infinite;
        }
        [data-testid="stSpinner"]::after {
            content: 'Loading data...';
            display: block;
            text-align: center;
            font-size: 14px;
            color: #888;
            margin-top: 8px;
        }
        @keyframes progress-sweep {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        /* Keep the spinner icon itself hidden since we have our own animation */
        [data-testid="stSpinner"] > div:first-child {
            display: none !important;
        }

        /* BAND D (600–768px) — overrides problematic mobile rules for this range.
           CSS cascade: this block appears AFTER all @media (max-width: 768px) blocks
           so it wins for 600–768px. True mobile (≤599px) is completely unaffected.
           To shift the lower boundary: change the 600px min-width value below.
           Band C upper bound (1024px) and the mobile threshold (768px) stay unchanged. */
        @media (min-width: 600px) and (max-width: 768px) {
            /* chart-toolbar top margin: pushes the toolbar (title + Copy link) below the
               Streamlit header bar. Increase if the toolbar is hidden under the header;
               decrease if there is too much empty space above the chart.
               Reference: true mobile uses 3.75rem; desktop uses 0. */
            .nhl-chart-toolbar {
                margin: 3.5rem 0 0.1rem 0;  /* ← adjust first value to move toolbar up/down */
            }
            /* chart-toolbar block-container top padding: extra lever if margin alone isn't
               enough to clear the header. Increase to push everything down, 0 to reset. */
            .block-container {
                padding-top: 0.5rem !important;  /* ← adjust to shift entire page content down */
            }
            /* controls-panel column: cancel the -1.7rem mobile pull that causes
               Metric Selections to fly off-screen at this viewport width. */
            [data-testid="stColumn"]:has(#comparison-controls-panel) {
                margin-top: 0 !important;
            }
            /* Metric Selections button: force single line — prevents text wrapping to two
               lines at this viewport width. If you want it to wrap instead, remove this. */
            [data-testid="stPopover"] button {
                white-space: nowrap !important;
            }
            /* Metric Selections button font size: shrink if button is still too wide.
               Default Streamlit size is ~0.875rem; go lower to squeeze it smaller. */
            [data-testid="stPopover"] button p {
                font-size: 1rem !important;  /* ← adjust to shrink/grow button label */
            }
            /* controls-row: "Whole career / Metric Selections" gap from chart bottom */
            [data-testid="stHorizontalBlock"]:has(#comparison-season-filter) {
                margin-top: -2.4rem !important;
            }
            /* controls-panel: Metric Selections button fine-offset within the row */
            div:has(> #comparison-controls-panel) + div {
                margin-top: 0 !important;
            }
            /* detail-layout: pulls Overview / Current Standings section up toward chart */
            div.element-container:has(#comparison-detail-layout) {
                margin-top: -3.7rem !important;
            }
            /* tabs anchor: gap above the invisible tab-anchor div (above pill row) */
            div.element-container:has(#comparison-tabs) {
                margin-top: -0.6rem !important;
            }
            /* tabs + div: gap above the visible tab-pill bar */
            div.element-container:has(#comparison-tabs) + div.element-container {
                margin-top: -0.45rem !important;
            }
            /* right-rail: predictions panel top offset (stacked layout) */
            div.element-container:has(#comparison-right-rail) + div {
                margin-top: 0.5rem !important;
            }
        }

        /* ── Skeleton loaders (paint-first placeholders) ──────────────── */
        /* Base shimmer block: grey background overlaid with a sweeping
           gradient via ::after. Real content swaps in by clearing the
           hosting st.empty() slot and re-rendering inside it. */
        .pp-skel-wrap {
            width: 100%;
            box-sizing: border-box;
        }
        .pp-skel-block {
            position: relative;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            width: 100%;
        }
        .pp-skel-block::after {
            content: '';
            position: absolute;
            inset: 0;
            background-image: linear-gradient(
                90deg,
                transparent 0%,
                rgba(255, 255, 255, 0.08) 50%,
                transparent 100%
            );
            transform: translateX(-100%);
            animation: pp-skel-shimmer 1.4s ease-in-out infinite;
        }
        @keyframes pp-skel-shimmer {
            0%   { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        /* Per-component sizing — heights tuned to match real content
           so the swap causes no layout drift. */
        .pp-skel-chart-toolbar { height: 28px; margin: 0 0 12px 0; max-width: 320px; }
        .pp-skel-chart         { width: 100%; }
        .pp-skel-tabs          { display: flex; gap: 8px; margin: 0 0 12px 0; }
        .pp-skel-pill          { height: 32px; width: 140px; border-radius: 16px; }
        .pp-skel-grid          { display: flex; flex-direction: column; gap: 10px; }
        .pp-skel-card          { height: 56px; }
        .pp-skel-heading       { height: 22px; margin: 0 0 12px 0; max-width: 220px; }
        .pp-skel-stack         { display: flex; flex-direction: column; gap: 10px; }
        .pp-skel-game          { height: 64px; }

        /* Lazy-load headshot utility: grey circle visible until the real
           <img loading="lazy"> decodes and paints on top. No JS. */
        .pp-skel-headshot-wrap {
            position: relative;
            display: inline-block;
            width: 32px;
            height: 32px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 50%;
            overflow: hidden;
            flex-shrink: 0;
        }
        .pp-skel-headshot-wrap > img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 50%;
            display: block;
        }

        /* Accessibility: respect prefers-reduced-motion */
        @media (prefers-reduced-motion: reduce) {
            .pp-skel-block::after { animation: none; }
        }
    </style>
"""
"""Full CSS block injected into the Streamlit page head."""


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def get_favicon_path() -> Path:
    """Return the absolute path to the custom site favicon asset.

    Args:
        None.

    Returns:
        Path: Absolute path to the favicon SVG file in the repository assets folder.
    """
    return Path(__file__).resolve().parent.parent / "assets" / "favicon.svg"


def get_header_logo_path() -> Path:
    """Return the absolute path to the preferred brand logo PNG.

    Returns:
        Path: Absolute path to the PuckPeak logo file in the repository assets folder.
    """
    return Path(__file__).resolve().parent.parent / "assets" / "PP.png"


def get_header_logo_data_uri() -> str:
    """Return the brand logo PNG as an inline image data URI.

    Returns:
        str: Base64-encoded PNG data URI for embedding the local brand image,
            or an empty string if the PNG asset is unavailable.
    """
    logo_path = get_header_logo_path()
    if logo_path.exists():
        logo_bytes = logo_path.read_bytes()
        return f"data:image/png;base64,{base64.b64encode(logo_bytes).decode('ascii')}"

    return ""


def get_bb_logo_data_uri() -> str:
    """Return the BB brand logo PNG as an inline image data URI.

    Returns:
        str: Base64-encoded PNG data URI for embedding the BB logo,
            or an empty string if the PNG asset is unavailable.
    """
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "BB.png"
    if logo_path.exists():
        logo_bytes = logo_path.read_bytes()
        return f"data:image/png;base64,{base64.b64encode(logo_bytes).decode('ascii')}"
    return ""


def inject_header_bb_logo() -> None:
    """Inject BB.png centered in the Streamlit top header bar via CSS pseudo-element.

    The logo is positioned absolutely at the horizontal center of the sticky
    header and sized to fit neatly within the bar height.
    Must be called once per app run, after st.set_page_config().
    """
    data_uri = get_bb_logo_data_uri()
    if not data_uri:
        return
    css = f"""
    <style>
        /* CENTER — BB logo pinned to the middle of the Streamlit top header bar */
        [data-testid="stHeader"] {{
            position: relative;
            margin-bottom: -5rem;  /* ← negative value pulls page content up (e.g. -1rem) */
        }}
        [data-testid="stHeader"]::after {{
            content: "";
            display: block;
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 270px;
            height: 48px;
            background-image: url('{data_uri}');
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            pointer-events: none;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def inject_css() -> None:
    """Inject the Puck Peak custom CSS into the Streamlit page.

    Covers: sidebar brand logo styling, tighter top spacing, sidebar compact/overlay layout,
    blue Add-Legend button override, compact controls toolbar styling,
    compact mobile header sizing,
    a real chart toolbar row with copy-link button, responsive stacking of the
    chart/stats panel split on laptop and mobile widths, and Plotly modebar sizing.

    Must be called once per app run, after st.set_page_config().
    """
    st.markdown(_CSS, unsafe_allow_html=True)


def inject_mobile_dropdown_fix() -> None:
    """Inject the CSS-only mobile dropdown fix after page config is set."""
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
