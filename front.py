import os
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from streamlit_geolocation import streamlit_geolocation

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="Camera + Geotag Capture", page_icon="📷", layout="centered")

st.markdown("""
# 📷 Camera + Geotag Capture
Take a photo, fetch your current location (with your permission), and stamp the image with the time and place.

> Privacy note: Location is requested by your browser and only used in this app. Photos and metadata are stored locally on the server this app runs on.
""")

# ----------------------------
# Utilities
# ----------------------------
@dataclass
class CaptureRecord:
    timestamp_local_iso: str
    timestamp_utc_iso: str
    latitude: Optional[float]
    longitude: Optional[float]
    accuracy_m: Optional[float]
    address: Optional[str]
    image_path: Optional[str]
    overlay_path: Optional[str]


def now_local():
    tz = ZoneInfo("Asia/Kolkata")  # adjust if you need a different default
    return datetime.now(tz)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    try:
        geolocator = Nominatim(user_agent="streamlit-camera-geotag")
        reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
        location = reverse((lat, lon), language="en")
        return location.address if location else None
    except Exception:
        return None


def draw_overlay(img: Image.Image, text: str) -> Image.Image:
    # Create a copy so original remains unchanged
    im = img.convert("RGB").copy()
    draw = ImageDraw.Draw(im)

    # Font setup (fallback-safe)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # Word-wrap text to fit image width
    max_width = im.width - 20
    lines = []
    words = text.split()
    while words:
        line = words.pop(0)
        while words and draw.textlength(line + " " + words[0], font=font) <= max_width:
            line += " " + words.pop(0)
        lines.append(line)

    # Compute text block size
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    text_height = sum(line_heights) + (len(lines) - 1) * 4
    text_width = max(draw.textlength(line, font=font) for line in lines)

    # Draw semi-transparent background rectangle at bottom
    margin = 10
    rect_height = text_height + 2 * margin
    rect_width = text_width + 2 * margin
    x0 = 10
    y0 = im.height - rect_height - 10
    x1 = x0 + rect_width
    y1 = y0 + rect_height

    # Black rectangle with 70% opacity overlay
    overlay = Image.new("RGBA", (rect_width, rect_height), (0, 0, 0, 180))
    im_rgba = im.convert("RGBA")
    im_rgba.paste(overlay, (x0, y0), overlay)
    draw = ImageDraw.Draw(im_rgba)

    # Draw text in white
    y = y0 + margin
    for line in lines:
        draw.text((x0 + margin, y), line, font=font, fill=(255, 255, 255, 255))
        y += (draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]) + 4

    return im_rgba.convert("RGB")


# ----------------------------
# Session State
# ----------------------------
if "records" not in st.session_state:
    st.session_state.records = []  # list[CaptureRecord]

# Storage
ensure_dir("captures")

# ----------------------------
# UI - Capture
# ----------------------------
st.subheader("1) Take a photo")
photo = st.camera_input("Tap to open your camera and take a snapshot")
