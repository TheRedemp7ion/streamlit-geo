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
st.subheader("2) Get your location")
st.caption("Click the button below and allow your browser to share location.")
loc = streamlit_geolocation()
lat = loc.get("latitude") if isinstance(loc, dict) else None
lon = loc.get("longitude") if isinstance(loc, dict) else None
acc = loc.get("accuracy") if isinstance(loc, dict) else None

if lat and lon:
    st.success(f"Location: {lat:.6f}, {lon:.6f} (±{acc:.0f} m)" if acc else f"Location: {lat:.6f}, {lon:.6f}")
    # Quick map preview
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=0),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame({"lat": [lat], "lon": [lon]}),
                get_position='[lon, lat]',
                get_radius=10,
                pickable=True,
            )
        ],
    ))
else:
    st.info("No location yet. Click the 'Get Location' button above.")

# ----------------------------
# Process & Save
# ----------------------------
st.subheader("3) Save & stamp your capture")

ready = (photo is not None) and (lat is not None and lon is not None)
if not ready:
    st.warning("Waiting for both photo and location…")

if ready:
    # Time metadata
    t_local = now_local()
    t_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    local_iso = t_local.isoformat(timespec="seconds")
    utc_iso = t_utc.isoformat(timespec="seconds")

    # Reverse geocode (best-effort)
    with st.spinner("Resolving address…"):
        address = reverse_geocode(lat, lon)

    # Compose overlay text
    lines = [
        f"Local Time: {local_iso}",
        f"UTC Time:   {utc_iso}",
        f"Coords:     {lat:.6f}, {lon:.6f}",
    ]
    if acc:
        lines.append(f"Accuracy:   ±{acc:.0f} m")
    if address:
        lines.append(f"Address:    {address}")
    overlay_text = "\n".join(lines)

    # Load PIL image from uploaded file
    image = Image.open(photo)

    # Draw overlay
    stamped = draw_overlay(image, overlay_text)

    # Persist files
    stamp = t_local.strftime("%Y%m%d_%H%M%S")
    base = f"captures/capture_{stamp}"
    raw_path = f"{base}.jpg"
    overlay_path = f"{base}_stamped.jpg"
    image.save(raw_path, format="JPEG", quality=95)
    stamped.save(overlay_path, format="JPEG", quality=95)

    # Display results
    st.success("Captured & stamped!")
    st.image(stamped, caption="Stamped preview", use_column_width=True)

    # Downloads
    with open(overlay_path, "rb") as f:
        st.download_button("Download stamped image", f, file_name=os.path.basename(overlay_path), mime="image/jpeg")

    meta = CaptureRecord(
        timestamp_local_iso=local_iso,
        timestamp_utc_iso=utc_iso,
        latitude=float(lat),
        longitude=float(lon),
        accuracy_m=float(acc) if acc else None,
        address=address,
        image_path=raw_path,
        overlay_path=overlay_path,
    )

    meta_json = json.dumps(asdict(meta), indent=2)
    st.download_button("Download metadata (JSON)", meta_json, file_name=f"{base}.json", mime="application/json")

    # Save to session state table
    st.session_state.records.append(meta)

# ----------------------------
# History / Table
# ----------------------------
st.subheader("📒 Session history")
if st.session_state.records:
    df = pd.DataFrame([asdict(r) for r in st.session_state.records])
    st.dataframe(df, use_container_width=True)
    if st.button("Clear session history"):
        st.session_state.records = []
        st.experimental_rerun()
else:
    st.caption("No captures yet this session.")

st.markdown("---")
st.caption("Built with Streamlit • Reverse geocoding via OpenStreetMap Nominatim (best-effort)")
