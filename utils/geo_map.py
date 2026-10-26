"""
utils/geo_map.py
────────────────
Shared helpers for:
  * Capturing device GPS via the browser Geolocation API — FULLY AUTOMATIC
    via streamlit-js-eval (coordinates populate the fields directly, no copy-paste)
  * Building and rendering an interactive Folium / OpenStreetMap damage map
  * Managing the in-session damage-record store (session_state backed)
  * Deduplication and CSV export
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import List, Optional

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ── Colour palette (matches CLASS_COLORS in model_loader) ─────────────────────
_CLASS_ICON_COLOR: dict = {
    "Longitudinal Crack": "red",
    "Transverse Crack":   "orange",
    "Alligator Crack":    "green",
    "Potholes":           "blue",
}
_CLASS_HEX: dict = {
    "Longitudinal Crack": "#FF6B6B",
    "Transverse Crack":   "#FFD93D",
    "Alligator Crack":    "#6BCB77",
    "Potholes":           "#00D4FF",
}
_CLASS_EMOJI: dict = {
    "Longitudinal Crack": "🔴",
    "Transverse Crack":   "🟡",
    "Alligator Crack":    "🟢",
    "Potholes":           "🔵",
}

# ── Session-state key ──────────────────────────────────────────────────────────
_RECORDS_KEY = "rp_damage_records"


def _init_records() -> None:
    if _RECORDS_KEY not in st.session_state:
        st.session_state[_RECORDS_KEY] = []


def get_records() -> List[dict]:
    _init_records()
    return st.session_state[_RECORDS_KEY]


def clear_records() -> None:
    st.session_state[_RECORDS_KEY] = []


def _dedup_key(lat: float, lon: float, label: str, score: float) -> str:
    """Hash key: same location (±0.0001 deg) + class + score bucket (1 dp)."""
    raw = f"{round(lat, 4)}|{round(lon, 4)}|{label}|{round(score, 1)}"
    return hashlib.md5(raw.encode()).hexdigest()


def add_detection(lat: float, lon: float, label: str, score: float, source: str = "image") -> bool:
    """
    Add a detection record. Returns True if it was new (not a duplicate).
    Silently skips (0, 0) sentinel coordinates.
    """
    if lat == 0.0 and lon == 0.0:
        return False
    _init_records()
    key = _dedup_key(lat, lon, label, score)
    existing_keys = {r["_key"] for r in st.session_state[_RECORDS_KEY]}
    if key in existing_keys:
        return False
    st.session_state[_RECORDS_KEY].append({
        "_key":       key,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":     source,
        "label":      label,
        "confidence": round(score, 4),
        "latitude":   lat,
        "longitude":  lon,
    })
    return True


def records_as_df() -> pd.DataFrame:
    records = get_records()
    if not records:
        return pd.DataFrame(columns=["timestamp", "source", "label", "confidence", "latitude", "longitude"])
    df = pd.DataFrame(records)
    return df[["timestamp", "source", "label", "confidence", "latitude", "longitude"]]


# ── JS expression that returns {lat, lon, acc} via the Geolocation API ────────
_GEO_JS = """
new Promise((resolve) => {
  if (!navigator.geolocation) { resolve(null); return; }
  navigator.geolocation.getCurrentPosition(
    (pos) => resolve({
      lat: pos.coords.latitude,
      lon: pos.coords.longitude,
      acc: pos.coords.accuracy || 0
    }),
    (err) => resolve({ error: err.message }),
    { enableHighAccuracy: true, timeout: 12000, maximumAge: 0 }
  );
})
"""


def _fetch_ip_location() -> tuple:
    """
    Fallback: get city-level lat/lon from the device's public IP.
    Uses ip-api.com (free, no API key, ~city-level accuracy).
    Returns (result_dict | None, error_str | None).
    """
    try:
        import requests as _req
        r = _req.get("http://ip-api.com/json/?fields=status,lat,lon,city,regionName,country",
                     timeout=8)
        data = r.json()
        print(f"[RoadPulseAI IP-GEO] Response: {data}")
        if data.get("status") == "success":
            return {
                "lat": float(data["lat"]),
                "lon": float(data["lon"]),
                "city": data.get("city", ""),
                "region": data.get("regionName", ""),
                "country": data.get("country", ""),
                "source": "ip",
            }, None
        else:
            return None, f"ip-api returned status: {data.get('status','unknown')}"
    except Exception as e:
        print(f"[RoadPulseAI IP-GEO] Error: {e}")
        return None, str(e)


def _is_bad_gps(lat: float, lon: float) -> bool:
    """
    Returns True if GPS coordinates are clearly invalid:
      - Both near (0, 0)  → Null Island
      - Longitude exactly 0 or near 0  → browser failed to get real fix
      - Latitude exactly 0 or near 0   → browser failed to get real fix
    """
    return abs(lon) < 1.0 or abs(lat) < 1.0

# ── Fallback HTML component (when streamlit-js-eval is not installed) ──────────
_GEO_COMPONENT_HTML = """
<!DOCTYPE html><html><head>
<style>
  body { margin:0; padding:4px 0; font-family: sans-serif; background: transparent; }
  #status { font-size: 0.82rem; color: #8888AA; }
  #coords { font-size: 0.85rem; color: #00D4FF; font-weight: 600; margin-top: 4px; }
</style></head><body>
<div id="status">📍 Requesting GPS location from browser…</div>
<div id="coords"></div>
<script>
(function() {
  var statusEl = document.getElementById("status");
  var coordsEl = document.getElementById("coords");
  if (!navigator.geolocation) {
    statusEl.innerHTML = "❌ Geolocation not supported by this browser.";
    return;
  }
  navigator.geolocation.getCurrentPosition(
    function(pos) {
      var lat = pos.coords.latitude.toFixed(6);
      var lon = pos.coords.longitude.toFixed(6);
      var acc = pos.coords.accuracy ? pos.coords.accuracy.toFixed(0) + "m" : "N/A";
      statusEl.innerHTML = "✅ GPS acquired — copy coordinates into the fields above.";
      coordsEl.innerHTML = "Lat: <b>" + lat + "</b> &nbsp;|&nbsp; Lon: <b>" + lon + "</b> &nbsp;|&nbsp; ±" + acc;
    },
    function(err) {
      var msg = "⚠️ Location error.";
      if (err.code === 1) msg = "⚠️ Permission denied — allow location in browser settings and reload.";
      else if (err.code === 2) msg = "⚠️ GPS unavailable — check device settings.";
      else if (err.code === 3) msg = "⚠️ Location request timed out.";
      statusEl.innerHTML = msg;
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
  );
})();
</script></body></html>
"""


def render_geo_input(key_suffix: str = "") -> tuple:
    """
    Renders the GPS input UI.  When streamlit-js-eval is installed the
    Auto-detect button fetches coordinates from the browser silently and
    writes them directly into session_state → the number inputs auto-populate
    and the page reruns so the map plots immediately.  No copy-paste required.

    Falls back to the embedded HTML component (manual copy-paste) when the
    package is not installed.

    Returns (lat, lon) as floats.
    """
    _init_records()

    # ── Try to import streamlit-js-eval ───────────────────────────────────────
    try:
        from streamlit_js_eval import streamlit_js_eval as _js_eval
        _has_js_eval = True
    except ImportError:
        _has_js_eval = False

    # ── Auto-fetch IP location on first load if no coordinates yet ─────────────
    _auto_fetched_key = f"_ip_auto_fetched_{key_suffix}"
    if not st.session_state.get(_auto_fetched_key, False):
        st.session_state[_auto_fetched_key] = True   # only do this once per session
        if st.session_state.get("rp_lat", 0.0) == 0.0 and st.session_state.get("rp_lon", 0.0) == 0.0:
            with st.spinner("📍 Detecting your location…"):
                ip_result, _ = _fetch_ip_location()
            if ip_result:
                st.session_state["rp_lat"] = ip_result["lat"]
                st.session_state["rp_lon"] = ip_result["lon"]
                st.session_state[f"_geo_ok_{key_suffix}"] = True
                st.session_state[f"_geo_acc_{key_suffix}"] = 0.0
                st.session_state[f"_geo_city_{key_suffix}"] = (
                    f"{ip_result.get('city','')}, {ip_result.get('country','')}"
                )
                print(f"[RoadPulseAI GPS] Auto-detected — Lat: {ip_result['lat']}  Lon: {ip_result['lon']}")
                st.rerun()


    # ── Header card ───────────────────────────────────────────────────────────
    st.markdown(
        """<div style="background:#12121A;border:1px solid #1E1E2E;border-radius:10px;
        padding:0.8rem 1.2rem;margin-bottom:0.8rem;">
        <b style="color:#E8E8F0;">📍 Device GPS Location</b>
        <div style="font-size:0.8rem;color:#6666AA;margin-top:4px;">
        Click <b style="color:#00D4FF;">🛰️ Auto-detect</b> to fetch your GPS location from the browser,
        or enter coordinates manually below.
        </div></div>""",
        unsafe_allow_html=True,
    )

    col_lat, col_lon, col_btn = st.columns([2, 2, 1])

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        detect_clicked = st.button(
            "🛰️ Auto-detect",
            key=f"geo_detect_{key_suffix}",
            use_container_width=True,
            help="Fetch GPS location from your browser — click Allow when prompted",
        )

    # ── Auto-detect flow (browser GPS via streamlit-js-eval) ──────────────────
    _pending_key = f"_geo_pending_{key_suffix}"

    if detect_clicked:
        if _has_js_eval:
            st.session_state[_pending_key] = True
        else:
            st.session_state[f"_show_geo_{key_suffix}"] = True

    if _has_js_eval and st.session_state.get(_pending_key, False):
        st.info("📡 Waiting for browser GPS… (click **Allow** if prompted by browser)")
        result = _js_eval(js_expressions=_GEO_JS, key=f"_geo_js_{key_suffix}")
        print(f"[RoadPulseAI GPS] JS eval result: {result}")

        if result is not None:
            st.session_state[_pending_key] = False
            if isinstance(result, dict) and "error" not in result:
                lat_val = float(result.get("lat", 0.0))
                lon_val = float(result.get("lon", 0.0))
                acc_val = float(result.get("acc", 0.0))
                print(f"[RoadPulseAI GPS] Browser GPS: lat={lat_val}, lon={lon_val}, acc={acc_val}")

                # ── Bad GPS fix (usually HTTP vs HTTPS issue) ──────────────────
                if _is_bad_gps(lat_val, lon_val):
                    st.warning(
                        f"⚠️ GPS returned an invalid fix "
                        f"(Lat: {lat_val:.4f}, Lon: {lon_val:.4f}). "
                        "Browser GPS requires **HTTPS** — enter coordinates manually below."
                    )
                else:
                    st.session_state["rp_lat"] = lat_val
                    st.session_state["rp_lon"] = lon_val
                    st.session_state[f"_geo_ok_{key_suffix}"] = True
                    st.session_state[f"_geo_acc_{key_suffix}"] = acc_val
                    st.rerun()
            elif isinstance(result, dict) and "error" in result:
                err_msg = result["error"]
                print(f"[RoadPulseAI GPS] error: {err_msg}")
                st.error(f"❌ GPS error: {err_msg}")
                st.warning("💡 Browser GPS needs HTTPS. Enter coordinates manually below.")
                st.session_state[_pending_key] = False
        else:
            # result is still None → browser hasn't responded yet, keep waiting
            st.caption("⏳ Still waiting for browser response…")

    # ── Lat / Lon inputs (pre-filled from session_state) ──────────────────────
    with col_lat:
        lat = st.number_input(
            "Latitude",
            min_value=-90.0, max_value=90.0,
            value=float(st.session_state.get("rp_lat", 0.0)),
            format="%.6f",
            key=f"geo_lat_{key_suffix}",
            help="Your device latitude (-90 to 90)",
        )
    with col_lon:
        lon = st.number_input(
            "Longitude",
            min_value=-180.0, max_value=180.0,
            value=float(st.session_state.get("rp_lon", 0.0)),
            format="%.6f",
            key=f"geo_lon_{key_suffix}",
            help="Your device longitude (-180 to 180)",
        )

    # ── Always show current coordinates (even 0,0) so user can see status ──────
    st.markdown(
        f'<div style="background:#0A0A1A;border:1px solid #1E1E3A;border-radius:8px;'
        f'padding:0.6rem 1rem;margin-top:6px;font-family:monospace;font-size:0.9rem;">'
        f'<span style="color:#6666AA;">Current GPS →</span> '
        f'<span style="color:#00D4FF;">Lat: <b>{lat:.6f}</b></span>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<span style="color:#FFD93D;">Lon: <b>{lon:.6f}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Success banner ──────────────────────────────────────────────────
    if st.session_state.get(f"_geo_ok_{key_suffix}", False) and (lat != 0.0 or lon != 0.0):
        acc = st.session_state.get(f"_geo_acc_{key_suffix}", 0.0)
        acc_text = f" (±{acc:.0f} m)" if acc > 0 else ""
        st.markdown(
            f'<div style="background:#0D1F0D;border:1px solid #2A4A2A;border-radius:8px;'
            f'padding:0.5rem 1rem;margin-top:4px;font-size:0.83rem;color:#6BCB77;">'
            f'✅ 🛰️ GPS acquired — Lat: <b>{lat:.6f}</b> | Lon: <b>{lon:.6f}</b>{acc_text}</div>',
            unsafe_allow_html=True,
        )
        print(f"[RoadPulseAI GPS] Latitude: {lat:.6f}  Longitude: {lon:.6f}")

    # ── Fallback manual HTML component ────────────────────────────────────────
    if st.session_state.get(f"_show_geo_{key_suffix}", False):
        import streamlit.components.v1 as components
        st.info(
            "🌐 **Browser location prompt** — click **Allow** when asked. "
            "Coordinates will appear below; copy them into the fields above."
        )
        components.html(_GEO_COMPONENT_HTML, height=70)

    # ── Validate and warn ─────────────────────────────────────────────────────
    if lat == 0.0 and lon == 0.0:
        st.warning(
            "⚠️ Location is (0, 0) — click **🛰️ Auto-detect** or enter GPS coordinates manually. "
            "Detections at (0, 0) will not be plotted on the map."
        )

    # Persist
    st.session_state["rp_lat"] = lat
    st.session_state["rp_lon"] = lon
    return lat, lon


# ── Folium map builder ─────────────────────────────────────────────────────────

def _damage_icon(label: str) -> folium.Icon:
    color = _CLASS_ICON_COLOR.get(label, "gray")
    return folium.Icon(color=color, icon="exclamation-sign", prefix="glyphicon")


def build_map(
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom: int = 16,
) -> folium.Map:
    """Build a Folium/OpenStreetMap map with a current-location marker + all damage records."""
    valid_center = (
        center_lat is not None
        and center_lon is not None
        and not (center_lat == 0.0 and center_lon == 0.0)
    )

    if valid_center:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")
        # Current position blue pulsing circle
        folium.CircleMarker(
            location=[center_lat, center_lon],
            radius=12,
            color="#00D4FF",
            weight=2,
            fill=True,
            fill_color="#00D4FF",
            fill_opacity=0.35,
            tooltip="📍 Your current location",
            popup=folium.Popup(
                f"<b>Your Location</b><br>Lat: {center_lat:.6f}<br>Lon: {center_lon:.6f}",
                max_width=200,
            ),
        ).add_to(m)
    else:
        # Default wide view (India) — no centre set yet
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="OpenStreetMap")

    # Plot all stored records
    records = get_records()
    for r in records:
        label     = r["label"]
        conf      = r["confidence"]
        ts        = r["timestamp"]
        r_lat     = r["latitude"]
        r_lon     = r["longitude"]
        source    = r.get("source", "")
        hex_color = _CLASS_HEX.get(label, "#AAAAAA")
        emoji     = _CLASS_EMOJI.get(label, "⚠️")

        popup_html = (
            f'<div style="font-family:sans-serif;min-width:180px;">'
            f'<b style="color:{hex_color};">{emoji} {label}</b><br>'
            f'<hr style="border-color:#eee;margin:4px 0;">'
            f'<b>Confidence:</b> {conf:.1%}<br>'
            f'<b>Time:</b> {ts}<br>'
            f'<b>Source:</b> {source}<br>'
            f'<b>Lat:</b> {r_lat:.6f}<br>'
            f'<b>Lon:</b> {r_lon:.6f}'
            f'</div>'
        )
        folium.Marker(
            location=[r_lat, r_lon],
            icon=_damage_icon(label),
            tooltip=f"{emoji} {label} ({conf:.0%})",
            popup=folium.Popup(popup_html, max_width=240),
        ).add_to(m)

    return m


def render_map_section(center_lat: float, center_lon: float, map_key: str = "folium_map") -> None:
    """Render the full map UI: header, legend, interactive map, stats, CSV download, record table."""
    records = get_records()

    st.markdown(
        '<div class="page-title" style="font-size:1.6rem;margin-top:1rem;">🗺️ GPS Damage Map</div>',
        unsafe_allow_html=True,
    )

    # Legend row
    legend_parts = []
    for label, color in _CLASS_HEX.items():
        emoji = _CLASS_EMOJI.get(label, "⚠️")
        legend_parts.append(
            f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
            f'border-radius:6px;padding:2px 10px;font-size:0.8rem;font-weight:600;margin:3px 2px;">'
            f'{emoji} {label}</span>'
        )
    st.markdown(" ".join(legend_parts) + '&nbsp;&nbsp;<span style="color:#00D4FF;font-size:0.8rem;">⬤ Your location</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    m = build_map(center_lat, center_lon)
    st_folium(m, use_container_width=True, height=480, key=map_key, returned_objects=[])

    st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

    # Stats row
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{len(records)}</div>'
            f'<div class="stat-label">Total Records</div></div>',
            unsafe_allow_html=True,
        )
    with stat_col2:
        unique_classes = len({r["label"] for r in records})
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{unique_classes}</div>'
            f'<div class="stat-label">Damage Types</div></div>',
            unsafe_allow_html=True,
        )
    with stat_col3:
        avg_conf = sum(r["confidence"] for r in records) / len(records) if records else 0
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{avg_conf:.0%}</div>'
            f'<div class="stat-label">Avg Confidence</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        df = records_as_df()
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name="RoadPulseAI_damage_map.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=len(records) == 0,
            key=f"csv_dl_{map_key}",
        )
    with btn_col2:
        if st.button("🗑️ Clear Map Records", use_container_width=True,
                     type="secondary", key=f"clear_{map_key}"):
            clear_records()
            st.rerun()

    if records:
        st.markdown("**📋 Detection Records**")
        st.dataframe(records_as_df(), use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div style="color:#444466;font-size:0.88rem;text-align:center;padding:1rem 0;">'
            "No records yet — run a detection with valid GPS coordinates to populate the map.</div>",
            unsafe_allow_html=True,
        )
