# IMPORT CONFIG
import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import requests
import base64
import re
import urllib.parse
from typing import Optional, Dict, Any, Tuple

# APP CONFIG

st.set_page_config(
    page_title="FlySmart | Flight Tracker",
    page_icon="logo.png",
    layout="centered"
)
# BACKGROUND IMAGE

def set_background(image_file: str):
    try:
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            [data-testid="stHeader"], [data-testid="stToolbar"] {{
                background: rgba(0, 0, 0, 0);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass

set_background("background.jpg")

# GLOBAL UI RULES
st.markdown(
    """
    <style>
      .flysmart-hotel-img {
        width: 100%;
        height: 150px;
        object-fit: cover;
        border-radius: 12px;
        display: block;
        border: 1px solid rgba(15, 23, 42, 0.08);
      }
      .flysmart-hotel-card {
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 16px;
        padding: 14px 14px;
        box-shadow: 0 6px 18px rgba(15,23,42,0.06);
        margin-bottom: 14px;
      }
      .flysmart-meta {
        color: rgba(15,23,42,0.72);
        font-size: 13px;
        line-height: 1.35;
      }
      .flysmart-smalllink a {
        font-size: 13px;
        text-decoration: none;
      }
      .flysmart-smalllink a:hover { text-decoration: underline; }
      hr { margin: 0.75rem 0; opacity: 0.25; }
      div.stButton > button { border-radius: 10px; }
      .flysmart-resolve-note { font-size: 12px; color: rgba(15,23,42,0.72); }
      .flysmart-policy-pill {
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        font-size:12px;
        font-weight:600;
        border:1px solid rgba(15,23,42,0.08);
        color:#0f172a;
        background:#f1f5f9;
      }
      .flysmart-title-wrap {
        display:flex;
        align-items:center;
        gap:12px;
        margin-bottom: 6px;
      }


    .flysmart-policy-row {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      margin-bottom: 8px;
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.04);
      border-left: 4px solid rgba(15, 23, 42, 0.25);
      font-size: 14px;
    }

    .flysmart-policy-label {
      font-weight: 600;
      min-width: 190px;
      color: #0f172a;
    }

    .flysmart-policy-time {
      color: rgba(15,23,42,0.75);
      font-size: 13px;
    }
    

    [data-testid="stDivider"] {
      display: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True    
)
# LOAD STATIC DATA
flights_df = pd.read_csv("flights.csv")
# HELPERS

IATA_IN_BRACKETS = re.compile(r"\((\w{3})\)")
def fmt_dt_uk(dt: Optional[datetime]) -> str:
    """UK format: dd/mm/yyyy HH:MM"""
    if not dt:
        return "—"
    return dt.strftime("%d/%m/%Y %H:%M")

def extract_iata(text: str) -> str:
    m = IATA_IN_BRACKETS.search(str(text))
    return m.group(1).upper().strip() if m else ""

def normalise_key(s: str) -> str:
    # normalise airline name keys so lookups don’t fail due to spacing/case
    return re.sub(r"\s+", " ", str(s or "").strip()).casefold()

def normalise_city(destination: str) -> str:
    s = str(destination)
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("–", " ").replace("-", " ")
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s:
        return ""
    tokens = s.split()
    if tokens and len(tokens[-1]) == 3 and tokens[-1].isupper():
        tokens = tokens[:-1]

    airport_tokens = {
        "intl", "international", "airport", "airpt", "terminal",
        "heathrow", "gatwick", "stansted", "luton", "city",
        "jfk", "laguardia", "newark",
        "o'hare", "ohare", "midway", "dulles",
        "schiphol", "charles", "de", "gaulle", "orly",
        "barajas", "el", "prat", "fiumicino", "ciampino",
        "narita", "haneda", "incheon", "changi",
        "king", "khaled", "abdulaziz", "maktoum"
    }

    city_tokens = []
    for t in tokens:
        if t.lower() in airport_tokens:
            break
        city_tokens.append(t)

    city = " ".join(city_tokens).strip()
    if not city and tokens:
        city = " ".join(tokens[:2]).strip()
    return city

def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def parse_datetime_best_effort(dt_str: str):
    if not dt_str:
        return None
    s = str(dt_str).strip()
    s = s.replace("T", " ").replace("Z", "")
    s = s[:16]
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M")
    except Exception:
        return None
        
def compute_local_time_from_offset(offset_seconds: int) -> datetime:
    """
    Convert OpenWeather timezone offset (seconds from UTC)
    into a local datetime.
    """
    return datetime.utcnow() + timedelta(seconds=int(offset_seconds))

def badge_html(label: str, tone: str = "neutral") -> str:
    tones = {
        "neutral": ("#0f172a", "#e2e8f0"),
        "green": ("#064e3b", "#d1fae5"),
        "amber": ("#7c2d12", "#ffedd5"),
        "red": ("#7f1d1d", "#fee2e2"),
        "blue": ("#1e3a8a", "#dbeafe"),
    }
    fg, bg = tones.get(tone, tones["neutral"])
    return f"""
    <span style="
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        font-size:12px;
        font-weight:600;
        color:{fg};
        background:{bg};
        border:1px solid rgba(15,23,42,0.08);
        vertical-align:middle;
    ">{label}</span>
    """

def booking_search_url(city: str, place_name: str = "") -> str:
    base = "https://www.booking.com/searchresults.html"
    query = f"{place_name} {city}".strip()
    params = {"ss": query}

    affiliate_id = None
    try:
        affiliate_id = st.secrets.get("booking", {}).get("affiliate_id", None)
    except Exception:
        affiliate_id = None

    if affiliate_id:
        params["aid"] = str(affiliate_id)

    return f"{base}?{urllib.parse.urlencode(params)}"

def render_fixed_hotel_image(img_url: str):
    st.markdown(
        f'<img class="flysmart-hotel-img" src="{img_url}" alt="Hotel photo" />',
        unsafe_allow_html=True
    )
@st.cache_data(ttl=600)
def openweather_current(api_key: str, *, city: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"

    if lat is not None and lon is not None:
        params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    else:
        params = {"q": city, "appid": api_key, "units": "metric"}

    r = requests.get(url, params=params, timeout=10)

    try:
        data = r.json()
    except Exception:
        data = {"cod": r.status_code, "message": r.text[:200]}
        
    cod = data.get("cod", r.status_code)
    try:
        cod_int = int(cod)
    except Exception:
        cod_int = r.status_code

    # For non-200, return payload
    if cod_int != 200:
        data["_http_status"] = r.status_code
        return data

    return data
# AIRPORT RESOLUTION (IATA -> city + coordinates)

@st.cache_data
def load_airports_lookup(path: str = "airports.csv") -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(path, dtype=str).fillna("")
    if "iata_code" not in df.columns:
        return {}
    df = df[df["iata_code"].str.len() == 3].copy()
    df["iata_code"] = df["iata_code"].str.upper().str.strip()

    lookup: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        iata = (r.get("iata_code") or "").upper().strip()
        if not iata:
            continue
        lat_s = r.get("latitude_deg", "")
        lon_s = r.get("longitude_deg", "")
        try:
            lat = float(lat_s) if lat_s else None
        except Exception:
            lat = None
        try:
            lon = float(lon_s) if lon_s else None
        except Exception:
            lon = None

        lookup[iata] = {
            "city": (r.get("municipality") or "").strip(),
            "country": (r.get("iso_country") or "").strip(),
            "lat": lat,
            "lon": lon,
            "name": (r.get("name") or "").strip(),
        }
    return lookup

try:
    airports_lookup = load_airports_lookup("airports.csv")
except FileNotFoundError:
    airports_lookup = {}

def resolve_place_context(location_text: str, fallback_iata: str = "") -> Dict[str, Any]:
    iata = extract_iata(location_text) or (fallback_iata or "").upper().strip()

    if iata and iata in airports_lookup:
        a = airports_lookup[iata]
        city = a.get("city") or a.get("name") or ""
        return {
            "iata": iata,
            "city": city,
            "lat": a.get("lat"),
            "lon": a.get("lon"),
            "resolved_from": "iata_db",
        }

    city = normalise_city(location_text)
    return {"iata": iata, "city": city, "lat": None, "lon": None, "resolved_from": "text_fallback"}

# AIRLINE POLICIES (ALWAYS AVAILABLE VIA FALLBACK)
@st.cache_data
def load_airline_policies(path: str = "airline_info.json") -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {"policies": {}, "fallback_policy": {}}

    if "policies" not in data or not isinstance(data["policies"], dict):
        data["policies"] = {}

    if "fallback_policy" not in data or not isinstance(data["fallback_policy"], dict):
        data["fallback_policy"] = {
            "check_in_open_hours_before": 24,
            "check_in_close_minutes_before": 60,
            "baggage_drop_close_minutes_before": 60,
            "boarding_gate_close_minutes_before": 20,
            "source_url": "",
            "last_verified": "",
            "notes": "Generic prototype defaults (unverified)."
        }

    norm_index = {}
    for k, v in data["policies"].items():
        norm_index[normalise_key(k)] = v

    data["_normalised_index"] = norm_index
    return data

try:
    airline_policy_data = load_airline_policies("airline_info.json")
except FileNotFoundError:
    airline_policy_data = {
        "policies": {},
        "_normalised_index": {},
        "fallback_policy": {
            "check_in_open_hours_before": 24,
            "check_in_close_minutes_before": 60,
            "baggage_drop_close_minutes_before": 60,
            "boarding_gate_close_minutes_before": 20,
            "source_url": "",
            "last_verified": "",
            "notes": "Generic prototype defaults (unverified)."
        }
    }

def get_policy_with_fallback(airline: str) -> Tuple[Dict[str, Any], bool]:
    norm = normalise_key(airline)
    idx = airline_policy_data.get("_normalised_index", {})
    if norm in idx:
        return idx[norm], False
    return airline_policy_data.get("fallback_policy", {}), True

def fmt_deadline(dep_dt: Optional[datetime], minutes_before: Optional[int]) -> str:
    if not dep_dt or minutes_before is None:
        return "—"
    try:
        mins = int(minutes_before)
    except Exception:
        return "—"
    t = dep_dt - timedelta(minutes=mins)
    return t.strftime("%H:%M")  


def fmt_window_hours(dep_dt: Optional[datetime], hours_before: Optional[int]) -> str:
    if not dep_dt or hours_before is None:
        return "—"
    try:
        hrs = int(hours_before)
    except Exception:
        return "—"
    t = dep_dt - timedelta(hours=hrs)
    return t.strftime("%d/%m/%Y %H:%M")  

# AIRPORT GUIDANCE
@st.cache_data
def load_airport_guidance(path: str = "airport_guidance.json") -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        out[str(k).upper().strip()] = v
    return out

try:
    airport_guidance = load_airport_guidance("airport_guidance.json")
except FileNotFoundError:
    airport_guidance = {}

def get_airport_guidance(iata: str) -> Dict[str, Any]:
    if not iata:
        return airport_guidance.get("DEFAULT", {})
    return airport_guidance.get(iata.upper().strip(), airport_guidance.get("DEFAULT", {}))
    
# AVIATIONSTACK
@st.cache_data(ttl=60)
def aviationstack_lookup(flight_iata: str) -> dict:
    key = st.secrets["aviationstack"]["api_key"]
    base_url = "https://api.aviationstack.com/v1/flights"
    params = {"access_key": key, "flight_iata": flight_iata.strip().upper(), "limit": 10}

    r = requests.get(base_url, params=params, timeout=12)

    if r.status_code == 403:
        try:
            msg = r.json()
        except Exception:
            msg = r.text[:200]
        raise requests.HTTPError(f"403 Forbidden from Aviationstack. Response: {msg}")

    r.raise_for_status()
    return r.json()

def pick_best_aviationstack_item(items: list):
    if not items:
        return None

    def status_score(x):
        s = (x.get("flight_status") or "").lower()
        if "active" in s:
            return 3
        if "scheduled" in s:
            return 2
        if "landed" in s:
            return 1
        return 0

    items_sorted = sorted(items, key=status_score, reverse=True)
    return items_sorted[0]
    
# GOOGLE PLACES + PHOTOS
@st.cache_data(ttl=3600)
def google_places_search_hotels(city: str, limit: int = 6) -> list:
    api_key = st.secrets["google"]["places_api_key"]

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.photos,places.googleMapsUri",
    }
    body = {"textQuery": f"hotels in {city}"}

    r = requests.post(url, headers=headers, json=body, timeout=15)
    r.raise_for_status()
    data = r.json()
    places = data.get("places") or []
    return places[:limit]

@st.cache_data(ttl=86400)
def google_photo_uri(photo_name: str, max_width: int = 900) -> str:
    api_key = st.secrets["google"]["places_api_key"]
    url = f"https://places.googleapis.com/v1/{photo_name}/media"
    params = {"key": api_key, "maxWidthPx": max_width, "skipHttpRedirect": "true"}

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data.get("photoUri", "")

def extract_photo_and_attrib(place: dict) -> Tuple[Optional[str], Optional[str]]:
    photos = place.get("photos") or []
    if not photos:
        return None, None

    p0 = photos[0]
    photo_name = p0.get("name")

    attribs = p0.get("authorAttributions") or []
    parts = []
    for a in attribs:
        name = a.get("displayName")
        uri = a.get("uri")
        if name and uri:
            parts.append(f"{name} ({uri})")
        elif name:
            parts.append(name)
    attrib_text = " | ".join(parts) if parts else None

    return photo_name, attrib_text

# BUILD STATIC DEMO LOOKUP
sample_flights = {
    row["flight_number"]: {
        "airline": row["airline"],
        "origin": row["origin"],
        "destination": row["destination"],
        "departure": row["departure"],
        "status": row["status"]
    }
    for _, row in flights_df.iterrows()
}

# HEADER
c_logo, c_title = st.columns([1, 8], vertical_alignment="center")
with c_logo:
    try:
        st.image("logo.png", width=54)
    except Exception:
        pass
with c_title:
    st.markdown("## FlySmart")
    st.caption("Track your flight. Know what matters. Travel stress-free.")

st.divider()

# SEARCH MODE
st.subheader("Find Your Flight")

mode = st.radio(
    "Choose tracking mode:",
    ["Live search (Aviationstack)", "Demo flights"],
    horizontal=True
)

selected_flight_number = None
live_payload = None
used_live = False

if mode == "Live search (Aviationstack)":
    with st.form("live_search_form", clear_on_submit=False):
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        with c1:
            live_query = st.text_input(
                "Flight number",
                placeholder="e.g., EK67, BAW277, SV22",
                label_visibility="collapsed"
            ).strip().upper()
        with c2:
            go = st.form_submit_button("Track", use_container_width=True)

    st.markdown(badge_html("Live status source: Aviationstack", "blue"), unsafe_allow_html=True)

    if go and live_query:
        try:
            data = aviationstack_lookup(live_query)
            items = data.get("data") or []
            best = pick_best_aviationstack_item(items)
            if not best:
                st.warning("No live results found for that flight number. Try another flight number.")
            else:
                live_payload = best
                selected_flight_number = safe_get(live_payload, "flight", "iata", default=live_query)
                used_live = True
        except KeyError:
            st.error("Aviationstack API key not found. Add it in Streamlit Secrets as [aviationstack] api_key.")
        except requests.HTTPError as e:
            st.warning(f"Live lookup failed (HTTP). Details: {e}")
        except Exception as e:
            st.warning(f"Live lookup failed. Details: {e}")

    if not selected_flight_number:
        st.info("Enter a flight number and click Track to view details.")
        st.stop()

else:

    policy_airlines_norm = set(airline_policy_data.get("_normalised_index", {}).keys())

    candidates = flights_df.copy()
    candidates["airline_norm"] = candidates["airline"].apply(normalise_key)
    curated = candidates[candidates["airline_norm"].isin(policy_airlines_norm)]
    
    curated_list = []
    seen_airlines = set()
    for _, r in curated.sort_values("flight_number").iterrows():
        a = r["airline_norm"]
        if a in seen_airlines:
            continue
        curated_list.append(r["flight_number"])
        seen_airlines.add(a)
        if len(curated_list) >= 5:
            break

    # Fallback: if none match (e.g., policies file missing), then show first 3-5 from dataset
    if not curated_list:
        curated_list = sorted(flights_df["flight_number"].tolist())[:5]

    demo_pick = st.selectbox(
        "Select a demo flight:",
        options=[""] + curated_list,
        index=0,
        placeholder="Start typing…"
    )

    if demo_pick:
        selected_flight_number = demo_pick
        live_payload = None
        used_live = False
    else:
        st.info("Select a demo flight to view details.")
        st.stop()

# UNIFY DETAILS (LIVE OR DEMO)
airline_name = ""
origin = ""
destination = ""
departure_str = ""
status = ""
arrival_str = ""
origin_iata = ""
dest_iata = ""

if live_payload:
    airline_name = safe_get(live_payload, "airline", "name", default="Unknown Airline")
    status_raw = safe_get(live_payload, "flight_status", default="Unknown")
    status = status_raw.title() if isinstance(status_raw, str) else str(status_raw)

    dep_airport = safe_get(live_payload, "departure", "airport", default="")
    origin_iata = (safe_get(live_payload, "departure", "iata", default="") or "").upper().strip()

    arr_airport = safe_get(live_payload, "arrival", "airport", default="")
    dest_iata = (safe_get(live_payload, "arrival", "iata", default="") or "").upper().strip()

    origin = f"{dep_airport} ({origin_iata})" if dep_airport and origin_iata else dep_airport or origin_iata or "Unknown"
    destination = f"{arr_airport} ({dest_iata})" if arr_airport and dest_iata else arr_airport or dest_iata or "Unknown"

    dep_est = safe_get(live_payload, "departure", "estimated", default=None)
    dep_sch = safe_get(live_payload, "departure", "scheduled", default=None)
    arr_est = safe_get(live_payload, "arrival", "estimated", default=None)
    arr_sch = safe_get(live_payload, "arrival", "scheduled", default=None)

    dep_dt = parse_datetime_best_effort(dep_est) or parse_datetime_best_effort(dep_sch)
    arr_dt = parse_datetime_best_effort(arr_est) or parse_datetime_best_effort(arr_sch)

    departure_str = dep_dt.strftime("%Y-%m-%d %H:%M") if dep_dt else "Unknown"
    arrival_str = arr_dt.strftime("%Y-%m-%d %H:%M") if arr_dt else ""
else:
    details = sample_flights[selected_flight_number]
    airline_name = details["airline"]
    origin = details["origin"]
    destination = details["destination"]
    departure_str = details["departure"]
    status = details["status"]
    arrival_str = ""
    origin_iata = extract_iata(origin)
    dest_iata = extract_iata(destination)

dep_time = parse_datetime_best_effort(departure_str)

# PAGE TABS
tab1, tab2, tab3 = st.tabs(["Flight Summary", "Airline Info & Weather", "Stay & Explore"])

# TAB 1
with tab1:
    st.subheader("Flight Summary")
    s_low = str(status).lower()
    if "delay" in s_low:
        tone = "red"
    elif "cancel" in s_low:
        tone = "red"
    elif "on time" in s_low or "scheduled" in s_low:
        tone = "green"
    elif "active" in s_low:
        tone = "blue"
    else:
        tone = "neutral"

    dep_time_display = fmt_dt_uk(dep_time)

    arr_time = parse_datetime_best_effort(arrival_str) if arrival_str else None
    arr_time_display = fmt_dt_uk(arr_time) if arr_time else ""

    st.markdown(
        f"""
**Flight:** {selected_flight_number} — {airline_name}  
**Route:** {origin} → {destination}  
**Departure:** {dep_time_display}  
{f"**Arrival:** {arr_time_display}  " if arr_time_display else ""}
**Status:** {badge_html(status, tone)}
        """,
        unsafe_allow_html=True
    )

    st.caption("Live data comes from Aviationstack when available; demo flights use a local dataset.")
        
    #Airport Guidance (Origin airport)
    st.divider()
    st.subheader("Airport Guidance")

    guidance_entry = get_airport_guidance(origin_iata)
    g_name = guidance_entry.get("airport_name", "Airport")
    g_lines = guidance_entry.get("guidance", [])

    if origin_iata:
        st.markdown(f"**{g_name} ({origin_iata})**")
        if g_lines:
            for line in g_lines:
                st.write(f"• {line}")
        else:
            st.info("No guidance available for this airport yet.")
    else:
        st.info("Origin airport code unavailable for guidance.")

# TAB 2
with tab2:
    st.subheader("Airline Policy Information")

    pol, is_fallback = get_policy_with_fallback(airline_name)

    if is_fallback:
        st.info("No airline-specific policy found. Showing generic prototype defaults (unverified).")

    check_open_h = pol.get("check_in_open_hours_before")
    check_close_m = pol.get("check_in_close_minutes_before")
    bag_close_m = pol.get("baggage_drop_close_minutes_before")
    gate_close_m = pol.get("boarding_gate_close_minutes_before")

    st.markdown(
    f"""
<div class="flysmart-policy-row">
  <div class="flysmart-policy-label">Check-in opens</div>
  <div class="flysmart-policy-time">{check_open_h}h before • {fmt_window_hours(dep_time, check_open_h)}</div>
</div>

<div class="flysmart-policy-row">
  <div class="flysmart-policy-label">Check-in closes</div>
  <div class="flysmart-policy-time">{check_close_m} min before • {fmt_deadline(dep_time, check_close_m)}</div>
</div>

<div class="flysmart-policy-row">
  <div class="flysmart-policy-label">Baggage drop closes</div>
  <div class="flysmart-policy-time">{bag_close_m} min before • {fmt_deadline(dep_time, bag_close_m)}</div>
</div>

<div class="flysmart-policy-row">
  <div class="flysmart-policy-label">Boarding gate closes</div>
  <div class="flysmart-policy-time">{gate_close_m} min before • {fmt_deadline(dep_time, gate_close_m)}</div>
</div>
""",
    unsafe_allow_html=True
)

    src = pol.get("source_url", "")
    last_v = pol.get("last_verified", "")
    notes = pol.get("notes", "")

    if src:
        st.markdown(
        f"<small><a href='{src}' target='_blank'>Airline policy source</a></small>",
        unsafe_allow_html=True
)

    st.divider()
    st.subheader("Live Weather at Destination")

    ctx = resolve_place_context(destination, fallback_iata=dest_iata)
    city = (ctx.get("city") or "").strip()
    lat = ctx.get("lat")
    lon = ctx.get("lon")

    if city:
        if ctx.get("resolved_from") == "iata_db":
            st.markdown(
                f"<div class='flysmart-resolve-note'>Resolved destination using IATA <b>{ctx.get('iata','')}</b> → <b>{city}</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='flysmart-resolve-note'>Resolved destination from text → <b>{city}</b></div>",
                unsafe_allow_html=True
            )

    if not city:
        st.warning("No valid destination city could be resolved for weather.")
    else:
        try:
            api_key = st.secrets["weather"]["api_key"]
            url = "https://api.openweathermap.org/data/2.5/weather"

            if lat is not None and lon is not None:
                params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
            else:
                params = {"q": city, "appid": api_key, "units": "metric"}

            r = requests.get(url, params=params, timeout=10)
            data = r.json()

            if data.get("cod") == 200:
                temp = data["main"]["temp"]
                desc = data["weather"][0]["description"].title()
                icon = data["weather"][0]["icon"]

                cols = st.columns([1, 4])
                with cols[0]:
                    st.image(f"http://openweathermap.org/img/wn/{icon}.png", width=64)
                with cols[1]:
                    st.success(f"Weather in {city}: **{temp} °C**, {desc}")
            else:
                st.warning(f"Weather not available for '{city}'.")
        except KeyError:
            st.error("OpenWeather API key not found. Add it to Streamlit Secrets as [weather] api_key.")
        except Exception:
            st.warning("Unable to fetch live weather data right now.")
    # Destination Readiness
    st.divider()
    st.subheader("Destination Readiness")

    dest_ctx = resolve_place_context(destination, fallback_iata=dest_iata)
    origin_ctx = resolve_place_context(origin, fallback_iata=origin_iata)

    dest_city = (dest_ctx.get("city") or "").strip()
    dest_lat = dest_ctx.get("lat")
    dest_lon = dest_ctx.get("lon")

    origin_city = (origin_ctx.get("city") or "").strip()
    origin_lat = origin_ctx.get("lat")
    origin_lon = origin_ctx.get("lon")

    if not dest_city:
        st.info("Destination readiness not available (could not resolve destination city).")
    else:
        try:
            api_key = st.secrets["weather"]["api_key"]

            # Destination weather 
            dest_w = openweather_current(api_key, city=dest_city, lat=dest_lat, lon=dest_lon)
                    # If OpenWeather didn’t return a valid weather payload
            if int(dest_w.get("cod", 0)) != 200:
                msg = dest_w.get("message", "Unknown error")
                st.info(f"Destination readiness unavailable (OpenWeather: {msg}).")
                st.stop()

            dest_tz = dest_w.get("timezone", None)  # seconds from UTC
            dest_temp = safe_get(dest_w, "main", "temp", default=None)

            origin_tz = None
            origin_temp = None
            if origin_city:
                try:
                    origin_w = openweather_current(api_key, city=origin_city, lat=origin_lat, lon=origin_lon)
                    origin_tz = origin_w.get("timezone", None)
                    origin_temp = safe_get(origin_w, "main", "temp", default=None)
                except Exception:
                    origin_tz = None
                    origin_temp = None

            readiness_lines = []

            if isinstance(dest_tz, int):
                dest_local = compute_local_time_from_offset(dest_tz)
                readiness_lines.append(f"Local time at destination: {dest_local.strftime('%H:%M')}")

            if isinstance(dest_tz, int) and isinstance(origin_tz, int):
                diff_hours = int((dest_tz - origin_tz) / 3600)
                if diff_hours == 0:
                    readiness_lines.append("Time difference vs departure: No difference")
                elif diff_hours > 0:
                    readiness_lines.append(f"Time difference vs departure: +{diff_hours}h ahead")
                else:
                    readiness_lines.append(f"Time difference vs departure: {diff_hours}h behind")

            if dest_temp is not None and origin_temp is not None:
                try:
                    delta = float(dest_temp) - float(origin_temp)
                    if abs(delta) < 2:
                        readiness_lines.append("Temperature change: Similar to departure")
                    elif delta > 0:
                        readiness_lines.append("Temperature change: Warmer than departure")
                    else:
                        readiness_lines.append("Temperature change: Cooler than departure")
                except Exception:
                    pass

            if readiness_lines:
                for x in readiness_lines:
                    st.write(f"• {x}")
            else:
                st.info("Readiness cues unavailable (insufficient data returned).")

        except KeyError:
            st.error("OpenWeather API key not found. Add it to Streamlit Secrets as [weather] api_key.")
        except Exception as e:
            st.warning("Unable to compute destination readiness right now.")
            st.caption(f"Debug (readiness): {type(e).__name__}: {e}")

# TAB 3
with tab3:
    st.subheader("Stay & Explore")

    ctx = resolve_place_context(destination, fallback_iata=dest_iata)
    city = (ctx.get("city") or "").strip()

    if not city:
        st.info("No destination city could be resolved for recommendations.")
        st.stop()

    st.caption(f"Recommended stays in **{city}** (Google Places).")

    try:
        places = google_places_search_hotels(city, limit=6)

        if not places:
            st.warning("No hotel results returned for this destination.")
        else:
            for p in places:
                name = safe_get(p, "displayName", "text", default="Unnamed Place")
                address = p.get("formattedAddress", "")
                maps_url = p.get("googleMapsUri", "")

                photo_name, attrib = extract_photo_and_attrib(p)
                img_url = google_photo_uri(photo_name, max_width=900) if photo_name else ""

                deal_url = booking_search_url(city, name)

                st.markdown('<div class="flysmart-hotel-card">', unsafe_allow_html=True)

                c_img, c_text, c_cta = st.columns([1.3, 2.6, 1.2], vertical_alignment="center")

                with c_img:
                    if img_url:
                        render_fixed_hotel_image(img_url)
                    else:
                        st.caption("No photo available")

                with c_text:
                    st.markdown(f"**{name}**")
                    if address:
                        st.markdown(f'<div class="flysmart-meta">{address}</div>', unsafe_allow_html=True)

                    if attrib:
                        with st.expander("Photo credits", expanded=False):
                            st.caption(attrib)

                with c_cta:
                    st.link_button("View deal", deal_url, use_container_width=True)
                    if maps_url:
                        st.markdown(
                            f'<div class="flysmart-smalllink" style="text-align:center; margin-top:6px;"><a href="{maps_url}" target="_blank">Map</a></div>',
                            unsafe_allow_html=True
                        )

                st.markdown("</div>", unsafe_allow_html=True)

    except KeyError:
        st.error("Google Places API key not found. Add it in Streamlit Secrets as [google] places_api_key.")
    except Exception as e:
        st.warning(f"Unable to load recommendations right now. Details: {e}")
# FOOTER
st.caption("Developed as part of a University Project • FlySmart Prototype")

