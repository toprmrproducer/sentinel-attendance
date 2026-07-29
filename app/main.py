import os
import time
import cv2
from fastapi import FastAPI, UploadFile, File, Form, Request, Depends
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from app import engine, attendance, tracking, live, review, payroll, telephony

app = FastAPI(title="Sentinel Attendance POC")
app.add_middleware(SessionMiddleware, secret_key="sentinel-poc-dev-secret-change-in-real-deploy")

ADMIN_USER = "admin"
ADMIN_PASS = "sentinel2026"

FOOTAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_footage")
SAMPLE_VIDEO = os.path.join(FOOTAGE_DIR, "vtest.avi")
LONG_FEED_VIDEO = os.path.join(FOOTAGE_DIR, "long_feed", "timessquare_45min.mp4")
GODSEYE_SOURCE = "godseye_main"
CAFE_VIDEO = os.path.join(FOOTAGE_DIR, "cafe", "spoonfull_baristacam.mp4")
CAFE_SOURCE = "cafe_spoonfull"


def require_login(request: Request):
    return request.session.get("authed", False)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = ""):
    err = f'<p style="color:#f66">{error}</p>' if error else ""
    return f"""
<!doctype html><html><head><title>Sentinel &mdash; Login</title>
<style>
body{{font-family:-apple-system,sans-serif;background:#0a0a0a;color:#eee;display:flex;
height:100vh;align-items:center;justify-content:center;margin:0}}
.box{{background:#141414;padding:32px;border-radius:12px;border:1px solid #262626;width:280px}}
h1{{font-size:18px;margin:0 0 16px}}
input{{width:100%;padding:10px;margin:6px 0;background:#1e1e1e;border:1px solid #2c2c2c;
border-radius:6px;color:#eee;box-sizing:border-box}}
button{{width:100%;padding:10px;margin-top:10px;background:#2f6feb;border:none;border-radius:6px;
color:#fff;font-weight:600;cursor:pointer}}
</style></head><body>
<div class="box"><h1>Sentinel &mdash; God's Eye</h1>{err}
<form method="post" action="/login">
<input name="username" placeholder="Username" autofocus>
<input name="password" type="password" placeholder="Password">
<button type="submit">Sign in</button>
</form></div></body></html>
"""


@app.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == ADMIN_USER and password == ADMIN_PASS:
        request.session["authed"] = True
        return RedirectResponse("/godseye", status_code=303)
    return RedirectResponse("/login?error=Invalid+credentials", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.on_event("startup")
def _boot_godseye_feed():
    live.start_source(GODSEYE_SOURCE, LONG_FEED_VIDEO, loop=True, sample_every=3)
    if os.path.exists(CAFE_VIDEO):
        live.start_source(CAFE_SOURCE, CAFE_VIDEO, loop=True, sample_every=3)


@app.post("/api/enroll")
async def api_enroll(name: str = Form(...), file: UploadFile = File(...)):
    tmp_path = f"/tmp/enroll_{int(time.time())}_{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    result = engine.enroll_person(name, tmp_path)
    os.remove(tmp_path)
    return JSONResponse(result)


@app.get("/api/working-hours")
def api_working_hours():
    return attendance.working_hours_today()


@app.get("/api/recent")
def api_recent():
    return attendance.recent_sightings()


def _annotated_frame_generator(video_path: str, log_to_attendance: bool = True):
    cap = cv2.VideoCapture(video_path)
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the sample footage
            continue
        frame_i += 1
        if frame_i % 5 == 0:  # run recognition every 5th frame, keep it watchable
            results = engine.recognize_in_frame(frame)
            for r in results:
                x1, y1, x2, y2 = r["bbox"]
                label = f'{r["match"]} ({r["match_score"]:.2f})' if r["match"] != "unknown" else "unknown"
                color = (0, 200, 0) if r["match"] != "unknown" else (0, 0, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if log_to_attendance:
                    attendance.log_sighting(r["match"], r["match_score"], source=os.path.basename(video_path))
        ok2, jpeg = cv2.imencode(".jpg", frame)
        if not ok2:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.05)


@app.get("/stream/sample")
def stream_sample():
    return StreamingResponse(
        _annotated_frame_generator(SAMPLE_VIDEO),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _mjpeg_from_live(source: str):
    while True:
        frame = live.get_frame(source)
        if frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.08)


@app.get("/stream/godseye/{source}")
def stream_godseye(source: str):
    return StreamingResponse(
        _mjpeg_from_live(source),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/godseye/detections/{source}")
def api_godseye_detections(source: str):
    return live.get_detections(source)


@app.get("/api/godseye/tracks/{source}")
def api_godseye_tracks(source: str, class_name: str = None):
    return tracking.list_tracks(source, class_name)


@app.get("/api/godseye/track/{source}/{track_id}")
def api_godseye_track(source: str, track_id: int):
    profile = tracking.track_profile(track_id, source)
    return profile or JSONResponse({"error": "not found"}, status_code=404)


@app.get("/api/godseye/zones/{source}")
def api_godseye_zones_get(source: str):
    return tracking.list_zones(source)


@app.post("/api/godseye/zones/{source}")
async def api_godseye_zones_post(source: str, request: Request):
    body = await request.json()
    zid = tracking.create_zone(source, body["label"], body["x1"], body["y1"], body["x2"], body["y2"])
    return {"ok": True, "id": zid}


@app.get("/api/godseye/zone-stats/{source}")
def api_godseye_zone_stats(source: str):
    return tracking.zone_stats(source)


@app.get("/api/godseye/throughput/{source}")
def api_godseye_throughput(source: str, class_name: str = "cup"):
    return tracking.throughput_per_minute(source, class_name)


@app.get("/api/godseye/anomalies/{source}")
def api_godseye_anomalies(source: str):
    return tracking.recent_anomalies(source, limit=50)


@app.get("/api/identities")
def api_identities():
    return attendance.all_identities()


@app.post("/api/review/{name}")
def api_review(name: str):
    sightings = attendance.sightings_for(name)
    summary = review.build_person_summary(name, sightings)
    if not summary:
        return {"ok": False, "error": "no sightings logged for this person yet"}
    result = review.request_review(summary)
    result["summary"] = summary
    return result


@app.get("/api/payroll/summary/{name}")
def api_payroll_summary(name: str):
    sightings = attendance.sightings_for(name)
    summary = review.build_person_summary(name, sightings)
    if not summary:
        return {"ok": False, "error": "no sightings logged yet", "hourly_wage": payroll.get_wage(name)}
    return {"ok": True, **payroll.wage_summary(name, summary["per_day"]), "per_day": summary["per_day"]}


@app.post("/api/payroll/wage/{name}")
async def api_set_wage(name: str, request: Request):
    body = await request.json()
    return {"ok": True, "name": name, "hourly_wage": payroll.set_wage(name, float(body["hourly_wage"]))}


@app.get("/api/payroll/cash/{source}")
def api_payroll_cash(source: str):
    cups = tracking.list_tracks(source, "cup")
    return payroll.cash_collected(len(cups))


@app.post("/api/payroll/price-per-cup")
async def api_set_price(request: Request):
    body = await request.json()
    return {"ok": True, "price_per_cup": payroll.set_price_per_cup(float(body["price"]))}


@app.get("/telephony/theft-alert-xml")
def telephony_alert_xml(msg: str = "anomaly detected"):
    from fastapi import Response
    return Response(content=telephony.theft_alert_xml(msg), media_type="application/xml")


@app.post("/api/telephony/test-alert")
def api_test_alert(request: Request):
    number = request.query_params.get("to") or os.environ.get("SENTINEL_ALERT_CALL_NUMBER")
    if not number:
        return {"ok": False, "error": "no number provided (pass ?to=919...) and SENTINEL_ALERT_CALL_NUMBER is unset"}
    return telephony.trigger_alert_call(number, "This is a test of the Sentinel theft-aversion auto-dial.")


@app.get("/godseye", response_class=HTMLResponse)
def godseye(request: Request, source: str = GODSEYE_SOURCE):
    if not require_login(request):
        return RedirectResponse("/login", status_code=303)
    SRC = source
    return f"""
<!doctype html>
<html><head><title>Sentinel &mdash; God's Eye</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#07080a;color:#e8e8e8;margin:0}}
header{{display:flex;justify-content:space-between;align-items:center;padding:14px 22px;
border-bottom:1px solid #1c1e22;background:#0b0d10}}
header h1{{font-size:16px;margin:0;letter-spacing:.3px}}
header a{{color:#7aa2f7;text-decoration:none;font-size:12px}}
.layout{{display:grid;grid-template-columns:1fr 320px;gap:0;height:calc(100vh - 51px)}}
.stage-wrap{{position:relative;padding:16px;overflow:auto}}
.stage{{position:relative;display:inline-block;border:1px solid #1c1e22;border-radius:8px;overflow:hidden}}
.stage img{{display:block;max-width:100%}}
.box{{position:absolute;border:2px solid #3fa7ff;border-radius:3px;cursor:pointer;
transition:box-shadow .1s;pointer-events:auto}}
.box:hover{{box-shadow:0 0 0 2px #ffffff55;background:#3fa7ff22}}
.box.person{{border-color:#3fd17a}}
.box .tag{{position:absolute;top:-20px;left:-2px;background:#111;padding:1px 6px;font-size:11px;
border-radius:4px;white-space:nowrap;border:1px solid #2a2a2a}}
.zone{{position:absolute;border:2px dashed #ffb020;background:#ffb02015;pointer-events:none}}
.zone .tag{{position:absolute;top:-18px;left:0;background:#3a2900;color:#ffb020;padding:1px 6px;
font-size:10px;border-radius:4px}}
.side{{border-left:1px solid #1c1e22;background:#0b0d10;overflow-y:auto;padding:16px}}
.side h2{{font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#8a8f98;margin:18px 0 8px}}
.side h2:first-child{{margin-top:0}}
.panel{{background:#111318;border:1px solid #1c1e22;border-radius:8px;padding:12px;font-size:12.5px;
margin-bottom:10px}}
.panel .k{{color:#8a8f98}}
.alert{{border-left:3px solid #ff5555;padding:6px 8px;margin-bottom:6px;font-size:11.5px;background:#160e0e;border-radius:4px}}
.alert.lighting_drop{{border-color:#ffcc00}}
.btn{{background:#2f6feb;border:none;color:#fff;padding:7px 12px;border-radius:6px;font-size:12px;
cursor:pointer;margin-top:8px}}
.btn.secondary{{background:#22252b}}
select{{width:100%;background:#1e1e1e;color:#eee;border:1px solid #2c2c2c;border-radius:6px;padding:6px;margin-bottom:6px}}
.hint{{color:#666;font-size:11px;margin-top:4px}}
.review-note{{white-space:pre-wrap;font-size:12px;line-height:1.5;margin-top:8px;color:#cfd3da}}
</style></head><body>
<header>
<h1>&#9673; SENTINEL &mdash; God's Eye <span style="color:#555;font-weight:400">/ {SRC}</span></h1>
<a href="/godseye?source={GODSEYE_SOURCE}">Times Square feed</a>&nbsp;&nbsp;
<a href="/godseye?source={CAFE_SOURCE}">Cafe feed</a>&nbsp;&nbsp;
<a href="/dashboard">Attendance dashboard</a>&nbsp;&nbsp;<a href="/logout">Log out</a>
</header>
<div class="layout">
  <div class="stage-wrap">
    <div class="stage" id="stage">
      <img id="feed" src="/stream/godseye/{SRC}"/>
      <div id="overlays"></div>
    </div>
    <p class="hint">Hover a box to see what it is. Click a person/object box to open its profile.
    Hold <b>Shift</b> and drag to draw a zone (e.g. "Entrance", "Zebra Crossing").</p>
  </div>
  <div class="side">
    <h2>Selected</h2>
    <div class="panel" id="selected-panel">Click any box on the feed.</div>

    <h2>Ask the model &mdash; review packet</h2>
    <select id="identity-select"><option value="">Pick an enrolled identity&hellip;</option></select>
    <button class="btn" onclick="runReview()">Generate review (Opus 4.8)</button>
    <div id="review-out"></div>

    <h2>Zones</h2>
    <div id="zone-stats"></div>

    <h2>Productivity <span class="hint" style="text-transform:none">(cups/min, proxy)</span></h2>
    <div id="throughput"></div>
    <div id="cash-collected"></div>

    <h2>Payroll <span class="hint" style="text-transform:none">(hours &times; wage)</span></h2>
    <div id="payroll-out"></div>

    <h2>Alerts</h2>
    <div id="alerts"></div>
  </div>
</div>

<script>
const SOURCE = "{SRC}";
const stage = document.getElementById('stage');
const overlays = document.getElementById('overlays');
const feed = document.getElementById('feed');
let zones = [];
let latestDetections = [];

function scale() {{
  return {{ x: feed.clientWidth / 640, y: feed.clientHeight / 360 }};
}}

async function pollDetections() {{
  try {{
    const dets = await (await fetch(`/api/godseye/detections/${{SOURCE}}`)).json();
    latestDetections = dets;
    renderOverlays();
  }} catch(e) {{}}
  setTimeout(pollDetections, 600);
}}

function renderOverlays() {{
  overlays.innerHTML = '';
  const s = scale();
  for (const z of zones) {{
    const [x1,y1,x2,y2] = z.bbox;
    const div = document.createElement('div');
    div.className = 'zone';
    div.style.left = (x1*s.x)+'px'; div.style.top = (y1*s.y)+'px';
    div.style.width = ((x2-x1)*s.x)+'px'; div.style.height = ((y2-y1)*s.y)+'px';
    div.innerHTML = `<div class="tag">${{z.label}}</div>`;
    overlays.appendChild(div);
  }}
  for (const d of latestDetections) {{
    const [x1,y1,x2,y2] = d.bbox;
    const div = document.createElement('div');
    div.className = 'box' + (d.class === 'person' ? ' person' : '');
    div.style.left = (x1*s.x)+'px'; div.style.top = (y1*s.y)+'px';
    div.style.width = ((x2-x1)*s.x)+'px'; div.style.height = ((y2-y1)*s.y)+'px';
    const label = d.identity ? `${{d.identity}}` : `${{d.class}} #${{d.track_id}}`;
    div.innerHTML = `<div class="tag">${{label}} (${{Math.round(d.conf*100)}}%)</div>`;
    div.title = label;
    div.onclick = () => showProfile(d);
    overlays.appendChild(div);
  }}
}}

async function showProfile(d) {{
  const panel = document.getElementById('selected-panel');
  panel.innerHTML = 'Loading&hellip;';
  const profile = await (await fetch(`/api/godseye/track/${{SOURCE}}/${{d.track_id}}`)).json();
  if (profile.error) {{ panel.innerHTML = 'No history yet for this track.'; return; }}
  panel.innerHTML = `
    <div><span class="k">Class:</span> ${{profile.class_name}}</div>
    <div><span class="k">Identity:</span> ${{profile.identity_name || 'unrecognized'}}</div>
    <div><span class="k">Track ID:</span> ${{profile.track_id}}</div>
    <div><span class="k">First seen:</span> ${{profile.first_seen.replace('T',' ').split('.')[0]}}</div>
    <div><span class="k">Last seen:</span> ${{profile.last_seen.replace('T',' ').split('.')[0]}}</div>
    <div><span class="k">On screen for:</span> ${{profile.dwell_seconds}}s across ${{profile.frame_count}} sampled frames</div>
  `;
}}

async function loadZoneStats() {{
  zones = await (await fetch(`/api/godseye/zones/${{SOURCE}}`)).json();
  const stats = await (await fetch(`/api/godseye/zone-stats/${{SOURCE}}`)).json();
  document.getElementById('zone-stats').innerHTML = stats.length ? stats.map(z =>
    `<div class="panel"><b>${{z.label}}</b><br><span class="k">Unique tracks:</span> ${{z.unique_tracks}} &middot; <span class="k">Events:</span> ${{z.total_events}}</div>`
  ).join('') : '<div class="hint">No zones drawn yet. Shift+drag on the feed to add one.</div>';
}}

async function loadAlerts() {{
  const alerts = await (await fetch(`/api/godseye/anomalies/${{SOURCE}}`)).json();
  document.getElementById('alerts').innerHTML = alerts.length ? alerts.map(a =>
    `<div class="alert ${{a.kind}}"><b>${{a.kind.replace('_',' ')}}</b> &middot; ${{a.ts.replace('T',' ').split('.')[0]}}<br>${{a.note}}</div>`
  ).join('') : '<div class="hint">No anomalies logged yet.</div>';
}}

async function loadThroughput() {{
  const rows = await (await fetch(`/api/godseye/throughput/${{SOURCE}}?class_name=cup`)).json();
  const box = document.getElementById('throughput');
  if (!rows.length) {{ box.innerHTML = '<div class="hint">No cup-class detections logged yet on this feed.</div>'; return; }}
  const last = rows.slice(-6);
  const avg = (last.reduce((s,r)=>s+r.count,0) / last.length).toFixed(1);
  box.innerHTML = `<div class="panel"><b>~${{avg}} cups/min</b> <span class="k">(avg, last ${{last.length}} min)</span><br>` +
    last.map(r => `<span class="k">${{r.minute.split('T')[1]}}</span> &mdash; ${{r.count}}`).join('<br>') + `</div>
    <div class="hint">Counts new "cup"-class detections entering frame per minute. Proxy for throughput, not a verified POS count.</div>`;
}}

async function loadIdentities() {{
  const ids = await (await fetch('/api/identities')).json();
  const sel = document.getElementById('identity-select');
  sel.innerHTML = '<option value="">Pick an enrolled identity&hellip;</option>' +
    ids.map(n => `<option value="${{n}}">${{n}}</option>`).join('');
  sel.onchange = loadPayroll;
  loadPayroll();
}}

async function loadPayroll() {{
  const name = document.getElementById('identity-select').value;
  const box = document.getElementById('payroll-out');
  if (!name) {{ box.innerHTML = '<div class="hint">Pick an identity above.</div>'; return; }}
  const s = await (await fetch(`/api/payroll/summary/${{encodeURIComponent(name)}}`)).json();
  if (!s.ok) {{ box.innerHTML = `<div class="hint">${{s.error}}</div>`; return; }}
  box.innerHTML = `<div class="panel">
    <b>${{name}}</b><br>
    <span class="k">Hourly wage:</span> $${{s.hourly_wage}}
    <input id="wage-input" type="number" value="${{s.hourly_wage}}" style="width:70px;background:#1e1e1e;color:#eee;border:1px solid #2c2c2c;border-radius:4px;margin-left:6px">
    <button class="btn secondary" style="margin:0 0 0 6px;padding:4px 8px" onclick="saveWage('${{name}}')">Set</button><br>
    <span class="k">Days counted:</span> ${{s.days_counted}}<br>
    <span class="k">Total hours:</span> ${{s.total_hours}}<br>
    <b>Estimated pay: $${{s.estimated_pay}}</b>
  </div>`;
}}

async function saveWage(name) {{
  const val = document.getElementById('wage-input').value;
  await fetch(`/api/payroll/wage/${{encodeURIComponent(name)}}`, {{
    method: 'POST', headers: {{'Content-Type':'application/json'}}, body: JSON.stringify({{hourly_wage: val}})
  }});
  loadPayroll();
}}

async function loadCashCollected() {{
  const c = await (await fetch(`/api/payroll/cash/${{SOURCE}}`)).json();
  document.getElementById('cash-collected').innerHTML =
    `<div class="panel"><b>~$${{c.estimated_cash}}</b> <span class="k">collected (proxy: ${{c.cup_count}} cups &times; $${{c.price_per_cup}})</span></div>`;
}}

async function runReview() {{
  const name = document.getElementById('identity-select').value;
  const out = document.getElementById('review-out');
  if (!name) {{ out.innerHTML = '<p class="hint">Pick someone first.</p>'; return; }}
  out.innerHTML = '<p class="hint">Asking Opus 4.8&hellip;</p>';
  const res = await (await fetch(`/api/review/${{encodeURIComponent(name)}}`, {{method:'POST'}})).json();
  if (!res.ok) {{ out.innerHTML = `<p class="hint">${{res.error}}</p>`; return; }}
  out.innerHTML = `<div class="review-note">${{res.note}}</div>`;
}}

// Shift+drag to draw a zone
let dragStart = null, dragBox = null;
stage.addEventListener('mousedown', (e) => {{
  if (!e.shiftKey) return;
  const r = stage.getBoundingClientRect();
  dragStart = {{x: e.clientX - r.left, y: e.clientY - r.top}};
  dragBox = document.createElement('div');
  dragBox.className = 'zone';
  dragBox.style.borderColor = '#fff';
  overlays.appendChild(dragBox);
}});
stage.addEventListener('mousemove', (e) => {{
  if (!dragStart || !dragBox) return;
  const r = stage.getBoundingClientRect();
  const cx = e.clientX - r.left, cy = e.clientY - r.top;
  const x = Math.min(dragStart.x, cx), y = Math.min(dragStart.y, cy);
  dragBox.style.left = x+'px'; dragBox.style.top = y+'px';
  dragBox.style.width = Math.abs(cx-dragStart.x)+'px';
  dragBox.style.height = Math.abs(cy-dragStart.y)+'px';
}});
stage.addEventListener('mouseup', async (e) => {{
  if (!dragStart || !dragBox) return;
  const s = scale();
  const r = stage.getBoundingClientRect();
  const cx = e.clientX - r.left, cy = e.clientY - r.top;
  const x1 = Math.min(dragStart.x, cx)/s.x, x2 = Math.max(dragStart.x, cx)/s.x;
  const y1 = Math.min(dragStart.y, cy)/s.y, y2 = Math.max(dragStart.y, cy)/s.y;
  dragBox.remove(); dragStart = null; dragBox = null;
  if (Math.abs(x2-x1) < 10 || Math.abs(y2-y1) < 10) return;
  const label = prompt('Label this zone (e.g. Entrance, Zebra Crossing):');
  if (!label) return;
  await fetch(`/api/godseye/zones/${{SOURCE}}`, {{
    method: 'POST', headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify({{label, x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x2), y2: Math.round(y2)}})
  }});
  loadZoneStats();
}});

pollDetections();
loadZoneStats();
loadAlerts();
loadIdentities();
loadThroughput();
loadCashCollected();
setInterval(loadZoneStats, 4000);
setInterval(loadAlerts, 4000);
setInterval(loadThroughput, 5000);
setInterval(loadCashCollected, 5000);
</script>
</body></html>
"""


@app.get("/")
def root(request: Request):
    return RedirectResponse("/godseye" if require_login(request) else "/login")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    if not require_login(request):
        return RedirectResponse("/login", status_code=303)
    return """
<!doctype html>
<html><head><title>Sentinel Attendance</title>
<meta http-equiv="refresh" content="5">
<style>
body{font-family:-apple-system,sans-serif;background:#0a0a0a;color:#eee;margin:0;padding:24px}
h1{font-size:20px;margin-bottom:4px} h2{font-size:14px;color:#999;margin-top:28px}
table{width:100%;border-collapse:collapse;margin-top:8px}
th,td{text-align:left;padding:8px;border-bottom:1px solid #222;font-size:13px}
th{color:#888;font-weight:500}
.stream{margin-top:12px;border:1px solid #222;border-radius:8px;overflow:hidden;max-width:768px}
.stream img{display:block;width:100%}
.badge{background:#1a1;padding:2px 8px;border-radius:10px;font-size:11px}
</style></head>
<body>
<h1>Sentinel Attendance &mdash; POC</h1>
<div class="stream"><img src="/stream/sample"/></div>

<h2>Working hours today</h2>
<table id="wh"><thead><tr><th>Name</th><th>First seen</th><th>Last seen</th><th>Span (min)</th><th>Sightings</th></tr></thead>
<tbody></tbody></table>

<h2>Recent sightings</h2>
<table id="rs"><thead><tr><th>Name</th><th>Time</th><th>Score</th><th>Source</th></tr></thead>
<tbody></tbody></table>

<script>
async function load(){
  const wh = await (await fetch('/api/working-hours')).json();
  document.querySelector('#wh tbody').innerHTML = wh.map(r =>
    `<tr><td>${r.name}</td><td>${r.first_seen}</td><td>${r.last_seen}</td><td>${r.span_minutes}</td><td>${r.sightings}</td></tr>`
  ).join('');
  const rs = await (await fetch('/api/recent')).json();
  document.querySelector('#rs tbody').innerHTML = rs.map(r =>
    `<tr><td>${r.name}</td><td>${r.ts.split('T')[1].split('.')[0]}</td><td>${r.match_score.toFixed(3)}</td><td>${r.source}</td></tr>`
  ).join('');
}
load(); setInterval(load, 3000);
</script>
</body></html>
"""
