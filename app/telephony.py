"""
Theft-aversion auto-dial: places a real outbound call via Vobiz (Plivo-compatible
API) to Shreyas's phone when an anomaly is flagged. This does NOT claim to detect
theft/intent, it dials on the same lighting-drop / motion-spike signals already
logged, so a human can immediately check the camera.

Requires PUBLIC_BASE_URL to be set to this server's real public URL once deployed
(Vobiz needs to fetch the answer_url over the internet, localhost will not work).
"""
import os
import json
import urllib.request
import urllib.parse

CRED_PATH = os.path.expanduser("~/.claude/credentials/vobiz.env")
API_BASE = "https://api.vobiz.ai/api"
FROM_DID = "918071583442"  # proven-working outbound caller ID, see dograh-hq-personal-droplet-blr1 memory

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8811")


def _load_creds():
    creds = {}
    with open(CRED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            creds[k] = v
    return creds


def trigger_alert_call(to_number: str, message: str):
    creds = _load_creds()
    auth_id = creds["VOBIZ_AUTH_ID"]
    auth_token = creds["VOBIZ_AUTH_TOKEN"]

    answer_url = f"{PUBLIC_BASE_URL}/telephony/theft-alert-xml?" + urllib.parse.urlencode({"msg": message})

    body = json.dumps({
        "from": FROM_DID,
        "to": to_number,
        "answer_url": answer_url,
        "answer_method": "GET",
    }).encode()

    req = urllib.request.Request(
        url=f"{API_BASE}/v1/Account/{auth_id}/Call/",
        data=body,
        headers={
            "content-type": "application/json",
            "X-Auth-ID": auth_id,
            "X-Auth-Token": auth_token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return {"ok": True, "response": data}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "error": e.read().decode()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def theft_alert_xml(message: str) -> str:
    """Plivo-compatible XML Vobiz fetches when the call connects."""
    safe_msg = message.replace("&", "and")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak voice="WOMAN" language="en-US">Sentinel alert. {safe_msg}. Repeating. {safe_msg}.</Speak>
</Response>"""
