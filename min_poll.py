import requests
import time

token = "8534413276:AAHzqgxVTOL2fapd8NV7UjppF4NXr1zSUek"
url = f"https://api.telegram.org/bot{token}/getUpdates"

print("Starting minimal loop...")
last_offset = 0

while True:
    try:
        print(f"Polling (offset={last_offset+1})...")
        r = requests.get(url, params={"offset": last_offset + 1, "timeout": 10}, timeout=15)
        print(f"Status: {r.status_code}")
        data = r.json()
        if data.get("ok"):
            res = data["result"]
            print(f"Received {len(res)} updates.")
            for update in res:
                print(f"!!! RECEIVED: {update.get('message', {}).get('text')}")
                last_offset = update["update_id"]
        else:
            print(f"Bot Error: {data}")
    except Exception as e:
        print(f"Req Error: {e}")
    time.sleep(1)
