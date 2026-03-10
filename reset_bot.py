import requests
import os

token = "8534413276:AAHzqgxVTOL2fapd8NV7UjppF4NXr1zSUek"
url = f"https://api.telegram.org/bot{token}"

print("Resetting Telegram Bot state...")

# 1. Delete Webhook (just in case)
r = requests.get(f"{url}/deleteWebhook?drop_pending_updates=True")
print(f"Delete Webhook: {r.json()}")

# 2. Get Me
r = requests.get(f"{url}/getMe")
print(f"Bot Info: {r.json()}")

# 3. Check for recent updates
r = requests.get(f"{url}/getUpdates?offset=-1")
print(f"Latest Update: {r.json()}")

print("Done.")
