"""Run this once. It prints a URL: open it in the Chrome profile
already signed into FlyWire, approve, copy the token back here."""
from caveclient import CAVEclient

client = CAVEclient(server_address="https://global.daf-apis.com")
client.auth.get_new_token()
