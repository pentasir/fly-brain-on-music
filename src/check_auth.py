"""Run once to verify CAVEclient auth works before pulling real data."""
from caveclient import CAVEclient

client = CAVEclient("flywire_fafb_production")
print("Connected. Available materialization versions:")
print(client.materialize.get_versions())
