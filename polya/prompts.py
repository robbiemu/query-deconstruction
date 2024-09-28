import tomllib

config = {}

with open("understanding.toml", "rb") as f:
    config["understanding"] = tomllib.load(f)

with open("plan.toml", "rb") as f:
    config["plan"] = tomllib.load(f)
