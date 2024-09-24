import tomllib

config = {}

with open("understanding.toml", "rb") as f:
    config["understanding"] = tomllib.load(f)
