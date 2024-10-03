import tomllib

config = {}

with open("understanding.toml", "rb") as f:
    config["understanding"] = tomllib.load(f)

with open("devise_plan.toml", "rb") as f:
    config["devise_plan"] = tomllib.load(f)

with open("carry_out_plan.toml", "rb") as f:
    config["carry_out_plan"] = tomllib.load(f)
