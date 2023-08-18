import json

try:
    with open('./variables/vars.json', 'r') as f:
        existing_data = json.load(f)
except FileNotFoundError:
    existing_data = {}


def save_var(var_name: str, var_value) -> None:
    existing_data[var_name] = var_value
    
    # Save all data back to the file
    with open('./variables/vars.json', 'w') as f:
        json.dump(existing_data, f, indent=4)


def get_all_vars() -> None:
    if not existing_data:
        print('There are no variables currently stored.')
    else:
        for var_name, var_value in existing_data.items():
            print(f"{var_name}: {var_value}")


def get_value_by_var_name(var_name: str, default=None):
    return existing_data.get(var_name, default)


def update_value_by_var_name(var_name: str, new_value, default=None):
    existing_data[var_name] = new_value if var_name in existing_data else default

    with open('./variables/vars.json', 'w') as f:
        json.dump(existing_data, f, indent=4)
