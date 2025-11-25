import glob
import pandas as pd
import os

get_users = {
    "emotibit": lambda path: get_users_from_emotibit(path),
    "shimmer": lambda path: get_users_from_shimmer(path),
    "empatica": lambda path: get_users_from_empatica(path),
}

# --- Utils ---


def get_users_from_emotibit(folder_path):
    """Users in Emotibit = names of subfolders (numeric)."""
    return {
        name
        for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name)) and name.isdigit()
    }


def get_users_from_shimmer(folder_path):
    """Users in Shimmer = CSV filenames without extension (numeric)."""
    return {
        os.path.splitext(name)[0]
        for name in os.listdir(folder_path)
        if name.endswith(".csv") and os.path.splitext(name)[0].isdigit()
    }


def get_users_from_empatica(folder_path):
    """Users in Empatica = CSV filenames without extension (numeric)."""
    return {
        os.path.splitext(name)[0]
        for name in os.listdir(folder_path)
        if name.endswith(".csv") and os.path.splitext(name)[0].isdigit()
    }


def find_common_users(root_folder, folders):
    users_sets = []
    for folder, func in folders.items():
        path = os.path.join(root_folder, folder)
        if not os.path.isdir(path):
            print(f"⚠️ {path} not found")
            return []
        users_sets.append(func(path))

    # Intersection of users across the three devices
    if not users_sets:
        return []
    common_users = set.intersection(*users_sets)
    return sorted(common_users, key=lambda x: int(x))


def load_data(base_path="./data", cut=True):

    devices_global = [
        f.name for f in os.scandir(base_path)
        if f.is_dir() and f.name != "Stamps"
    ]

    def base(dev):
        if dev.startswith("shimmer"):
            return "shimmer"
        if dev.startswith("emotibit"):
            return "emotibit"
        if dev.startswith("empatica"):
            return "empatica"
        return None

    # --- Folder dictionary ---
    folders = {dev: get_users[base(dev)] for dev in devices_global}

    data = {dev: {} for dev in devices_global}
    data["stamps"] = {}
    users_list = find_common_users(base_path, folders=folders)

    common_columns = {dev: set() for dev in devices_global}

    # --------------------------------------------------
    # SHIMMER
    # --------------------------------------------------
    for device in devices_global:
        if base(device) == "shimmer":

            csv_files = sorted(glob.glob(f"{base_path}/{device}/*.csv"))

            for csv in csv_files:
                user = os.path.splitext(os.path.basename(csv))[0]
                if user not in users_list:
                    continue

                df = pd.read_csv(csv, sep=";").astype(float)
                data[device][user] = {}

                for col in ["Bvp", "Gsr"]:
                    if col in df.columns:
                        idx = df[col].dropna().index
                        data[device][user][col] = (
                            df[["Timestamp", col]]
                            .iloc[idx]
                            .sort_values("Timestamp")
                            .reset_index(drop=True)
                        )
                        common_columns[device].add(col)

    # --------------------------------------------------
    # EMOTIBIT
    # --------------------------------------------------
    for device in devices_global:
        if base(device) == "emotibit":

            col_map = {"PG": "Bvp", "EA": "Gsr"}

            for user in users_list:
                data[device][user] = {}

                for key, val in col_map.items():
                    files = glob.glob(
                        f"{base_path}/{device}/{user}/parsed/*{key}.csv")
                    if not files:
                        continue

                    df = pd.read_csv(files[0])[
                        ["LocalTimestamp", key]].astype(float)
                    df.rename(
                        columns={"LocalTimestamp": "Timestamp", key: val}, inplace=True)

                    idx = df[val].dropna().index
                    data[device][user][val] = (
                        df[["Timestamp", val]]
                        .iloc[idx]
                        .sort_values("Timestamp")
                        .reset_index(drop=True)
                    )
                    common_columns[device].add(val)

    # --------------------------------------------------
    # EMPATICA
    # --------------------------------------------------
    for device in devices_global:
        if base(device) == "empatica":

            csv_files = sorted(glob.glob(f"{base_path}/{device}/*.csv"))

            for csv in csv_files:
                user = os.path.splitext(os.path.basename(csv))[0]
                if user not in users_list:
                    continue

                df = pd.read_csv(csv, sep=";")
                data[device][user] = {}

                for col in ["Bvp", "Gsr"]:
                    if col in df.columns:
                        idx = df[col].dropna().index
                        data[device][user][col] = (
                            df[["Timestamp", col]]
                            .iloc[idx]
                            .sort_values("Timestamp")
                            .reset_index(drop=True)
                        )
                        common_columns[device].add(col)

    # --------------------------------------------------
    # COMMON COLUMNS
    # --------------------------------------------------
    common = list(set.intersection(*common_columns.values()))
    common.sort()

    # --------------------------------------------------
    # EVENTS (STAMPS)
    # --------------------------------------------------
    for user in users_list:
        events = pd.read_csv(f"{base_path}/Stamps/{user}.txt",
                             header=None, names=['Event', 'Timestamp']
                             )
        events["Timestamp"] = pd.to_numeric(
            events["Timestamp"], errors="coerce").astype(float)
        events["ElapsedTime"] = events["Timestamp"] - events["Timestamp"].min()
        data["stamps"][str(user)] = events

    # --------------------------------------------------
    # CUT ACCORDING TO EVENTS
    # --------------------------------------------------
    if cut:
        for device in devices_global:
            for user in data[device]:
                events = data["stamps"][user]

                start_baseline = events[events["Event"] ==
                                        "full_test_start"]["Timestamp"].values[0]
                finish = events[events["Event"] ==
                                "full_test_end"]["Timestamp"].values[0]

                for col in data[device][user]:
                    df = data[device][user][col]
                    df = df[
                        (df["Timestamp"] >= start_baseline) &
                        (df["Timestamp"] <= finish)
                    ].reset_index(drop=True)
                    data[device][user][col] = df

    return devices_global, users_list, data, common
