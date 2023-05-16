import pandas as pd
import numpy as np
import os


class Base:
    def __init__(self, data: pd.DataFrame, path: str) -> None:
        # Các cột bắt buộc
        dropped_cols = ["TIME", "PROFIT", "SYMBOL"]
        for col in dropped_cols:
            if col not in data.columns:
                raise Exception(f"Thiếu cột {col}")

        # Check kiểu dữ liệu
        if data["TIME"].dtype != "int64": raise
        if data["PROFIT"].dtype != "float64": raise

        # Check sự giảm dần của cột TIME
        if data["TIME"].diff().max() > 0:
            raise Exception("Cột TIME phải giảm dần")

        # Check các chu kì và lấy index
        first_cyc = data["TIME"].min()
        last_cyc = data["TIME"].max()
        array_cyc = data["TIME"].unique()
        self.INDEX = []
        for cyc in range(last_cyc, first_cyc-1, -1):
            if cyc not in array_cyc:
                raise Exception(f"Thiếu chu kì {cyc}")

            self.INDEX.append(data[data["TIME"] == cyc].index[0])

        self.INDEX.append(data.shape[0])
        self.INDEX = np.array(self.INDEX)

        # Check các cột có kiểu dữ liệu không phải là số
        for col in data.columns:
            if col not in dropped_cols and data[col].dtype == "object":
                dropped_cols.append(col)

        print(f"Các cột không được coi là biến: {dropped_cols}")

        # Check path lưu data
        if type(path) != str or not os.path.exists(path):
            raise Exception(f"Không tồn tại thư mục {path}")

        if not path.endswith("/") and not path.endswith("\\"):
            path += "/"

        self.path = path

        # Các thuộc tính
        self.data = data
        self.PROFIT = np.array(data["PROFIT"], dtype=np.float64)

        operand_data = data.drop(columns=dropped_cols)
        operand_name = operand_data.columns
        self.operand_name = {i:operand_name[i] for i in range(len(operand_name))}
        self.OPERAND = np.transpose(np.array(operand_data, dtype=np.float64))
