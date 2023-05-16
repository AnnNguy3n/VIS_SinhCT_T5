from base import Base
import pandas as pd
import numpy as np
import os
from datetime import datetime
import shared_func


class Method(Base):
    def __init__(self, data: pd.DataFrame, path: str, measure="geomean", target=1.3, IRpC=1.06, max_loop=10) -> None:
        '''
        # Các tham số
        * measure: Cách đánh giá công thức sau khi đã có lợi nhuận các năm
        * target: Ngưỡng để lưu công thức theo cách đánh giá trên
        * IRpC: Interest Rate per Cycle - tỉ lệ vốn nhận được sau một chu kì không đầu tư
        * max_loop: số lần lặp tối đa để tìm ngưỡng tối ưu của mỗi công thức

        # Cách sinh công thức
        Sinh vét cạn các công thức có các cụm có cùng cấu trúc

        # Cách đánh giá công thức
        * Không đầu tư ở năm đầu tiên, các value của năm đó dùng để lặp tìm ngưỡng tối ưu
        * Từ năm thứ hai trở đi, đầu tư tất cả các công ty có value vượt ngưỡng trong cả
        năm đầu tư và năm trước đó. Lợi nhuận trong năm đó bằng trung bình cộng lợi nhuận
        của tất cả các công ty được đầu tư trong năm đó
        * Có 2 trường hợp không đầu tư: Không có công ty nào vượt ngưỡng của cả năm đầu
        tư và năm trước đó (lý do a), hoặc không có công ty nào vượt ngưỡng (lý do b)
        * Khi không đầu tư, tiền vốn được gửi ngân hàng và nhận lợi nhuận
        * Sau 1 năm không đầu tư vì lí do b, năm sau đó sẽ đầu tư mọi công ty có value
        vượt ngưỡng
        '''
        super().__init__(data, path)
        self.measure = getattr(shared_func, measure)
        self.__measure = measure
        self.target = target
        self.IRpC = IRpC
        self.max_loop = max_loop

    def _measure(self, weight):
        loop_threshold = weight[self.INDEX[-2]:self.INDEX[-1]]
        loop_threshold = np.unique(loop_threshold)
        loop_threshold[::-1].sort()
        if (loop_threshold <= -1.7976931348623157e+308).all():
            return -1.0, -1.0, -1.0, -1.0

        max_threshold = 1.7976931348623157e+308
        max_profit = -1.0
        count_loop = 0
        for threshold in loop_threshold:
            count_loop += 1
            list_profit = []
            reason = 0
            for i in range(self.INDEX.shape[0]-3):
                inv_cyc_val = weight[self.INDEX[-i-3]:self.INDEX[-i-2]]
                inv_cyc_sym = self.data.iloc[self.INDEX[-i-3]:self.INDEX[-i-2]]["SYMBOL"].reset_index(drop=True)
                if reason == 0: # Không đầu tư do không có công ty nào vượt ngưỡng 2 năm liền
                    pre_cyc_val = weight[self.INDEX[-i-2]:self.INDEX[-i-1]]
                    pre_cyc_sym = self.data.iloc[self.INDEX[-i-2]:self.INDEX[-i-1]]["SYMBOL"].reset_index(drop=True)
                    a = np.where(pre_cyc_val > threshold)[0]
                    b = np.where(inv_cyc_val > threshold)[0]
                    coms = np.intersect1d(pre_cyc_sym[a], inv_cyc_sym[b])
                    if len(coms) == 0:
                        list_profit.append(self.IRpC)
                        if b.shape[0] == 0:
                            reason = 1
                    else:
                        inv_pro = self.PROFIT[self.INDEX[-i-3]:self.INDEX[-i-2]][inv_cyc_sym.isin(coms)]
                        profit = inv_pro.mean()
                        list_profit.append(profit)
                else: # reason == 1, Không đầu tư do không có công ty nào vượt ngưỡng trong năm trước đó
                    b = np.where(inv_cyc_val > threshold)[0]
                    coms = inv_cyc_sym[b]
                    if len(coms) == 0:
                        list_profit.append(self.IRpC)
                    else:
                        inv_pro = self.PROFIT[self.INDEX[-i-3]:self.INDEX[-i-2]][inv_cyc_sym.isin(coms)]
                        profit = inv_pro.mean()
                        list_profit.append(profit)
                        reason = 0

            total_profit = self.measure(np.array(list_profit))
            if total_profit > max_profit:
                max_profit = total_profit
                max_threshold = threshold

            if count_loop == self.max_loop:
                break

        if max_profit < self.target:
            return -1.0, -1.0, -1.0, -1.0

        inv_cyc_val = weight[self.INDEX[0]:self.INDEX[1]]
        inv_cyc_sym = self.data.iloc[self.INDEX[0]:self.INDEX[1]]["SYMBOL"].reset_index(drop=True)
        if reason == 0:
            pre_cyc_val = weight[self.INDEX[1]:self.INDEX[2]]
            pre_cyc_sym = self.data.iloc[self.INDEX[1]:self.INDEX[2]]["SYMBOL"].reset_index(drop=True)
            a = np.where(pre_cyc_val > threshold)[0]
            b = np.where(inv_cyc_val > threshold)[0]
            coms = np.intersect1d(pre_cyc_sym[a], inv_cyc_sym[b])
            if len(coms) == 0:
                inv_profit = self.IRpC
                # inv_com = ["NI"]
            else:
                inv_pro = self.PROFIT[self.INDEX[0]:self.INDEX[1]][inv_cyc_sym.isin(coms)]
                inv_profit = inv_pro.mean()
                # inv_com = coms
        else:
            # reason == 1
            b = np.where(inv_cyc_val > threshold)[0]
            coms = inv_cyc_sym[b]
            if len(coms) == 0:
                inv_profit = self.IRpC
                # inv_com = ["NI"]
            else:
                inv_pro = self.PROFIT[self.INDEX[0]:self.INDEX[1]][inv_cyc_sym.isin(coms)]
                inv_profit = inv_pro.mean()
                # inv_com = coms

        return max_profit, max_threshold, ..., inv_profit

    def fill_operand(self, formula, struct, idx, temp_0, temp_op, temp_1, mode, add_sub_done, mul_div_done):
        if mode == 0: # Sinh dấu cộng trừ đầu mỗi cụm
            gr_idx = list(struct[:,2]-1).index(idx)

            start = 0
            if (formula[0:idx] == self.last_formula[0:idx]).all():
                start = self.last_formula[idx]

            for op in range(start, 2):
                new_formula = formula.copy()
                new_struct = struct.copy()
                new_formula[idx] = op
                new_struct[gr_idx,0] = op
                if op == 1:
                    new_add_sub_done = True
                    new_formula[new_struct[gr_idx+1:,2]-1] = 1
                    new_struct[gr_idx+1:,0] = 1
                else:
                    new_add_sub_done = False

                if self.fill_operand(new_formula, new_struct, idx+1, temp_0, temp_op, temp_1, 1, new_add_sub_done, mul_div_done):
                    return True

        elif mode == 1:
            start = 0
            if (formula[0:idx] == self.last_formula[0:idx]).all():
                start = self.last_formula[idx]

            valid_operand = shared_func.get_valid_operand(formula, struct, idx, start, self.OPERAND.shape[0])
            if valid_operand.shape[0] > 0:
                if formula[idx-1] < 2:
                    temp_op_new = formula[idx-1]
                    temp_1_new = self.OPERAND[valid_operand].copy()
                else:
                    temp_op_new = temp_op
                    if formula[idx-1] == 2:
                        temp_1_new = temp_1 * self.OPERAND[valid_operand]
                    else:
                        temp_1_new = temp_1 / self.OPERAND[valid_operand]

                if idx + 1 == formula.shape[0] or (idx+2) in struct[:,2]:
                    if temp_op_new == 0:
                        temp_0_new = temp_0 + temp_1_new
                    else:
                        temp_0_new = temp_0 - temp_1_new
                else:
                    temp_0_new = np.array([temp_0]*valid_operand.shape[0])

                if idx + 1 == formula.shape[0]:
                    temp_0_new[np.isnan(temp_0_new)] = -1.7976931348623157e+308
                    temp_0_new[np.isinf(temp_0_new)] = -1.7976931348623157e+308
                    for w_i in range(temp_0_new.shape[0]):
                        weight = temp_0_new[w_i]
                        temp_formula = formula.copy()
                        temp_formula[idx] = valid_operand[w_i]
                        profit, threshold, inv_com, inv_pro = self._measure(weight)
                        if profit >= self.target:
                            self.list_formula.append(temp_formula)
                            self.list_history_profit.append(profit)
                            self.list_threshold.append(threshold)
                            # self.list_inv_coms.append(inv_com)
                            self.list_inv_prof.append(inv_pro)
                            self.count[0:3:2] += 1

                    self.last_formula[:] = formula[:]
                    self.last_formula[idx] = self.OPERAND.shape[0]
                    if self.count[0] >= self.count[1] or self.count[2] >= self.count[3]:
                        return True
                else:
                    temp_list_formula = np.array([formula]*valid_operand.shape[0])
                    temp_list_formula[:,idx] = valid_operand
                    if idx + 2 in struct[:,2]:
                        if add_sub_done:
                            new_idx = idx + 2
                            new_mode = 1
                        else:
                            new_idx = idx + 1
                            new_mode = 0
                    else:
                        if mul_div_done:
                            new_idx = idx + 2
                            new_mode = 1
                        else:
                            new_idx = idx + 1
                            new_mode = 2

                    for i in range(valid_operand.shape[0]):
                        if self.fill_operand(temp_list_formula[i], struct, new_idx, temp_0_new[i], temp_op_new, temp_1_new[i], new_mode, add_sub_done, mul_div_done):
                            return True

        elif mode == 2:
            start = 2
            if (formula[0:idx] == self.last_formula[0:idx]).all():
                start = self.last_formula[idx]

            if start == 0:
                start = 2

            valid_op = shared_func.get_valid_op(struct, idx, start)
            for op in valid_op:
                new_formula = formula.copy()
                new_struct = struct.copy()
                new_formula[idx] = op
                if op == 3:
                    new_mul_div_done = True
                    for i in range(idx+2, 2*new_struct[0,1]-1, 2):
                        new_formula[i] = 3

                    for i in range(1, new_struct.shape[0]):
                        for j in range(new_struct[0,1]-1):
                            new_formula[new_struct[i,2] + 2*j + 1] = new_formula[2+2*j]
                else:
                    new_struct[:,3] += 1
                    new_mul_div_done = False
                    if idx == 2*new_struct[0,1] - 2:
                        new_mul_div_done = True
                        for i in range(1, new_struct.shape[0]):
                            for j in range(new_struct[0,1]-1):
                                new_formula[new_struct[i,2] + 2*j + 1] = new_formula[2+2*j]

                if self.fill_operand(new_formula, new_struct, idx+1, temp_0, temp_op, temp_1, 1, add_sub_done, new_mul_div_done):
                    return True

        return False

    def generate(self, num_f_in_a_file=10000, num_f_target=1000000000):
        self.last_formula = ...
        self.list_formula = ...
        self.list_history_profit = ...
        self.list_threshold = ...
        self.list_inv_coms = ...
        self.list_inv_prof = ...
        self.count = ...

        try:
            file_name = "history.npy"
            temp = list(np.load(self.path + file_name, allow_pickle=True))
            self.history = temp
        except:
            self.history = np.array([0, 0]), 0

        self.last_formula = self.history[0].copy()
        self.last_divisor_idx = self.history[1]

        self.count = np.array([0, num_f_in_a_file, 0, num_f_target])
        last_operand = self.last_formula.shape[0] // 2
        num_operand = last_operand - 1

        while True:
            num_operand += 1
            print("Đang chạy sinh công thức có số toán hạng là ", num_operand, ". . .")
            self.list_formula = []
            self.list_history_profit = []
            self.list_threshold = []
            # self.list_inv_coms = []
            self.list_inv_prof = []

            list_uoc_so = []
            for i in range(1, num_operand+1):
                if num_operand % i == 0:
                    list_uoc_so.append(i)

            start_divisor_idx = 0
            if num_operand == last_operand:
                start_divisor_idx = self.history[1]

            formula = np.full(num_operand*2, 0)
            for i in range(start_divisor_idx, len(list_uoc_so)):
                print("Số phần tử trong 1 cụm", list_uoc_so[i])
                struct = np.array([[0, list_uoc_so[i], 1+2*list_uoc_so[i]*j, 0] for j in range(num_operand//list_uoc_so[i])])
                if num_operand != last_operand or i != self.last_divisor_idx:
                    self.last_formula = formula.copy()
                    self.last_divisor_idx = i

                while self.fill_operand(formula, struct, 0, np.zeros(self.OPERAND.shape[1]), 0, np.zeros(self.OPERAND.shape[1]), 0, False, False):
                    self.save_history()

            if self.save_history():
                break

    def save_history(self):
        file_name = "history.npy"
        np.save(self.path + file_name, (self.last_formula, self.last_divisor_idx))
        print("Đã lưu lịch sử.")
        if self.count[0] == 0:
            return False

        inv_time = self.data["TIME"].max()
        df = pd.DataFrame({
            "formula": self.list_formula,
            f"{self.__measure}_profit": self.list_history_profit,
            "Threshold": self.list_threshold,
            # "Invested_coms": self.list_inv_coms,
            "Profit": self.list_inv_prof,
            "TIME": [inv_time] * len(self.list_formula)
        })
        while True:
            pathSave = self.path + f"formula_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".csv"
            if not os.path.exists(pathSave):
                df.to_csv(pathSave, index=False)
                self.count[0] = 0
                self.list_formula = []
                self.list_history_profit = []
                self.list_threshold = []
                # self.list_inv_coms = []
                self.list_inv_prof = []
                print("Đã lưu công thức")
                if self.count[2] >= self.count[3]:
                    raise Exception("Đã sinh đủ công thức theo yêu cầu.")

                return False