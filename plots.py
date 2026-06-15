import os
from rsplotlib import use
# from multiprocessing import Process, freeze_support
import rsplotlib.pyplot as plt
from rsplotlib import style
import xlrd
import datetime
import rsnum as np
from rsplotlib.gridspec import GridSpec
# import test_times
# import version
import sys
import re
import time
import json
from rsplotlib.ticker import MaxNLocator, MultipleLocator
import threading
import math
from rsplotlib.pylab import mpl

os.environ["MPLCONFIGDIR"] = "/Applications/plots/rsplotlib_cache"
use("Agg")

mpl.rcParams['font.sans-serif'] = ["Helvetica", 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False

def version(versio):
    print(">>> Script Name: " + "\033[0:32mReport Plots\033[0m")
    print(">>> Script version: " + f"\033[0:33m{versio}\033[0m")
    print(" ")


def read_by_chart():
    config_path2 = "/Applications/plots/Frameworks/config.ini"
    with open(config_path2, "r") as f:
        config_info = f.read()
        config = []
        for i in config_info.split("\n")[:-1]:
            pre_config = list(i.split(","))
            config.append(pre_config)
    return config


def select_chart():
    print(" " + "-" * 78)
    by_chart = read_by_chart()[0]
    print("  \033[0:33mopp\033[0m")
    print("   \033[0:32mlocal \033[0m  ", "1.", by_chart[1].split(" ")[0], " 2.", by_chart[2].split(" ")[0], " 3.", by_chart[3].split(" ")[0], " 4.", by_chart[4].split(" ")[0], " 5.", by_chart[5].split(" ")[0])
    print("    \033[0:31minsight\033[0m", )
    print(" " + "-" * 78)
    print()


def time_delte(ti):
    yyr = {1: 0, 2: 31, 3: 60, 4: 91, 5: 121, 6: 152, 7: 182, 8: 213, 9: 244, 10: 274, 11: 305, 12: 335}
    yy = {1: 0, 2: 31, 3: 59, 4: 90, 5: 120, 6: 151, 7: 181, 8: 212, 9: 243, 10: 273, 11: 304, 12: 334}
    t = ti.split()
    if "/" in t[0]:
        s = t[0].split("/")
    elif "-" in t[0]:
        s = t[0].split("-")
    y = int(s[0])
    y_ = y - 1
    if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0):  # 判断闰年条件
        ys = 365
        ms = yyr
    else:
        if (y_ % 4 == 0 and y_ % 100 != 0) or (y_ % 400 == 0):
            ys = 366
        else:
            ys = 365
        ms = yy
    ss = t[1].split(":")
    if len(ss) < 3:
        ss.append(00)
    y_times = (int(s[0]) * ys + ms[int(s[1])] + int(s[2])) * 24 * 3600
    h_times = int(ss[0]) * 3600 + int(ss[1]) * 60 + int(ss[2])
    return y_times + h_times


def is_number(s_su):
    try:
        float(s_su)
        return True
    except:
        return False


def ret_number(s_ua):
    try:
        return int(s_ua)
    except:
        return None


class Reportopp:
    def __init__(self):
        self.colar_list1 = None
        self.test_result = None
        self.endtime = None
        self.start_time = None
        self.test_iterm = None
        self.ax2 = None
        self.enter_limit = None
        self.ax1 = None
        self.version = None
        self.test_date_list = None
        self.station_name = None
        self.station = None
        self.station_testlist = []
        self.data_csv = []
        self.up_limt = []
        self.down_limt = []
        self.line_type = None
        self.mark = None
        self.if_testtime = 1
        self.dict_summary = []

    def read_test_station(self):
        pre_path = '/Applications/plots'
        path_f = pre_path + "/Frameworks/"
        config_path1 = path_f + "data_Type.json"
        with open(config_path1, "r") as f:
            cvs_data_type = json.load(f, )[self.station_name]
        mark_path = path_f + "config.ini"
        with open(mark_path, "r") as f:
            f_read = f.readlines()
            self.mark = f_read[-2].split("\n")[0]
            self.if_testtime = f_read[-1]
            self.enter_limit = f_read[-3].split("\n")[0]
        if "insight" == cvs_data_type:
            config_path1 = path_f + "testlist.xls"
        elif "local" == cvs_data_type:
            config_path1 = path_f + "testlist_local.xls"
        book = xlrd.open_workbook(config_path1)
        sheet1 = book.sheets()[0]
        nrows = sheet1.nrows
        for i in range(1, nrows):
            check_station = sheet1.row_values(i)
            for j in range(check_station.count("")):
                check_station.remove("")
            if check_station:
                if check_station[1] == self.station_name:
                    self.station = check_station[0]
                    self.station_testlist = check_station[2:]
                    break
        print(" %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "初始化完成。。。")

    def data_read(self, path_csv):
        self.data_csv = []
        try:
            with open(path_csv, 'r', newline='', encoding='utf-8', errors='replace') as csv_date:
                ss_read = csv_date.readlines()
        except:
            with open(path_csv, 'r', newline='', encoding='gbk', errors='replace') as csv_date:
                ss_read = csv_date.readlines()
        check_upper = 0
        for i in range(len(ss_read)):
            self.data_csv.append(
                ss_read[i].split("\t")[0].split(",")[:-1] + ss_read[i].split("\t")[0].split(",")[-1].split("\r"))
            if check_upper == 0:
                if "upper limit" in ss_read[i].split("\t")[0].split(",")[0].lower():
                    up_q = i
                    check_upper = 1
        self.test_iterm = self.data_csv[1]
        test_qty = len(self.data_csv)
        test_qty_start = 0
        for j in self.data_csv:
            if "Measurement Unit" in j[0]:
                test_qty_start = self.data_csv.index(j) + 1
                break
        self.test_date_list = range(test_qty_start, test_qty)
        self.up_limt = ss_read[up_q].split(",")
        self.down_limt = ss_read[up_q + 1].split(",")
        self.station_name = self.data_csv[0][0]
        wipass = self.data_csv[0][1]
        if ":" in wipass:
            self.version = wipass.split(":")[-1]
        else:
            self.version = wipass.split("-")[0]

    def plot_run(self, x, s, lw, c, ls, marker):
        self.ax1.plot(x, s, lw=lw, c=c, ls=ls, marker=marker)

    def test_cpk(self, test_item, s_index):
        test_data = self.data_csv
        try:
            test_rowm = self.test_iterm.index(test_item)
            test_value_list = [float(test_data[i][test_rowm]) for i in self.test_date_list if
                               is_number(test_data[i][test_rowm]) and float(test_data[i][test_rowm]) != 99999]
            if test_value_list:
                avg = sum(test_value_list) / len(test_value_list)
                fang = [(i - avg) ** 2 for i in test_value_list]
                sigm = (sum(fang) / (len(test_value_list) - 1)) ** 0.5
                up_limt = self.up_limt[test_rowm]
                down_limt = self.down_limt[test_rowm]
                if sigm == 0:
                    cp, cpk_, cpu, cpl, cpk = "NA", "NA", "NA", "NA", "NA"
                else:
                    if "NA" not in up_limt and "NA" not in down_limt:
                        c = (float(up_limt) + float(down_limt)) / 2
                        t = (float(up_limt) - float(down_limt)) / 2
                        ca = (avg - c) / t
                        cp = t / 3 / sigm
                        cpk_ = round(cp * abs(1 - ca), 3)
                        cpu = round((float(up_limt) - avg) / (3 * sigm), 3)
                        cpl = round((avg - float(down_limt)) / (3 * sigm), 3)
                        cpk = round(min(cpk_, cpl, cpu), 3)
                    elif "NA" not in up_limt and "NA" in down_limt:
                        cpk = round((float(up_limt) - avg) / (3 * sigm), 3)
                    elif "NA" in up_limt and "NA" not in down_limt:
                        cpk = round((avg - float(down_limt)) / (3 * sigm), 3)
                    else:
                        cpk = "NA"
                try:
                    ss_s = math.log10(abs(cpk))
                    if ss_s > 6:
                        cck = int(ss_s)
                        cpk_1 = round(cpk / (10 ** cck), 2)
                        cpk = f"{cpk_1}E{cck}"
                except:
                    pass
                return round(avg, 3), round(sigm, 3), cpk, max(test_value_list), min(test_value_list)
            else:
                return "NA", "NA", "NA", "NA", "NA"
        except:
            return "NA", "NA", "NA", "NA", "NA"

    @staticmethod
    def test_only1_cpk(sigm, avg, uplimt, downlimt):
        try:
            up_limt = float(uplimt)
            down_limt = float(downlimt)
            c = (float(up_limt) + float(down_limt)) / 2
            t = (float(up_limt) - float(down_limt)) / 2
            ca = (avg - c) / t
            if sigm == 0:
                cp, cpk_, cpu, cpl, cpk = "NA", "NA", "NA", "NA", "NA"
            else:
                cp = t / 3 / sigm
                cpk_ = round(cp * abs(1 - ca), 2)
                cpu = round((float(up_limt) - avg) / (3 * sigm), 2)
                cpl = round((avg - float(down_limt)) / (3 * sigm), 2)
                cpk = round(min(cpk_, cpl, cpu), 2)
        except:
            cpl, cpu, cpk = 0, 0, 0
        return cpl, cpu, cpk

    def colms(self, path_p, byshow, station):
        ng_flot = 0
        if byshow == "Slot Number":
            try:
                byshow_index = self.test_iterm.index("Slot Number")
            except:
                try:
                    byshow_index = self.test_iterm.index("tc=Slot Number tech=Unit")
                except:
                    try:
                        byshow_index = self.test_iterm.index("tech=Unit;tc=Slot Number")
                    except:
                        try:
                            byshow_index = self.test_iterm.index("Test Pass/Fail Status")
                            print(f" 未在csv文件找到 {byshow}")
                        except:
                            print(f" 未在csv文件找到 {byshow}")
                            sys.exit()
            byshow_index1 = self.test_iterm.index("Station ID")
        else:
            try:
                byshow_index = self.test_iterm.index(str(byshow))
                byshow_index1 = ""
            except:
                if byshow == "Version":
                    try:
                        byshow_index = self.test_iterm.index("WiPAS-Version")
                    except:
                        try:
                            byshow_index = self.test_iterm.index("Test Pass/Fail Status")
                            print(f" 未在csv文件找到 {byshow}")
                        except:
                            print(f" 未在csv文件找到 {byshow}")
                            sys.exit()
                elif byshow == "Serial Number":
                    try:
                        byshow_index = self.test_iterm.index("SerialNumber")
                    except:
                        try:
                            byshow_index = self.test_iterm.index("Serial Number")
                        except:
                            try:
                                byshow_index = self.test_iterm.index("Test Pass/Fail Status")
                                print(f" 未在csv文件找到 {byshow}")
                            except:
                                print(f" 未在csv文件找到 {byshow}")
                                sys.exit()
                else:
                    try:
                        byshow_index = self.test_iterm.index("Test Pass/Fail Status")
                        print(f" 未在csv文件找到 {byshow}")
                    except:
                        print(f" 未在csv文件找到 {byshow}")
                        sys.exit()
                byshow_index1 = ""
        plt_idix = 0
        lpdet_list = []
        for test_name in self.station_testlist:
            # ax_list = []
            print(" %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), self.station, plt_idix, test_name)
            test_index = []
            test_name_list = []
            for i in self.test_iterm:
                i_new = i.lower()
                if re.findall(test_name.lower(), i_new, flags=0):
                    if "LPDET_Sense" in i:
                        lpdet_list.append(self.test_iterm.index(i))
                    index_ = self.test_iterm.index(i)
                    test_index.append(index_)
                    test_name_list.append(i)
                if "starttime" in i_new or "start time" in i_new:
                    self.start_time = self.test_iterm.index(i)
                if "endtime" in i_new or "stop time" in i_new:
                    self.endtime = self.test_iterm.index(i)
                if "test pass" in i_new or "fail status" in i_new:
                    self.test_result = self.test_iterm.index(i)
            x = list(i for i in range(1, len(test_index) + 1))
            if len(test_index) == 2:
                len_1, len_2 = 0.9, 0.1
            else:
                len_1, len_2 = 1 - len(test_index) * 0.03, len(test_index) * 0.03
            test_only1 = []
            na_date = 0
            fail_cont = 0
            plt_colar = ["#1431F5", "#72F64A", "#74F9FD", "#FEFB54", "#EA51F7", "#EE8731", "#74247B", "#92693B",
                         "#000000"]

            text_colar_ = {"#1431F5": "w", "#72F64A": "k", "#74F9FD": "k", "#FEFB54": "k",
                           "#EA51F7": "w", "#EE8731": "k", "#74247B": "w", "#92693B": "w", "#000000": "w"}
            dict_i = []
            dis_onliny = {}
            test_time_colm = []
            apple_pass_colm = []
            upper_limit = []
            lower_limit = []
            for i in test_index:
                if "NA" not in self.up_limt[i] and "NA" not in self.down_limt[i]:
                    upper_limit.append(float(self.up_limt[i]))
                    lower_limit.append(float(self.down_limt[i]))
                elif "NA" not in self.up_limt[i] and "NA" in self.down_limt[i]:
                    upper_limit.append(float(self.up_limt[i]))
                    lower_limit.append(None)
                elif "NA" in self.up_limt[i] and "NA" not in self.down_limt[i]:
                    upper_limit.append(None)
                    lower_limit.append(float(self.down_limt[i]))
                else:
                    upper_limit.append(None)
                    lower_limit.append(None)
            if len(x) != 1:
                fig = plt.figure(figsize=(14, 7.5), dpi=144)
                gs = GridSpec(100, 100)
                plt.axis('off')
                self.ax1 = fig.add_subplot(gs[4:55, 4:75])
                # ax_list.append(self.ax1)
                fig = plt.gcf()
                fig.set_facecolor('#FFFFFF')
                self.ax1.patch.set_facecolor("#FFFFFF")
                # 启用主网格（主网格线在主刻度值上）
                self.ax1.grid(c='grey')
                # 启用副网格（副网格线在副刻度值上）
                self.ax1.yaxis.grid(True, which='minor', ls='--', c='#BFBFBF')
                self.ax1.xaxis.grid(True, which='minor', ls='--', c='#BFBFBF')
                # 显式在 self.ax1 上设置主刻度定位器，确保主网格线显示在坐标刻度值上
                self.ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.title(test_name, fontsize=17, fontweight='bold', style="normal")
            else:
                fig = plt.figure(figsize=(14.819, 4.916), dpi=144)
                plt.subplots_adjust(left=0.0006, right=0.999, top=0.997, bottom=0.005)
                gs = GridSpec(100, 100)
                plt.axis('off')
                style.use("default")
                ax_kuang = fig.add_subplot(gs[0:100, 0:100])
                ax_kuang.tick_params(bottom=False, top=False, left=False, right=False)
                plt.xticks([])
                plt.yticks([])
                self.ax1 = fig.add_subplot(gs[10:90, 16:75])
                plt.title(test_name, fontsize=17, fontweight='bold', style="normal")
            ss_max = []
            ss_min = []
            ylimt_up = []
            ylimt_down = []
            s_upper_limit = [float(i) for i in upper_limit if i or ret_number(i) is not None]
            s_lower_limit = [float(i) for i in lower_limit if i or ret_number(i) is not None]
            test_value_total = []
            plot_list = []
            for test_i in self.test_date_list:
                try:
                    sad = self.data_csv[test_i]
                    result = sad[self.test_result].lower()
                    if "pass" in result:
                        test_time = time_delte(sad[self.endtime]) - time_delte(
                            sad[self.start_time])
                        test_time_colm.append(test_time)
                        if "pass" != result:
                            apple_pass_colm.append(test_time)
                except:
                    pass
                s = []
                true_index = []
                for i in test_index:
                    test_value = str(self.data_csv[test_i][i])
                    if test_value == "NA" or not is_number(test_value) or test_value == "":
                        na_date += 1
                        s.append(None)
                    elif float(test_value) == 99999:
                        fail_cont += 1
                    else:
                        s.append(float(test_value))
                        true_index.append(test_index.index(i))
                        if upper_limit[test_index.index(i)]:
                            if s_upper_limit and float(test_value) > float(upper_limit[test_index.index(i)]):
                                fail_cont += 1
                        if lower_limit[test_index.index(i)]:
                            if s_lower_limit and float(lower_limit[test_index.index(i)]) > float(test_value):
                                fail_cont += 1
                        test_value_total.append(float(test_value))
                if byshow_index1 == "":
                    byshow_value = self.data_csv[test_i][byshow_index]
                else:
                    byshow_value = self.data_csv[test_i][byshow_index1] + " " + self.data_csv[test_i][byshow_index]
                if plt_idix == 0:
                    self.dict_summary.append(byshow_value)
                if byshow_value not in dict_i:
                    dict_i.append(byshow_value)
                if len(dict_i) > len(plt_colar):
                    ss1 = len(dict_i) // len(plt_colar)
                    colar_true = plt_colar * ss1
                    s_then = len(dict_i) % len(plt_colar)
                    self.colar_list = colar_true + plt_colar[:s_then]
                else:
                    self.colar_list = plt_colar[:len(dict_i)]
                if s_upper_limit:
                    list(set(s_upper_limit))
                    if s_upper_limit == [0]:
                        max_upper_limit = None
                    else:
                        max_upper_limit = max(s_upper_limit)
                else:
                    max_upper_limit = None
                if s_lower_limit:
                    list(set(s_lower_limit))
                    if s_lower_limit != [0]:
                        min_lower_limit = min(s_lower_limit)
                    else:
                        min_lower_limit = None
                else:
                    min_lower_limit = None

                if len(x) != 1:
                    plot_qty = len(x)
                    plot_qq = len(s)
                    if plot_qty == plot_qq and plot_qty >= 2:
                        if lpdet_list == test_index:
                            s = [i for i in reversed(s)]
                        st = threading.Thread(target=self.plot_run,
                                              args=(x, s, 1.3, self.colar_list[dict_i.index(byshow_value)], '-', '.'))
                        st.start()
                        plot_list.append(st)
                    elif plot_qty != plot_qq and plot_qty >= 2:
                        y = []
                        if true_index:
                            for i in range(max(true_index) + 1):
                                if i in true_index:
                                    y.append(s[true_index.index(i)])
                                else:
                                    true_index.append(i)
                                    y.append(None)
                            true_index.sort()
                            true_index.append(max(true_index) + 1)
                            true_index.remove(0)
                            st = threading.Thread(target=self.plot_run, args=(
                                true_index, y, 1.3, self.colar_list[dict_i.index(byshow_value)], '-', '.'))
                            st.start()
                            plot_list.append(st)
                else:
                    if s:
                        dis_onliny.setdefault(byshow_value, []).append(s)
                        test_only1 += s
            ng_flot_check = "OK"
            if not test_name_list:
                test_name_list = ["1", "2"]
                x = [1, 2]
                ss_max = [2, 2]
                ss_min = [0, 0]
                test_index = [1, 2]
                upper_limit = [None, None]
                lower_limit = [None, None]
                # rsplotlib 的 rcParams 设置 font.sans-serif 时会自动调用
                # _font_resolver，把对应的字体文件（如 Arial Unicode.ttf）
                # 注册到 plotters 的字体数据库中，从而真正影响文本渲染。
                # 因此下面的两行设置是有实际效果的——渲染"请确认关键字是否正确"
                # 时将使用 Arial Unicode MS。

                self.ax1.text(1.3, 1.2, '请确认关键字是否正确', fontsize=35, family='Arial Unicode MS', c='r')
                print(" 请确认关键字是否正确")
                ng_flot += 1
                ng_flot_check = "NG"
            else:
                for i in range(test_name_list.count(1)):
                    test_name_list.remove(1)
                for i in range(test_name_list.count(2)):
                    test_name_list.remove(2)
            max_, min_ = 1, 0
            show_i_1 = []

            if len(x) == 1:
                ss = lower_limit + test_only1 + upper_limit
                test_only2 = [float(i) for i in ss if is_number(i)]
                min_test_only1 = min(test_only2)
                max_test_only1 = max(test_only2)
                delta_test_only1 = max_test_only1 - min_test_only1
                # s2 = int((max(test_only1) - min(test_only1)) / s1)
                if lower_limit == [None]:
                    min_test_only1 = min_test_only1 - delta_test_only1 * 0.2
                if upper_limit == [None]:
                    max_test_only1 = max_test_only1 + delta_test_only1 * 0.2
                bins = np.linspace(min_test_only1, max_test_only1, 70)
                test_only1 = [float(i) for i in test_only1 if is_number(i)]
                if len(dict_i) == 1:
                    n, bins, _patches = self.ax1.hist(test_only1, bins=bins, align="mid", facecolor="b", alpha=0.8)
                    n_max = max(bins)
                    n_min = min(bins)
                    y_max = max(n)  # 最高的柱子高度
                    max_, min_ = y_max * 1.2, 0
                else:
                    y_limit = []
                    keys_only = []
                    vals = []
                    colors_list = []
                    if len(test_only1) > len(plt_colar):
                        ss1 = len(test_only1) // len(plt_colar)
                        colar_true = plt_colar * ss1
                        s_then = len(test_only1) % len(plt_colar)
                        self.colar_list = colar_true + plt_colar[:s_then]
                    else:
                        self.colar_list = plt_colar[:len(dict_i)]
                    for key, value in dis_onliny.items():
                        if key not in keys_only:
                            if value != [[None]]:
                                keys_only.append(key)
                                ss1 = []
                                for i in value:
                                    if is_number(i[0]):
                                        ss1.append(float(i[0]))
                                y_limit.append(len(value))
                                vals.append(ss1)
                                colors = self.colar_list[keys_only.index(key)]
                                colors_list.append(colors)
                                show_i_1.append(key)
                    n, bins, _patches = self.ax1.hist(vals, bins=bins, color=colors_list, histtype="stepfilled", alpha=0.8)
                    n_max = max(bins)
                    n_min = min(bins)
                    n_i = []
                    for i_n in n:
                        n_i.append(max(i_n))
                    y_max = max(n_i)
                    max_, min_ = y_max * 1.2, 0
                    if y_max <= 0:
                        y_max = 1
                    max_, min_ = int(y_max) * 1.2, 0
                
                if None not in upper_limit and None not in lower_limit:
                    spec_limit = (upper_limit[0] - lower_limit[0])
                    low_ = float(lower_limit[0])
                    uper_ = float(upper_limit[0])
                    self.ax1.plot([low_, low_], [0, max_], c="red", lw=2, ls="-")
                    self.ax1.plot([uper_, uper_], [0, max_], c="red", lw=2, ls="-")
                    plt.xlim(low_ - spec_limit * 0.02, uper_ + spec_limit * 0.02)
                elif None in upper_limit and None not in lower_limit:
                    spec_limit = (n_max * 1.3 - lower_limit[0])
                    low_ = float(lower_limit[0])
                    self.ax1.plot([low_, low_], [0, max_], c="red", lw=2, ls="-")
                    plt.xlim(lower_limit[0] - spec_limit * 0.02, n_max * 1.1 + spec_limit * 0.02)
                elif None not in upper_limit and None in lower_limit:
                    spec_limit = (upper_limit[0] - n_min * 0.7)
                    uper_ = float(upper_limit[0])
                    self.ax1.plot([uper_, uper_], [0, max_], c="red", lw=2, ls="-")
                    plt.xlim(n_min * 0.9 - spec_limit * 0.02, upper_limit[0] + spec_limit * 0.02)
                else:
                    spec_limit = (n_max * 1.3 - n_min * 0.7)
                    plt.xlim(n_min * 0.9 - spec_limit * 0.02, n_max * 1.1 + spec_limit * 0.02)
                plt.ylabel('Count')  # 绘制y轴
                self.ax1.grid(axis="y")
                fig.set_facecolor('#FFFFFF')
                self.ax1.patch.set_facecolor("w")
                plt.ylim(0, y_max * 1.2)
                ax4 = fig.add_subplot(gs[10:90, 1:11])
                # ax_list.append(ax4)
                ax4.tick_params(bottom=False, top=False, left=False, right=False)
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                test_only2 = [i for i in test_only1 if is_number(i)]
                sigm, avg = round(np.std(test_only2), 2), round(np.mean(test_only2), 2)
                cp, cpu, cpk = self.test_only1_cpk(sigm, avg, upper_limit[0], lower_limit[0])
                fonn = 10
                ax4.text(1, 95, f"Data Count: {len(test_only1)}", fontsize=fonn)
                ax4.text(1, 87, f"Max: {round(max(test_only2), 2)}", fontsize=fonn)
                ax4.text(1, 79, f"Min: {round(min(test_only2), 2)}", fontsize=fonn)
                ax4.text(1, 72, f"Mean: {avg}", fontsize=fonn)
                ax4.text(1, 65, f"Std. Dev.: {sigm}", fontsize=fonn)
                ax4.text(1, 58, f"Cpu: {cpu}", fontsize=fonn)
                ax4.text(1, 51, f"Cpl: {cp}", fontsize=fonn)
                ax4.text(1, 44, f"Cpk: {cpk}", fontsize=fonn)
                ax4.text(1, 37, f"Upper Limit: {upper_limit[0]}", fontsize=fonn)
                ax4.text(1, 30, f"Lower Limit: {lower_limit[0]}", fontsize=fonn)
                ax4.text(1, 23, f"NA Count: {na_date}", fontsize=fonn)
                ax4.text(1, 16, f"Failure Count: {fail_cont}  ({round(fail_cont / len(test_only1), 2)}%)",
                         fontsize=fonn)
                plt.axis("off")
                x_s = spec_limit
                if x_s < 6:
                    sim_x = 0.25
                elif 6 <= x_s <= 12:
                    sim_x = 0.5
                elif 12 < x_s < 60:
                    sim_x = 1
                else:
                    sim_x = int(x_s // 40) + 1
                x_xlim = sim_x * 4
                xminorLocator = MultipleLocator(sim_x)
                self.ax1.xaxis.set_minor_locator(xminorLocator)
                xmajorLocator = MultipleLocator(x_xlim)
                self.ax1.xaxis.set_major_locator(xmajorLocator)
                y_s = max_ - min_
                if y_s < 0:
                    y_s = 0
                sdsd = int(y_s // 20)
                if sdsd == 0:
                    ymajorLocator = MultipleLocator(1)
                    self.ax1.yaxis.set_major_locator(ymajorLocator)
                elif sdsd > 8:
                    ymajorLocator = MultipleLocator((int(sdsd // 10) + 1) * 12)
                    self.ax1.yaxis.set_major_locator(ymajorLocator)
                else:
                    ymajorLocator = MultipleLocator(sdsd + 2)
                    self.ax1.yaxis.set_major_locator(ymajorLocator)
                ax5 = fig.add_subplot(gs[8:94, 77:99])
                # ax_list.append(ax5)
                ax5.tick_params(bottom=False, top=False, left=False, right=False)
                ax5.spines['right'].set_visible(False)
                ax5.spines['top'].set_visible(False)
                ax5.spines['left'].set_visible(False)
                ax5.spines['bottom'].set_visible(False)
                plt.xticks([])
                plt.yticks([])
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                ax5.patch.set_facecolor("#FFFFFF")
                ax5.text(2, 95, byshow, fontsize=10)
                if not show_i_1:
                    show_i_1 = dict_i
                if len(show_i_1) > 15:
                    show_i_1 = show_i_1[:15]
                s = []
                for i in show_i_1:
                    s.append(i)
                show_i_1.sort()
                color_ = self.colar_list
                for i in show_i_1:
                    ax5.text(18, 90.8 - show_i_1.index(i) * 6, i, fontsize=8)
                    cc = color_[s.index(i)]
                    plt.plot([5, 14], [91.8 - show_i_1.index(i) * 6, 91.8 - show_i_1.index(i) * 6], c=cc, lw=11, solid_capstyle='round',
                             ls="-")
                    ax5.text(5, 90.8 - show_i_1.index(i) * 6, self.dict_summary.count(i), fontsize=8, c=text_colar_[cc])
                plt.grid(c='black', linestyle='--', lw=0.3)
                plt.grid(True)
            else:
                if ng_flot_check != "NG":
                    if test_value_total:
                        max_test_value_total = max(test_value_total)
                        min_test_value_total = min(test_value_total)
                    else:
                        max_test_value_total = None
                        min_test_value_total = None
                    plt.xlim(len_1, len(test_index) + len_2)
                    self.ax1.tick_params(labelsize=9)
                    if lpdet_list == test_index:
                        upper_limit.sort()
                        lower_limit.sort(reverse=True)
                    if None not in upper_limit and None not in lower_limit and upper_limit and lower_limit:
                        if self.enter_limit == "low_and_upper_limit:1":
                            try:
                                s_max, s_min = max(upper_limit), min(lower_limit)
                            except:
                                s_max, s_min = upper_limit[0], lower_limit[0]
                        else:
                            if max_test_value_total is not None and min_test_value_total is not None:
                                s_max = max(max(upper_limit), max_test_value_total)
                                s_min = min(min(lower_limit), min_test_value_total)
                            else:
                                s_max = max(upper_limit)
                                s_min = min(lower_limit)
                            # s_max, s_min = max(max(upper_limit), max_test_value_total), min(min(lower_limit),
                            #                                                                 min_test_value_total)
                        self.ax1.plot(x, upper_limit, c='r', lw=2, ls='-', markersize=8)
                        self.ax1.plot(x, lower_limit, c='r', lw=2, ls='-', markersize=8)

                    else:
                        if None in upper_limit:
                            self.ax1.plot(x, upper_limit, c='r', lw=2, ls='-', markersize=8, marker='.',
                                          markeredgewidth=1)
                        else:
                            self.ax1.plot(x, upper_limit, c='r', lw=2, ls='-', markersize=8)
                        if None in lower_limit:
                            self.ax1.plot(x, lower_limit, c='r', lw=2, ls='-', markersize=8, marker='.',
                                          markeredgewidth=1)
                        else:
                            self.ax1.plot(x, lower_limit, c='r', lw=2, ls='-', markersize=8)
                        s_max_ = [i for i in [max_upper_limit, max_test_value_total, min_test_value_total,
                                              min_lower_limit] if is_number(i)]
                        s_min_ = [i for i in [max_upper_limit, max_test_value_total,
                                              min_test_value_total, min_lower_limit] if is_number(i)]
                        if not s_max_:
                            s_max_ = [0]
                        if not s_min_:
                            s_min_ = [0]
                        delta_u = max(s_max_) - min(s_min_)
                        s_max, s_min = max(s_max_), min(s_min_)
                        if not is_number(max_upper_limit) and not is_number(min_lower_limit):
                            s_max = max(s_max_) + 2 * delta_u
                            s_min = min(s_min_) - 2 * delta_u
                        else:
                            if not is_number(max_upper_limit):
                                s_max = max(s_max_) + 0.2 * delta_u
                            if not is_number(min_lower_limit):
                                s_min = min(s_min_) - 0.2 * delta_u
                    ss_max.append(s_max)
                    ss_min.append(s_min)
                    ss_max = [i for i in ss_max if is_number(i)]
                    ss_min = [i for i in ss_min if is_number(i)]
                    yy_max = max(ss_max)
                    yy_min = min(ss_min)
                    yy_spec = yy_max - yy_min
                    if yy_max + yy_spec * 0.2 > yy_min - yy_spec * 0.1:
                        plt.ylim(yy_min - yy_spec * 0.15, yy_max + yy_spec * 0.15)
                        ylimt_up.append(yy_max + yy_spec * 0.1)
                        ylimt_down.append(yy_min - yy_spec * 0.1)
                    elif yy_max + yy_spec * 0.2 < yy_min - yy_spec * 0.1:
                        plt.ylim(yy_max + yy_spec * 0.15, yy_min - yy_spec * 0.15)
                        ylimt_up.append(yy_min - yy_spec * 0.1)
                        ylimt_down.append(yy_max + yy_spec * 0.1)
                    else:
                        plt.ylim(yy_max + 5, yy_min - 5)
                        ylimt_up.append(yy_max + 5)
                        ylimt_down.append(yy_min - 5)

                    if upper_limit and lower_limit:
                        y_s = max(ss_max) - min(ss_min)
                    else:
                        upper_limit22 = [float(i) for i in upper_limit if i]
                        lower_limit22 = [float(i) for i in lower_limit if i]
                        if upper_limit22 and lower_limit22:
                            y_s = max(max_, max(upper_limit22)) - min(min_, min(lower_limit22))
                        elif not upper_limit22 and not lower_limit22:
                            y_s = yy_max - yy_min
                        elif not upper_limit22 and lower_limit22:
                            y_s = max_ - min(min_, min(lower_limit22))
                        elif upper_limit22 and not lower_limit22:
                            y_s = max(max_, max(upper_limit22)) - min_
                        else:
                            y_s = 0
                else:
                    plt.ylim(0, 3)
                    plt.xlim(0, 5)
                    y_s = 5
                if ylimt_up:
                    y_s = max(ylimt_up) - min(ylimt_down)
                if y_s <= 0.5:
                    y_1 = round(y_s / 5, 2)
                elif 0.5 < y_s <= 1:
                    y_1 = round(y_s / 5, 1)
                elif 1 < y_s <= 3:
                    y_1 = 0.25
                elif 3 < y_s <= 5:
                    y_1 = 0.5
                elif 5 < y_s < 10:
                    y_1 = 1
                elif 10 <= y_s <= 20:
                    y_1 = 2.5
                elif 20 < y_s <= 25:
                    y_1 = 5
                else:
                    y_1 = int(int(y_s / 5) // 5) * 5
                y_2 = y_1 / 5
                try:
                    ymajorLocator = MultipleLocator(y_1)
                    yymajorLocator = MultipleLocator(y_2)
                    self.ax1.yaxis.set_minor_locator(yymajorLocator)
                    self.ax1.yaxis.set_major_locator(ymajorLocator)
                except:
                    yys = float('{:.3f}'.format(y_s / 5))
                    ymajorLocator = MultipleLocator(yys)
                    yymajorLocator = MultipleLocator(yys / 5)
                    self.ax1.yaxis.set_minor_locator(yymajorLocator)
                    self.ax1.yaxis.set_major_locator(ymajorLocator)
                x_s = len(x)
                if x_s < 6:
                    sim_x = 0.25
                elif 6 <= x_s <= 20:
                    sim_x = 0.5
                elif 20 < x_s <= 80:
                    sim_x = 1
                else:
                    sim_x = int(x_s // 80)
                    if sim_x <= 5:
                        sim_x = 5
                    elif sim_x <= 10:
                        sim_x = 10
                    elif sim_x <= 15:
                        sim_x = 15
                    elif sim_x <= 20:
                        sim_x = 20
                    elif sim_x <= 25:
                        sim_x = 25
                    elif sim_x <= 30:
                        sim_x = 30
                    elif sim_x <= 35:
                        sim_x = 35
                    elif sim_x <= 40:
                        sim_x = 40
                    elif sim_x <= 45:
                        sim_x = 45
                    else:
                        sim_x = 50
                x_xlim = sim_x * 5
                # 使用 MultipleLocator(sim_x) 让副刻度按主刻度等分比例分布
                # （主刻度间隔为 x_xlim=sim_x*5，副刻度间隔为 sim_x，即主刻度 5 等分）
                xminorLocator = MultipleLocator(sim_x)
                self.ax1.xaxis.set_minor_locator(xminorLocator)
                # 参考 Y 轴的定义方式：直接用 MultipleLocator 设置主刻度定位器。
                # 这样：
                # 1. 主刻度从 0 开始（X 列表的第一个值会显示出来）
                # 2. 主网格线的位置和显示的刻度值位置一致
                # MultipleLocator(x_xlim) 默认从 0 开始生成刻度：0, x_xlim, 2*x_xlim, ...
                xmajorLocator = MultipleLocator(x_xlim)
                self.ax1.xaxis.set_major_locator(xmajorLocator)
                # 保证主刻度从 0 开始（X 列表第一个值），且与主网格线位置严格一致
                x_min_val, x_max_val = 0.0, float(len(x))
                x_x = xmajorLocator.tick_values(x_min_val, x_max_val)
                # 转为 Python int 列表
                x_x = [int(round(v)) for v in x_x]
                plt.xticks(x_x)
                plt.subplots_adjust(left=0, right=1, top=0.995, bottom=0)
                ax11 = fig.add_subplot(gs[99:, :])
                # ax_list.append(ax11)
                ax11.axis("on")
                ax11.spines['right'].set_visible(False)
                ax11.spines['top'].set_visible(False)
                ax11.spines['left'].set_visible(False)
                ax11.spines['bottom'].set_visible(False)
                self.ax2 = fig.add_subplot(gs[59:98, 2:78])
                # ax_list.append(self.ax2)
                # self.ax2.tick_params(bottom=False, top=False, left=False, right=False)
                # self.ax2.spines['right'].set_visible(False)
                # self.ax2.spines['top'].set_visible(False)
                # self.ax2.spines['left'].set_visible(False)
                # self.ax2.spines['bottom'].set_visible(False)
                plt.axis("off")
                plt.xticks([])
                plt.yticks([])
                print_1 = "#   | Test Name" + " " * 245 + "| Average" + " " * 13 + "| Std. Est." + " " * 11 + "| Cpk"
                self.ax2.text(1, 92, "-" * 313, fontsize=7, c="grey")
                self.ax2.text(1, 95, print_1, fontsize=7, c="black")
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                axright = fig.add_subplot(gs[0:99, 76:100])
                plt.xticks([])
                plt.yticks([])
                plt.grid(True)
                axright.tick_params(bottom=False, top=False, left=False, right=False)
                axright.spines['right'].set_visible(False)
                axright.spines['top'].set_visible(False)
                axright.spines['left'].set_visible(False)
                axright.spines['bottom'].set_visible(False)
                axright.patch.set_facecolor("#D2D3D3")
                ax5 = fig.add_subplot(gs[1:55, 77:99])
                # ax_list.append(ax5)
                ax5.tick_params(bottom=False, top=False, left=False, right=False)
                ax5.spines['right'].set_visible(False)
                ax5.spines['top'].set_visible(False)
                ax5.spines['left'].set_visible(False)
                ax5.spines['bottom'].set_visible(False)
                plt.xticks([])
                plt.yticks([])
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                ax5.patch.set_facecolor("#FFFFFF")
                ax5.text(2, 95, byshow, fontsize=10)
                if len(dict_i) > 20:
                    dict_i = dict_i[:20]
                s = []
                for i in dict_i:
                    s.append(i)
                dict_i.sort()
                color_ = self.colar_list
                for i in dict_i:
                    ax5.text(18, 90.8 - dict_i.index(i) * 4.5, i, fontsize=8)
                    cc = color_[s.index(i)]
                    # solid_capstyle='round' 让横线两端呈半圆形（rsplotlib 已支持：'butt' | 'round' | 'projecting'）
                    plt.plot([5, 14], [91.8 - dict_i.index(i) * 4.5, 91.8 - dict_i.index(i) * 4.5], c=cc, lw=11, solid_capstyle='round',
                             ls="-")
                    ax5.text(5, 90.8 - dict_i.index(i) * 4.5, self.dict_summary.count(i), fontsize=8, c=text_colar_[cc])
                # plt.grid(c='black', linestyle='--', lw=0.3)
                plt.grid(True)
                if lpdet_list == test_index:
                    test_name_list.sort(reverse=True)
                ax9 = fig.add_subplot(gs[70:, 77:])
                # ax_list.append(ax9)
                plt.axis("off")
                # ax9.patch.set_facecolor("#D2D3D3")
                plt.xticks([])
                plt.yticks([])
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                try:
                    test_tine_put = f"{round(sum(test_time_colm) / len(test_time_colm), 2)}(s)"
                except:
                    test_tine_put = "No PASS"
                self.station_name = station
                if (self.station_name == "S-COND" or self.station_name == "SCOND") and "_" in self.version:
                    self.version = self.version.split("_")[0]
                elif (self.station_name == "A-COND" or self.station_name == "ACOND") and "_" in self.version:
                    self.version = self.version.split("_")[0]
                elif " " in self.version.lower():
                    self.version = self.version.split(" ")[0]
                self.version = self.version.replace("\n", "")
                ax9.text(1, 100, f"   Station Name:  {self.station_name}", fontsize=10, c="k")
                ax9.text(1, 90, f"WiPAS Version:  {self.version}", fontsize=10, c="k")
                if self.if_testtime == "test_time:1":
                    ax9.text(1, 80, f"         Test Time:  {test_tine_put}", fontsize=10, c="k")
                    ax9.text(1, 70, f"       Test Count:  {len(self.test_date_list)}", fontsize=10, c="k")
                    if apple_pass_colm:
                        ax9.text(1, 50, f"Apple Pass Count:  {len(apple_pass_colm)}", fontsize=10, c="k")
                        ax9.text(1, 60, f"      Pass Count:  {len(test_time_colm) - len(apple_pass_colm)}", fontsize=10,
                                 c="k")
                    else:
                        ax9.text(1, 60, f"      Pass Count:  {len(test_time_colm)}", fontsize=10, c="k")
                else:
                    ax9.text(1, 80, f"       Test Count:  {len(self.test_date_list)}", fontsize=10, c="k")
                    if apple_pass_colm:
                        ax9.text(1, 60, f"Apple Pass Count:  {len(apple_pass_colm)}", fontsize=10, c="k")
                        ax9.text(1, 70, f"      Pass Count:  {len(test_time_colm) - len(apple_pass_colm)}", fontsize=10,
                                 c="k")
                    else:
                        ax9.text(1, 70, f"      Pass Count:  {len(test_time_colm)}", fontsize=10, c="k")
            if len(x) != 1:
                for i in test_name_list:
                    test_name_list_index = test_name_list.index(i)
                    test_iterm = i.lower()
                    if test_name_list_index >= 9:
                        ll = 2
                    else:
                        ll = 4
                    limit_spec = (max(ss_max) - min(ss_min)) * 0.1
                    mark_cpk1, mark_cpk2 = min(ss_min) - limit_spec, max(ss_max) + limit_spec
                    test_cpk_2 = self.test_cpk(i, "index_s")
                    cpk_ = test_cpk_2[2]
                    if str(cpk_) != "NA" and self.mark == "marker:1" and "E" not in str(cpk_):
                        if 0 < cpk_ < 1.33:
                            if "sens" not in test_iterm and "per" not in test_iterm and "ber" not in test_iterm and "margindb" not in test_iterm:
                                self.ax1.plot(
                                    [test_name_list_index + 1, test_name_list_index + 1],
                                    [mark_cpk1, mark_cpk2], c="yellow",
                                    lw=14, ls='-', alpha=0.4)
                    if test_name_list_index < 14:
                        try:
                            if 0 >= float(test_cpk_2[2]) or float(test_cpk_2[2]) >= 1.33:
                                color = "black"
                            else:
                                if "sens" in test_iterm or "per" in test_iterm or "ber" in test_iterm or "margindb" in test_iterm:
                                    color = "black"
                                else:
                                    color = "red"
                        except:
                            color = "black"
                        font_size = 8
                        print_i = str(test_name_list.index(i) + 1) + ". " + " " * int(ll) + str(i)
                        self.ax2.text(1, 87.5 - test_name_list.index(i) * 6.4, print_i, fontsize=font_size)
                        self.ax2.text(77, 87.5 - test_name_list.index(i) * 6.4, test_cpk_2[0], fontsize=font_size)
                        self.ax2.text(85.3, 87.5 - test_name_list.index(i) * 6.4, test_cpk_2[1], fontsize=font_size)
                        self.ax2.text(92, 87.5 - test_name_list.index(i) * 6.4, test_cpk_2[2], fontsize=font_size,
                                      c=color)
                        self.ax2.text(1, 85.8 - test_name_list.index(i) * 6.4, "-" * 275, fontsize=8, c="grey")
                if len(test_name_list) <= 14:
                    for y in range(len(test_name_list), 14):
                        self.ax2.text(1, 85.8 - y * 6.4, "-" * 275, fontsize=8, c="grey")
            for thread in plot_list:
                thread.join()
            plt.savefig(os.path.join(path_p, f'{plt_idix} {test_name}.png'))
            fig.clf()
            fig.clear()
            plt.close("all")
            plt_idix += 1
        if ng_flot > 0:
            print(f" 关键词: {len(self.station_testlist)}, 失败: {ng_flot}")
        else:
            print(f" 关键词: {len(self.station_testlist)}")


if __name__ == "__main__":
    # freeze_support()
    versions = "4.0.1(20260613)"
    version(versions)
    select_chart()
    opp2 = Reportopp()
    read_by_chart = read_by_chart()
    while True:
        bychart_ = input(f" 第一步、请选择show类型, 输入对应的数字, 并回车 ({read_by_chart[1][1]})\n ")
        # bychart_ = "1"
        try:
            if 0 < int(bychart_) < 6:
                by_ = read_by_chart[0][int(bychart_)]
                print(" %s 您选择了:" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), by_)
                break
            else:
                by_ = ""
                print(" %s 输入错误, 请输入对应数字" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                continue
        except:
            if bychart_ == "":
                by_ = "Serial Number"
                print(" %s 您选择了:" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "Serial Number")
                break
            else:
                by_ = ""
                print(" %s 输入错误, 请输入对应数字" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                continue
    print(" 第二步、请拉入文件夹或 csv 文件")
    while True:
        path_s = input(" =>").rstrip()
        if "\\" in path_s:
            path_ = path_s.replace("\\", "")
        elif path_s:
            path_ = path_s
        else:
            print(" %s 无法找到文件夹及 csv 文件, 已退出！" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            sys.exit(0)
        if os.path.isdir(path_):
            i_index = 0
            path_list = os.listdir(path_)
            path_list.sort()
            for filename_i in path_list:
                if "csv" == filename_i.split(".")[-1]:
                    start_time = time.time()
                    filename = f"{path_}/{filename_i}"
                    opp2.data_read(filename)
                    opp2.read_test_station()
                    filename_ = filename.split("/")[-1].split(".")[0]
                    path2 = f"{path_}/plots/{filename_}-plots/"
                    if not os.path.exists(path2):
                        os.makedirs(path2)
                    print(" %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), filename)
                    i_index += 1
                    print(f" %s 已读取第{i_index}个csv, 站位为 {opp2.station}" % (
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    opp2.colms(path2, by_, opp2.station)
                    end_time = time.time()
                    print(end_time - start_time)
        elif os.path.isfile(path_):
            if "csv" in path_:
                start_time = time.time()
                new_path = "/".join(path_.split("/")[:-1])
                opp2.data_read(path_)
                opp2.read_test_station()
                filename_ = path_.split("/")[-1].split(".")[0]
                path2 = f"{new_path}/plots/{filename_}-plots/"
                if not os.path.exists(path2):
                    os.makedirs(path2)
                print(f" %s 已读取csv, 站位为 {opp2.station}" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                opp2.colms(path2, by_, opp2.station)
                end_time = time.time()
                print(end_time - start_time)

            else:
                print(" %s 非 csv 文件！" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        print()
        print(" 处理完成。")
