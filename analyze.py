import pandas as pd
import os
from shutil import copytree

sa_path = "./eval-html-WN/0/log.csv"
data_path = "./eval-html-WN/0/imgs"
output_path = "./eval-html-WN/0/collect"
if not os.path.exists(output_path):
    os.mkdir(output_path)
# sn_path = "./eval-html-SN/0/log.csv"
# wa_path = "./eval-html-WA/0/log.csv"
# wn_path = "./eval-html-WN/0/log.csv"

df = pd.read_csv(sa_path)


def collect(se: pd.Series, src: str, out: str):
    if not os.path.exists(out):
        os.mkdir(out)

    se.to_csv(os.path.join(out, "metrics.csv"))

    for index, row in zip(se.index, se):
        copytree(os.path.join(src, str(index)), os.path.join(out, str(index)))


collect(
    df.sort_values(by="D", ascending=False).D[:5],
    data_path,
    os.path.join(output_path, "best_D"),
)
collect(
    df.sort_values(by="L2", ascending=False).L2[:5],
    data_path,
    os.path.join(output_path, "best_L2"),
)
collect(
    df.sort_values(by="D").D[:5],
    data_path,
    os.path.join(output_path, "least_D"),
)
collect(
    df.sort_values(by="L2").L2[:5],
    data_path,
    os.path.join(output_path, "least_L2"),
)
