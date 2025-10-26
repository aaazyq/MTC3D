# %%
# ! pip install pypinyin
import os

# %%
cropped_mask = os.listdir("/Users/yangzidong/Desktop/yuqing/medical/202510/code_github/cropped_mask")
images_vis = os.listdir("/Users/yangzidong/Desktop/yuqing/medical/202510/code_github/images_vis/")


# %%
for name in images_vis:
    if name not in cropped_mask:
        print(name)

# %%
import pandas as pd
file_path = "/Users/yangzidong/Desktop/yuqing/medical/202510/MTC_3D/data_CT cohort for MTC_3D-lnm.xlsx"

data = pd.read_excel(file_path)

# %%
from pypinyin import pinyin, Style

# Function to convert Chinese names to Pinyin
def name_to_pinyin(name):
    if isinstance(name, str):
        return ''.join([i[0] for i in pinyin(name, style=Style.NORMAL)])
    return name

# Convert '姓名' column to Pinyin
data['name'] = data['姓名'].apply(name_to_pinyin)

# Convert '病理分级' to binary (0/1)
data['level'] = data['病理分级'].apply(lambda x: 1 if x == '高级别' else 0)

# 多音字
data.loc[44, "name"] = "adinaadejiang"
data.loc[116, "name"] = "lichangyong"
data.loc[128, "name"] = "sunlihua"
data.loc[230, "name"] = "zengxiaoling"
data.loc[246, "name"] = "zhangzhaohui"
data.loc[282, "name"] = "zengfeng"

# Display the updated DataFrame head
display(data[['姓名', '病理分级','name', 'level']].head())

# %%
data["level"].value_counts()

# %%
for i in range(len(data)):
    name = data["name"][i]
    if name not in cropped_mask:
        print(i, name)




# %%
cleaned_data = data.loc[~data.index.isin([19, 56, 59, 86, 97, 105, 106, 120, 147, 187, 225, 226, 250, 271, 304])]
cleaned_data = cleaned_data.reset_index(drop=True)

# %%
cleaned_data["level"].value_counts()

# %%
for name in cropped_mask:
    if name not in data["name"].tolist():
        print(name)


