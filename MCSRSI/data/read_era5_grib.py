import pygrib as pg
from datetime import datetime

# 需要读取什么变量
variable_list = ["2 metre dewpoint temperature", "2 metre temperature", "Surface pressure", "Total precipitation",
    "Skin temperature", "100 metre U wind component", "100 metre V wind component", "Clear-sky direct solar radiation at surface",
    "Downward UV radiation at the surface", "Surface net solar radiation", "Surface solar radiation downwards",
    "Low cloud cover", "Total cloud cover"]

def read_grib(file_path):
    grb = pg.open(file_path)

    # 以字典为例读取grib数据,字典初始化
    # source_dict = {}
    # for variable in variable_list:
    #     source_dict[variable] = []
    # index = 0
    
    print(grb)

    variable_name_list = []

    for variable in grb:
        grb_date = variable.validityDate
        grb_time = variable.validityTime
        grb_datetime = datetime(grb_date // 10000, (grb_date % 10000) // 100, grb_date % 100, grb_time // 100,grb_time % 100)
        
        # 查看每个变量所有可以读取的内容
        # print(variable.keys())

        # 读取变量的值
        # source_dict[variable_list[index % len(variable_list)]].append(variable.values)
        # index += 1

        if variable.name in variable_name_list:
            break
        variable_name_list.append(variable.name)
    
    print(variable_name_list)
    
read_grib('./Data/20250709.grib')