import pathlib
cur_path = pathlib.Path(__file__).parent.absolute()
print(cur_path)

cur_2 = pathlib.Path().absolute()
print(cur_2)

par_dir = cur_2.parent.absolute()
print(par_dir)
