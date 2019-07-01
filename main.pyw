import tkinter as tk
from tkinter import filedialog
from main_window import DataFrame, FunctionFrame

import pandas as pd


def root_create():
    filename = filedialog.askopenfilename(defaultextension=".csv")
    if len(filename) != 0:
        anc.destroy()
        root = tk.Tk()
        root.filename = filename
        root.dataset = pd.read_csv(filename)
        data_frame = DataFrame(root)
        data_frame.grid(row=0, column=0)
        functions_frame = FunctionFrame(root, data_frame)
        functions_frame.grid(row=0, column=1, sticky='nw')
        root.mainloop()


anc = tk.Tk()
anc.title('Добро пожаловать')
button = tk.Button(anc, height=20, width=50, command=root_create, text='Открыть базу данных')
button.pack()
anc.mainloop()
