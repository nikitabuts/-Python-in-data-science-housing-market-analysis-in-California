from tkinter import filedialog
import tkinter as tk
import os
import pandas as pd


class HistogramWindow(tk.Toplevel):  # Окно вывода гистограммы в приложении
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.title("Категоризированная гистограмма")
        self.ax = None

        self.bar()

        self.im = tk.PhotoImage(file='pic.png')
        self.label = tk.Label(self, image=self.im)
        self.label.pack()
        # кнопка для сохранения отчета в файл
        self.button = tk.Button(self, text='Сохранить отчёт', command=self.saving)
        self.button.pack()

        self.grab_set()
        self.focus_set()
        os.remove("pic.png")

    def bar(self):  # Функция вывода гистограммы
        df = pd.DataFrame({'lab': self.master.dataset.ocean_proximity.unique(),
                           'val': self.master.dataset.ocean_proximity.value_counts()})
        self.ax = df.plot.bar(x='lab', y='val', rot=0)
        self.ax.figure.savefig('pic.png')

    def saving(self):  # Функция сохранения гистограммы в png формате
        file = filedialog.asksaveasfilename(defaultextension=".png")
        if file:
            self.ax.figure.savefig(file)
