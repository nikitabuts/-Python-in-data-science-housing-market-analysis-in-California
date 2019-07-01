import tkinter as tk
import os
from tkinter import filedialog


class WiskerWindow(tk.Toplevel):  # Окно вывода диаграммы Бокса-Вискерса
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.title("Диаграмма Бокса-Вискерса")

        self.im = tk.PhotoImage
        self.label = tk.Label(self)
        self.plot = None

        self.parameter_label = tk.Label(self, width=30, text='Укажите параметры поиска', anchor='w')
        self.parameter_entry = tk.Listbox(self, width=30, height=9, selectmode=tk.SINGLE)
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 9]:
            self.parameter_entry.insert(tk.END, str(self.master.dataset.columns[i]))
            self.button = tk.Button(self, text='Найти', command=self.box_visc)
        self.parameter_entry.grid(row=0, column=1)
        self.button.grid(row=1, column=1)

        self.box_visc()

        self.save_button = tk.Button(self, text='Сохранить отчёт', command=self.saving)
        self.save_button.grid(row=1, column=0)

        self.grab_set()
        self.focus_set()

    def box_visc(self):  # Функция создания и отображения диаграммы
        if len(self.parameter_entry.curselection()) == 0:
            string = 'median_house_value'
        elif self.parameter_entry.curselection()[0] == 8:
            string = 'ocean_proximity'
        else:
            string = self.master.dataset.columns[self.parameter_entry.curselection()[0]]
        self.plot = self.data.assign(index=self.data.groupby('ocean_proximity').
                                cumcount()).pivot('index', 'ocean_proximity', string).plot(kind='box')
        self.plot.figure.savefig('pic.png')
        self.im = tk.PhotoImage(file='pic.png')
        self.label.config(image=self.im)
        self.label.grid(row=0, column=0)
        os.remove("pic.png")

    def saving(self):  # Функция сохранения диаграммы в новый файл в формате png
        file = filedialog.asksaveasfilename(defaultextension=".png")
        if file:
            self.plot.figure.savefig(file)
