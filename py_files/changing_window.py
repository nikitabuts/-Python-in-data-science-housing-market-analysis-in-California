import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from tkinter import messagebox


class ChangingWindow(tk.Toplevel):
    def __init__(self, master, window):
        super().__init__()

        self.master = master
        self.window = window

        self.focused = self.window.table.selection()[0]

        self.parameter_label = tk.Label(self, width=30, text='Укажите изменяемый параметр', anchor='w')
        self.parameter_entry = tk.Listbox(self, width=30, height=5, selectmode=tk.SINGLE)
        self.scroll = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.parameter_entry.yview)

        self.value_label = tk.Label(self, width=30, text='Укажите значение параметра', anchor='w')
        self.value_entry = tk.Spinbox(self, width=30)

        self.button = tk.Button(self, text='Изменить', command=self.change)

        self.init_child()

    def init_child(self):
        self.title('Изменить запись')
        self.geometry('600x320+100+100')
        self.resizable(True, True)

        self.parameter_entry.config(yscrollcommand=self.scroll.set)
        for i in range(9):
            self.parameter_entry.insert(tk.END, str(self.master.dataset.columns[i]))

        self.parameter_label.grid(row=0, column=0)
        self.parameter_entry.grid(row=0, column=1)
        self.scroll.grid(row=0, column=2, sticky='w', ipady=20)

        self.value_label.grid(row=1, column=0)
        self.value_entry.grid(row=1, column=1)

        self.button.grid(row=2, column=0, columnspan=3, ipadx='100')

        self.grab_set()
        self.focus_set()

    def change(self):
        columns = self.master.dataset.columns
        ocean_proximity_array = self.master.dataset.ocean_proximity.unique()
        number = (int(self.focused[1:], 16) - 1)
        column = columns[int(self.parameter_entry.curselection()[0])]
        if column == 'ocean_proximity':
            try:
                var = float(self.parameter_entry.get())
            except ValueError:
                var = self.value_entry.get()
                if var in ocean_proximity_array:
                    self.master.dataset.loc[number, column] = self.parameter_entry.get()
                    self.window.table.set(self.focused, column, var)
                else:
                    messagebox.showinfo("Ошибка!", "Должна быть введена строка из перечня: {}"
                                        .format(ocean_proximity_array))
        else:
            try:
                var = float(self.value_entry.get())
                self.master.dataset.loc[number, column] = var
                self.window.table.set(self.focused, column, var)
            except ValueError:
                messagebox.showinfo("Ошибка!", "Должно быть введено вещественное число")
