import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import numpy as np
import pandas as pd
from writing_file_window import new_csv


class TextWindow(tk.Toplevel):  # Окно вывода текстового отчета по выбранным индексам и атрибутам
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.search_columns = []
        self.new_dataset = pd.DataFrame
        self.begin_label = tk.Label(self, width=30, text='Укажите начальный индекс', anchor='w')
        self.begin_entry = tk.Spinbox(self, width=30)

        self.end_label = tk.Label(self, width=30, text='Укажите конечный индекс', anchor='w')
        self.end_entry = tk.Spinbox(self, width=30)

        self.parameter_label = tk.Label(self, width=30, text='Укажите параметры выборки', anchor='w')
        self.parameter_entry = tk.Listbox(self, width=30, height=5, selectmode=tk.EXTENDED)
        self.scroll = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.parameter_entry.yview)

        self.button = tk.Button(self, text='Найти', command=self.try_search)

        self.new_table = ttk.Treeview(self, height=13, selectmode=tk.EXTENDED)
        self.new_scroll = tk.Scrollbar(self, command=self.new_table.yview)
        self.new_table.config(yscrollcommand=self.new_scroll.set)
        self.save_button = tk.Button(self, command=self.save, text='Сохранить данные')
        self.reset_button = tk.Button(self, command=self.reset, text='Новая выборка')

        self.init_child()

    def init_child(self):  # Упаковка выджетов
        self.title('Текстовый отчёт')
        self.geometry('500x220+100+100')
        self.resizable(True, True)

        self.parameter_entry.config(yscrollcommand=self.scroll.set)
        for i in range(9):
            self.parameter_entry.insert(tk.END, str(self.master.dataset.columns[i]))

        self.begin_label.grid(row=0, column=0)
        self.begin_entry.grid(row=0, column=1)

        self.end_label.grid(row=1, column=0)
        self.end_entry.grid(row=1, column=1)

        self.parameter_label.grid(row=2, column=0)
        self.parameter_entry.grid(row=2, column=1)
        self.scroll.grid(row=2, column=2, sticky='w', ipady=20)

        self.button.grid(row=3, column=0, columnspan=3, ipadx='150')
        self.grab_set()
        self.focus_set()

    def try_search(self):  # Функция проверки на существование данных в БД
        self.new_dataset = self.searching()
        if not self.new_dataset.empty:
            self.search()

    def search(self):  # Функция поиска данных в БД по выбранным индексам
        self.geometry('350x400')
        self.begin_label.grid_remove()
        self.begin_entry.grid_remove()
        self.end_label.grid_remove()
        self.end_entry.grid_remove()
        self.parameter_label.grid_remove()
        self.parameter_entry.grid_remove()
        self.scroll.grid_remove()
        self.button.grid_remove()

        self.new_table.column('#0', width=100, stretch=tk.YES)
        self.new_table["columns"] = self.search_columns
        self.new_table.heading('#0', text='number')
        self.new_table.column('#0', width=100, stretch=tk.YES)
        for i in range(len(self.search_columns)):
            self.new_table.heading(str(self.search_columns[i]), text=self.search_columns[i])
            self.new_table.column(self.search_columns[i], width=100, stretch=tk.YES)
        matrix = self.new_dataset.values

        for i in range(len(matrix)):
            cells = []
            for j in range(len(self.search_columns)):
                try:
                    cells.append(matrix[i][j])
                except IndexError:
                    break
            tuple(cells)
            self.new_table.insert('', 'end', text=str(i), values=cells)

        self.new_table.grid(row=0, column=0)
        self.new_scroll.grid(row=0, column=1, sticky='w', ipady=70)
        self.save_button.grid(row=1, column=0, columnspan=2, ipadx=100)
        self.reset_button.grid(row=2, column=0, columnspan=2, ipadx=100)

    def searching(self):  # Функция вывода ошибки при неверно введенных данных
        if self.begin_entry.get() == '' or self.end_entry.get() == '' or len(self.parameter_entry.curselection()) == 0:
            messagebox.showinfo("Ошибка!", "Невведённые данные")
        else:
            try:
                start = int(self.begin_entry.get())
                finish = int(self.end_entry.get())
            except ValueError:
                messagebox.showinfo("Ошибка!", "Некорректные данные")
            else:
                if len(self.master.dataset) < start or start < 0:
                    messagebox.showinfo("Ошибка!", "Начальный индекс вне диапазона!")
                elif len(self.master.dataset) < finish:
                    messagebox.showinfo("Ошибка!", "Конечный ииндекс вне диапазона!")
                elif finish <= start:
                    messagebox.showinfo("Ошибка!", "Значение конца диапазона должно быть больше значения его начала!")
                else:
                    columns = self.master.dataset.columns
                    self.search_columns = [columns[self.parameter_entry.curselection()[i]]
                                      for i in range(len(self.parameter_entry.curselection()))]
                    array = np.array([])
                    for i in np.arange(len(self.search_columns)):
                        array = np.append(array, self.search_columns[i])

                    sub_dataset = self.master.dataset.loc[start: finish, array]
                    return sub_dataset

    def save(self):  # Функция сохранения текстового отчета в файл
        new_csv(self.master, self.new_dataset, '')

    def reset(self):  # Функция для ввода данных для нового отчета
        self.new_table.delete(*self.new_table.get_children())
        self.new_table.grid_remove()
        self.new_scroll.grid_remove()
        self.save_button.grid_remove()
        self.reset_button.grid_remove()

        self.begin_label.grid()
        self.begin_entry.grid()
        self.end_label.grid()
        self.end_entry.grid()
        self.parameter_label.grid()
        self.parameter_entry.grid()
        self.scroll.grid()
        self.button.grid()
