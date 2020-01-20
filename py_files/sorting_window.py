import tkinter as tk
from tkinter import messagebox


class SortingWindow(tk.Toplevel):  # Окно сортировки данных по убыванию/возрастанию
    def __init__(self, master, window):
        super().__init__()

        self.master = master
        self.window = window

        self.array = []

        self.parameter_label = tk.Label(self, width=30, text='Укажите параметр сортировки', anchor='w')
        self.parameter_entry = tk.Listbox(self, width=30, height=5, selectmode=tk.SINGLE)

        self.scroll = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.parameter_entry.yview)

        self.ascending_button = tk.Button(self, text='Сортировать по возрастанию', command=self.sort_ascending)
        self.descending_button = tk.Button(self, text='Сортировать по убыванию', command=self.sort_descending)

        self.init_child()

    def init_child(self):  # Упаковка виджетов
        self.title('Отсортировать записи')
        self.geometry('440x220+100+100')
        self.resizable(True, True)

        self.parameter_entry.config(yscrollcommand=self.scroll.set)
        for i in range(10):
            self.parameter_entry.insert(tk.END, str(self.master.dataset.columns[i]))

        self.parameter_label.grid(row=0, column=0)
        self.parameter_entry.grid(row=0, column=1)
        self.scroll.grid(row=0, column=2, sticky='w', ipady=20)

        self.ascending_button.grid(row=1, column=0, columnspan=3, ipadx=100)
        self.descending_button.grid(row=2, column=0, columnspan=3, ipadx=100)

        self.grab_set()
        self.focus_set()

    def sort_descending(self):  # Функция сортировки по убыванию
        if len(self.parameter_entry.curselection()) == 0:
            messagebox.showinfo("Ошибка!", "Выберите параметр сортировки!")
        else:
            columns = self.master.dataset.columns
            column = columns[int(self.parameter_entry.curselection()[0])]

            self.master.dataset.sort_values(column, inplace=True, ascending=False)
            self.array = self.master.dataset.values

            self.update_window()

    def sort_ascending(self):  # Функция сортировки по возрастанию
        if len(self.parameter_entry.curselection()) == 0:
            messagebox.showinfo("Ошибка!", "Выберите параметр сортировки!")
        else:
            columns = self.master.dataset.columns
            column = columns[int(self.parameter_entry.curselection()[0])]

            self.master.dataset.sort_values(column, inplace=True, ascending=True)
            self.array = self.master.dataset.values

            self.update_window()

    def update_window(self):  # Вывод отсортированных данных в таблицу в приложении
        self.window.table.delete(*self.window.table.get_children())
        matrix = self.master.dataset.values
        for i in range(len(self.master.dataset.longitude)):
            cells = []
            for j in range(10):
                try:
                    cells.append(matrix[i][j])
                except IndexError:
                    break
            tuple(cells)
            self.window.table.insert('', 'end', text=str(i), values=cells)
