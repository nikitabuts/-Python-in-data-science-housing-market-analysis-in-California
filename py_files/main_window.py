import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

from adding_window import AddingWindow
from delete_func import direct_deleting
from searching_window import SearchingWindow
from sorting_window import SortingWindow
from changing_window import ChangingWindow
from writing_file_window import new_csv
from static_report_window import StaticWindow
from text_report_window import TextWindow
from histogramm_window import HistogramWindow
from diagram_window import WiskerWindow
from grad_window import GradationWindow


class DataFrame(tk.Frame):  # Вывод БД в приложении
    def __init__(self, master=None):
        super().__init__(master)

        self.master = master
        self.master.title('Дома Калифорнии')
        self.master.geometry('1340x650+0+0')

        self.titles = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                       "population", "households", "median_income", "ocean_proximity", "median_house_value"]

        self.table = ttk.Treeview(self, height=30, selectmode=tk.EXTENDED)
        self.scrollbar = tk.Scrollbar(self)
        self.matrix = self.master.dataset.values
        self.design()

    def design(self):
        self.table["columns"] = ("longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                                 "population", "households", "median_income", "ocean_proximity", "median_house_value")
        self.table.heading('#0', text='number')
        self.table.column('#0', width=100, stretch=tk.YES)
        for i in range(10):
            self.table.heading(str(self.titles[i]), text=self.titles[i])
            self.table.column(self.titles[i], width=100, stretch=tk.YES)

        for i in range(len(self.master.dataset.longitude)):
            cells = []
            for j in range(len(self.titles)):
                try:
                    cells.append(self.matrix[i][j])
                except IndexError:
                    break
            tuple(cells)
            self.table.insert('', 'end', text=str(i), values=cells)

        self.table.grid(row=0, column=0)
        self.table.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.table.yview)
        self.scrollbar.grid(row=0, column=1, ipady=300)


class FunctionFrame(tk.Frame):  # Все функции приложения
    def __init__(self, master, prev):
        super().__init__(master)

        self.master = master
        self.prev = prev

        self.label_func = tk.Label(self, width=30, height=3, text='Функции')

        self.add_button = tk.Button(self, width=30, height=2, text=u"Добавить запись")
        self.delete_button = tk.Button(self, width=30, height=2, text=u"Удалить запись")
        self.search_button = tk.Button(self, width=30, height=2, text=u"Искать записи")
        self.sort_button = tk.Button(self, width=30, height=2, text=u"Сортировать записи")
        self.change_button = tk.Button(self, width=30, height=2, text=u"Изменить запись")
        self.save_button = tk.Button(self, width=30, height=2, text=u"Обновить бвзу данных")
        self.save_as_button = tk.Button(self, width=30, height=2, text=u"Сохранить бвзу данных")

        self.label_reports = tk.Label(self, width=30, height=3, text='Отчёты')
        self.text_report_button = tk.Button(self, height=2, width=30, text=u"Простой текстовый отчет")
        self.static_report_button = tk.Button(self, height=2, width=30, text=u"Текстовый статистический отчет")
        self.hist_button = tk.Button(self, width=30, height=2, text=u"Категоризированная гистограмма")
        self.whisker_button = tk.Button(self, width=30, height=2, text=u"Диаграмма Бокса-Вискера")
        self.grad_button = tk.Button(self, width=30, height=2, text=u"Диаграмма рассеивания")

        self.design()

    def design(self):
        self.label_func.pack()

        self.add_button.config(command=self.adding)
        self.add_button.pack()

        self.change_button.config(command=self.changing)
        self.change_button.pack()

        self.delete_button.config(command=self.deleting)
        self.delete_button.pack()

        self.search_button.config(command=self.searching)
        self.search_button.pack()

        self.sort_button.config(command=self.sorting)
        self.sort_button.pack()

        self.save_button.config(command=self.saving)
        self.save_button.pack()

        self.save_as_button.config(command=self.saving_as)
        self.save_as_button.pack()

        self.label_reports.pack()

        self.text_report_button.config(command=self.text_report)
        self.text_report_button.pack()

        self.static_report_button.config(command=self.static_report)
        self.static_report_button.pack()

        self.hist_button.config(command=self.hist_report)
        self.hist_button.pack()

        self.whisker_button.config(command=self.whisker_report)
        self.whisker_button.pack()

        self.grad_button.config(command=self.grad_report)
        self.grad_button.pack()

    def adding(self):  # Функция добавления дома
        AddingWindow(self.master, self.prev)

    def deleting(self):  # Функция удаления дома
        if len(self.prev.table.selection()) == 0:
            messagebox.showinfo("Ошибка!", "Не указаны строки под удаление")
        else:
            direct_deleting(self.master, self.prev)

    def searching(self):  # Функция поиска домов по индексу
        SearchingWindow(self.master)

    def sorting(self):  # Сортировка по возрастанию/убыванию выбранных атрибутов
        SortingWindow(self.master, self.prev)

    def changing(self):  # Функция изменения выбранного дома
        if len(self.prev.table.selection()) == 0:
            messagebox.showinfo("Ошибка!", "Не указана строка под изменение")
        else:
            ChangingWindow(self.master, self.prev)

    def saving(self):  # Перезапись открытого файла
        new_csv(self.master, self.master.dataset, self.master.filename)

    def saving_as(self):  # Сохранение БД в новом файле
        new_csv(self.master, self.master.dataset, '')

    def text_report(self):  # Простой текстовый отчет по выбранным индексам и атрибутам
        TextWindow(self.master)

    def static_report(self):  # Статистический отчет по выбранным атрибутам
        StaticWindow(self.master)

    def hist_report(self):  # Категоризированная гистограмма
        HistogramWindow(self.master.dataset)

    def whisker_report(self):  # Диаграмма Бокса-Вискера
        WiskerWindow(self.master.dataset)

    def grad_report(self):  # Диаграмма рассеивания
        GradationWindow(self.master)
