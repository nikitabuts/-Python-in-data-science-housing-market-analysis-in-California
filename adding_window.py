import tkinter as tk
import numpy as np
import pandas as pd
from tkinter import messagebox

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


class AddingWindow(tk.Toplevel):
    def __init__(self, master, window):
        super().__init__()
        self.master = master
        self.window = window
        self.classes = ['INLAND', 'NEAR BAY', '<1H OCEAN', 'NEAR OCEAN', 'ISLAND']
        self.flag_prediction1 = False
        self.flag_prediction2 = False

        self.longitude_label = tk.Label(self, width=30, text='Введите параметр longitude', anchor='w')
        self.longitude_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 0]),
                                          to=np.max(self.master.dataset.iloc[:, 0]))

        self.latitude_label = tk.Label(self, width=30, text='Введите параметр latitude', anchor='w')
        self.latitude_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 1]),
                                         to=np.max(self.master.dataset.iloc[:, 1]))

        self.housing_median_age_label = tk.Label(self, width=30, text='Введите параметр housing_median_age', anchor='w')
        self.housing_median_age_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 2]),
                                                   to=np.max(self.master.dataset.iloc[:, 2]))

        self.total_rooms_label = tk.Label(self, width=30, text='Введите параметр total_rooms', anchor='w')
        self.total_rooms_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 3]),
                                            to=np.max(self.master.dataset.iloc[:, 3]))

        self.total_bedrooms_label = tk.Label(self, width=30, text='Введите параметр total_bedrooms', anchor='w')
        self.total_bedrooms_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 4]),
                                               to=np.max(self.master.dataset.iloc[:, 4]))

        self.population_label = tk.Label(self, width=30, text='Введите параметр population', anchor='w')
        self.population_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 5]),
                                           to=np.max(self.master.dataset.iloc[:, 5]))

        self.households_label = tk.Label(self, width=30, text='Введите параметр households', anchor='w')
        self.households_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 6]),
                                           to=np.max(self.master.dataset.iloc[:, 6]))

        self.median_income_label = tk.Label(self, width=30, text='Введите параметр median_income', anchor='w')
        self.median_income_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 7]),
                                              to=np.max(self.master.dataset.iloc[:, 7]))

        self.median_house_value_label = tk.Label(self, width=30, text='Введите параметр median_house_value', anchor='w')
        self.median_house_value_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 9]),
                                                   to=np.max(self.master.dataset.iloc[:, 9]))

        self.ocean_proximity_label = tk.Label(self, width=30, text='Введите параметр ocean_proximity', anchor='w')
        self.ocean_proximity_entry = tk.Listbox(self, width=30, height=6, selectmode=tk.SINGLE)

        self.price_predict_label = tk.Label(self, width=30, text='Хотите ли вы, чтобы цена квартиры\n'
                                                                 ' прогнозировалась автоматически?(y/n)', anchor='w')
        self.price_predict_entry = tk.Spinbox(self, width=30)

        self.class_predict_label = tk.Label(self, width=30, text='Хотите ли вы, чтобы зона квартиры\n'
                                                                 ' прогнозировалась автоматически?(y/n)', anchor='w')
        self.class_predict_entry = tk.Spinbox(self)

        self.Button = tk.Button(self, text='Добавить', command=self.apply)

        self.init_child()

    def init_child(self):
        self.title('Добавить запись')
        self.geometry('460x385+100+100')
        self.resizable(False, False)

        self.longitude_label.grid(row=0, column=0)
        self.longitude_entry.grid(row=0, column=1)

        self.latitude_label.grid(row=1, column=0)
        self.latitude_entry.grid(row=1, column=1)

        self.housing_median_age_label.grid(row=2, column=0)
        self.housing_median_age_entry.grid(row=2, column=1)

        self.total_rooms_label.grid(row=3, column=0)
        self.total_rooms_entry.grid(row=3, column=1)

        self.total_bedrooms_label.grid(row=4, column=0)
        self.total_bedrooms_entry.grid(row=4, column=1)

        self.population_label.grid(row=5, column=0)
        self.population_entry.grid(row=5, column=1)

        self.households_label.grid(row=6, column=0)
        self.households_entry.grid(row=6, column=1)

        self.median_income_label.grid(row=7, column=0)
        self.median_income_entry.grid(row=7, column=1)

        self.median_house_value_label.grid(row=8, column=0)
        self.median_house_value_entry.grid(row=8, column=1)

        self.ocean_proximity_label.grid(row=9, column=0)
        for i in range(len(self.classes)):
            self.ocean_proximity_entry.insert(tk.END, self.classes[i])
        self.ocean_proximity_entry.grid(row=9, column=1)

        self.price_predict_label.grid(row=10, column=0, sticky='w')
        self.price_predict_entry.grid(row=10, column=1, sticky='w')

        self.class_predict_label.grid(row=11, column=0, sticky='w')
        self.class_predict_entry.grid(row=11, column=1,sticky='w')

        self.Button.grid(row=12, column=0, columnspan=2, ipadx=200)

        self.grab_set()
        self.focus_set()

    def apply(self):
        flag = True
        if self.price_predict_entry.get() == 'y':
            self.flag_prediction1 = True
        if self.class_predict_entry.get() == 'y':
            self.flag_prediction2 = True
        if len(self.ocean_proximity_entry.curselection()) == 0:
            flag = False
            messagebox.showinfo("Ошибка!", "Не указан параметр ocean_proximity_entry!")
        else:
            dict = {
                0: self.longitude_entry.get(),
                1: self.latitude_entry.get(),
                2: self.housing_median_age_entry.get(),
                3: self.total_bedrooms_entry.get(),
                4: self.total_bedrooms_entry.get(),
                5: self.population_entry.get(),
                6: self.households_entry.get(),
                7: self.median_income_entry.get(),
                8: self.classes[self.ocean_proximity_entry.curselection()[0]],
                9: self.median_house_value_entry.get(),
            }
            for i in [0, 1, 2, 3, 4, 5, 6, 7, 9]:
                try:
                    a = float(dict[i])
                except ValueError:
                    flag = False
                    messagebox.showinfo("Ошибка!", "Неверный тип переменной " + str(self.master.dataset.columns[i]) + "!")

            if len(self.ocean_proximity_entry.curselection()) == 0:
                flag = False
                messagebox.showinfo("Ошибка!", "Не указан параметр ocean_proximity_entry!")

            if flag:
                data_frame = self.appending(dict)
                self.master.dataset = self.master.dataset.append(data_frame, ignore_index=True)
                matrix = data_frame.values
                cells = []
                for j in range(10):
                    try:
                        cells.append(matrix[0][j])
                    except IndexError:
                        break
                tuple(cells)
                self.window.table.insert('', 'end', text=str((len(self.master.dataset.longitude) - 1)), values=cells)

    def appending(self, dict):
        ocean_proximity_array = self.master.dataset.ocean_proximity.unique()
        auxiliary_array = np.array([])
        columns_of_dataset = self.master.dataset.columns

        for iteration in np.arange(self.master.dataset.shape[1]):
            if columns_of_dataset[iteration] == 'median_house_value' and self.flag_prediction1:
                # проверка на то, что найден именно столбец median_house_value
                x = self.master.dataset.drop(['median_house_value', 'ocean_proximity'], axis=1)
                y = self.master.dataset.median_house_value
                auxiliary_sub_array = auxiliary_array[:-1].astype(float)
                weights = regression_weights(x, y)
                variable = int(predict(auxiliary_sub_array, regression_weights(x, y)))
            elif columns_of_dataset[iteration] == 'ocean_proximity' and self.flag_prediction2:
                auxiliary_sub_array = [auxiliary_array[0:2]]
                variable = classificator(self.master.dataset, auxiliary_sub_array)
            else:
                variable = dict[iteration]

            auxiliary_array = np.append(auxiliary_array, variable)

        df = pd.DataFrame(columns=self.master.dataset.columns)
        df.loc[len(self.master.dataset)] = [float(auxiliary_array[0]), float(auxiliary_array[1]),
                                            float(auxiliary_array[2]), float(auxiliary_array[3]),
                                            float(auxiliary_array[4]), float(auxiliary_array[5]),
                                            float(auxiliary_array[6]), float(auxiliary_array[7]),
                                            (auxiliary_array[8]), float(auxiliary_array[9])]
        return df


def class_balancing(dataset):
    helped_dataset = dataset.copy()
    helped_dataset.ocean_proximity, d = helped_dataset.ocean_proximity.factorize()
    series = pd.Series(helped_dataset.ocean_proximity)
    maximum = series.value_counts().max()

    x = pd.DataFrame(columns=['longitude', 'latitude', 'ocean_proximity']).values
    for i in np.arange(len(series.unique())):
        helped_2 = series.unique()[i]
        array = helped_dataset[helped_dataset['ocean_proximity'] == helped_2].loc[:,
                ['longitude', 'latitude', 'ocean_proximity']].values.reshape(-1, 3)
        help_3 = array.copy()
        while len(array) < maximum:
            if maximum - len(array) >= len(help_3):
                array = np.append(array, help_3, axis=0)
            else:
                array = np.append(array, help_3[0: maximum - len(help_3)], axis=0)
        x = np.append(x, array, axis=0)

    x = pd.DataFrame(x, columns=['longitude', 'latitude', 'ocean_proximity'])
    return x


def classificator(dataset, array):
    balanced_dataset = class_balancing(dataset)
    variable = len(balanced_dataset)

    value = dataset.ocean_proximity.unique()
    y = pd.DataFrame(balanced_dataset).iloc[:, 2].values.reshape((variable,)).astype(int)
    x = balanced_dataset.drop('ocean_proximity', axis=1).values.astype(float)

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(n_estimators=500, max_depth=20)
    clf.fit(x_train, y_train)
    prediction = clf.predict(array)
    y_pred = clf.predict(x_test)
    z = pd.Series(y).unique()
    for i in range(len(value)):
        if prediction == z[i]:
            prediction = value[i]
    return prediction


def regression_weights(x, y):
    x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
    weights = np.linalg.inv(x.T @ x) @ x.T @ y.values
    return weights


def predict(x, weights):  # Не работает - доделать
    b = weights[0]
    weights = weights[1: len(weights)]
    predictions = b + x @ weights.T
    return predictions


def score(y_test, y_pred):
    d = np.abs(y_test - y_pred)
    epsilons = list(map(float, input().split()))
    scores = np.array([])
    for i in np.arange(len(epsilons)):
        summary = 0
        for j in np.arange(len(d)):
            if d[j] <= epsilons[i]:
                summary += 1
        score = summary / len(d)
        scores = np.append(scores, score)
    scoresDF = pd.DataFrame({"Epsilon": epsilons, "Score": scores})

    return scoresDF
