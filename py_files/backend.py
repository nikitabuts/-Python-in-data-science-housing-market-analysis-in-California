import numpy as np
import pandas as pd

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import GridSearchCV


def make_DataFrame(dataset):  # Создает DataFrame, состоящий из 1 строки
    ocean_proximity_array = dataset.ocean_proximity.unique()
    auxiliary_array = np.array([])
    columns_of_dataset = dataset.columns
    operations = ('y', 'n')
    dictionary = {'longitude': 'числом',
                  'latitude': 'числом',
                  'housing_median_age': 'числом',
                  'total_rooms': 'числом',
                  'total_bedrooms': 'числом',
                  'population': 'числом',
                  'households': 'числом',
                  'median_income': 'числом',
                  'median_house_value': 'числом',
                  'ocean_proximity': 'строкой'}

    while True:
        print("Хотите ли вы, чтобы цена квартиры прогнозировалась автоматически? y / n")
        operation = str(input())
        if operation in operations:
            if (operation == 'y'):
                flag_prediction1 = True
            else:
                flag_prediction1 = False
            break
        else:
            continue
    while True:
        print("Хотите ли вы, чтобы зона квартиры прогнозировалась автоматически? y / n")
        operation = str(input())
        if operation in operations:
            if (operation == 'y'):
                flag_prediction2 = True
            else:
                flag_prediction2 = False
            break
        else:
            continue
    for iteration in np.arange(dataset.shape[1]):  # пока не заполнены все поля строки данных - выполняем:
        type_of_prev_string_of_data = type(
            dataset.iloc[-1, iteration])  # тип переменной i - ого столбца последней строки
        while True:  # бесконечный цикл
            if type(dataset.iloc[:, iteration][0]) != str and dataset.iloc[:,
                                                              iteration].mean() > 100000 and flag_prediction1:
                # проверка на то, что найден именно столбец median_house_value
                X = dataset.drop(['median_house_value', 'ocean_proximity'], axis=1)
                y = dataset.median_house_value
                auxiliary_sub_array = auxiliary_array[:-1].astype(float)
                weights = regression_weights(X, y)
                variable = int(predict((auxiliary_sub_array), regression_weights(X, y)))
                break
            else:
                if columns_of_dataset[iteration] == 'ocean_proximity' and flag_prediction2:
                    auxiliary_sub_array = [auxiliary_array[0:2]]
                    variable = classificator(dataset, auxiliary_sub_array)
                    break
                else:
                    print("Введите значение для столбца {}, оно должно быть {}".format(columns_of_dataset[iteration],
                                                                                       dictionary[columns_of_dataset[
                                                                                           iteration]]))
                    try:
                        variable = type_of_prev_string_of_data(input())
                    except ValueError:  # если переменная другого типа - вводим значение снова
                        print("Переменная не соответствует типу столбца, введите новое значение:")
                        continue
                    else:  # если значение переменной совпадает с нужным - завершаем цикл
                        if type_of_prev_string_of_data is str:
                            # если введенное пользователем значение существует - завершаем цикл
                            if variable in ocean_proximity_array:
                                break
                            else:
                                print("Введите значение из списка предложенных: {}".format(ocean_proximity_array))
                                continue
                        break
        # после обработки ошибки ввода добавляем значение в массив
        auxiliary_array = np.append(auxiliary_array, variable)

    df = pd.DataFrame(columns=dataset.columns)
    df.loc[len(dataset)] = [float(auxiliary_array[0]), float(auxiliary_array[1]), float(auxiliary_array[2]),
                            float(auxiliary_array[3]),
                            float(auxiliary_array[4]), float(auxiliary_array[5]), float(auxiliary_array[6]),
                            float(auxiliary_array[7]), auxiliary_array[8], float(auxiliary_array[9])]
    return df


def appending(dataset):
    data_frame = make_DataFrame(dataset)
    dataset = dataset.append(data_frame, ignore_index=True)
    return dataset


def classificator(dataset, array):

    y = dataset.ocean_proximity
    z, value = y.factorize()
    X = dataset.loc[:, ['longitude', 'latitude']].values
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(n_estimators=100, max_depth=30)
    clf.fit(X_train, y_train)

    prediction = clf.predict(array)
    for i in range(len(value)):
        if prediction in z:
            prediction = value[i]
    return prediction[0]


def regression_weights(X, y):
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    weights = np.linalg.inv(X.T @ X) @ X.T @ y.values
    return weights;


def predict(X, weights):  # Не работает - доделать
    B = weights[0]
    weights = weights[1: len(weights)]
    predictions = B + X @ weights.T
    return predictions;


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


# Поправка: в коде нет провреки на существование номеров столбцов, которые хочет удалить пользователь
def deleting_by_numbers(dataset):
    # вводятся номера строк, которые хочет удалить пользователь
    while True:
        try:
            numbers = list(map(int, input().split()))
            dataset.drop(set(numbers), inplace=True)
        except KeyError:
            print("Один из элементов с заданным индексом не существует, введите другие индексы")
            continue
        else:
            break
    dataset.reset_index(inplace=True, drop=True)
    return dataset


def deleting_by_range(dataset):
    # вводятся начало и конец границ удаления
    flag_start = False
    while True:
        if not flag_start:
            try:
                start = int(input())
            except ValueError:
                print("Введите целочисленное значение начала диапазона удаления")
                continue
            else:
                if start >= 0:
                    flag_start = True
        # выполняется только если ввели начало диапазона
        if (flag_start):
            try:
                finish = int(input())
            except ValueError:
                print("Введите целочисленное значение конца диапазона удаления")
                continue
            else:
                if (start < finish):
                    dataset.drop(np.arange(start, finish + 1), inplace=True)
                    break
                else:
                    print("Значение конца диапазона должно быть больше значения его начала!")
                    flag_start = False
                    continue
        else:
            print("Начало диапазона должно быть больше 0!")
            continue
    dataset.reset_index(inplace=True, drop=True)
    return dataset


def sort_descending(dataset):  # Сортировка по убыванию
    while True:
        columns = dataset.columns
        column = str(input())
        if column in columns:
            break
        else:
            continue

    dataset.sort_values(column, inplace=True, ascending=True)
    return dataset


def sort_ascending(dataset):  # Сортировка о возрастанию
    while True:
        columns = dataset.columns
        column = str(input())
        if column in columns:
            break
        else:
            continue

    dataset.sort_values(column, inplace=True)

    return dataset


def searching(dataset):  # Поиск по диапазону индексов и нужных столбцов таблицы
    flag_start = False
    while True:
        if not flag_start:
            try:
                start = int(input())
            except ValueError:
                print("Введите целочисленное значение начала диапазона поиска")
                continue
            else:
                if start >= 0:
                    flag_start = True

        if (flag_start):
            try:
                finish = int(input())
            except ValueError:
                print("Введите целочисленное значение конца диапазона поиска")
                continue
            else:
                if (start < finish):

                    break
                else:
                    print("Значение конца диапазона должно быть больше значения его начала!")
                    flag_start = False
                    continue
        else:
            print("Начало диапазона должно быть больше 0!")
            continue
    print("Индексы от '{}' до '{}'".format(start, finish))

    columns = dataset.columns
    search_columns = list(map(str, input().split()))
    print("Названия столбцов: '{}''".format(search_columns))
    array = np.array([])
    for i in np.arange(len(search_columns)):
        if search_columns[i] in columns:
            array = np.append(array, search_columns[i])
    array = list(set(array))  # Убираем повторяющиеся строки

    sub_dataset = dataset.loc[start: finish, array]
    return sub_dataset


def change(dataset):  # Изменение параметров по индексу
    columns = dataset.columns
    ocean_proximity_array = dataset.ocean_proximity.unique()
    flag_number = False
    flag_columns = False
    while True:
        try:
            if not flag_number:
                print("Введите номер элемента для изменения")
                number = int(input())
        except ValueError:
            print("Введите целое число!")
            continue
        else:
            if number in np.arange(len(dataset)):  # Проверка на ввод существующего индекса
                flag_number = True
                if not flag_columns:
                    print("Введите имя столбца:")
                    column = str(input())

                if column in columns:  # Проверка на корректность ввода имени столбца
                    flag_columns = True
                    if column == 'ocean_proximity':
                        print("Введите строку:")
                        flag_break = False

                        while not flag_break:
                            print("Введите строку из списка: {}".format(ocean_proximity_array))
                            string = str(input())
                            if string in ocean_proximity_array:
                                dataset.loc[number, column] = string
                                flag_break = True
                        break

                    else:
                        print("Введите число:")
                        while True:
                            try:
                                digit = float(input())
                                dataset.loc[number, column] = digit
                            except ValueError:
                                print("Нужно ввести число!")
                                continue
                            else:
                                break
                        break
                else:
                    continue

            else:
                print("Введенный номер находится вне границ таблицы")
                continue

    return dataset


def writing_to_csv(dataset):
    dataset.to_csv(index=False)
    return None