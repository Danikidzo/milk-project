import os
import pandas as pd
import matplotlib.pyplot as plt

def load_and_anylise(data_directory="data"):
    """
    Загружает данные из CSV-файлов в указанной, объединяет их в DataFrame, строит графики зависимости тока от напряжения для антибиотиков.
    
    Параметры:
        data_directory (str): Путь к папке с CSV-файлами (по умолчанию "data").
    
    Возвращает:
        pd.DataFrame: DataFrame с объединёнными данными.
    """
    # Сбор данных
    data_rows = []
    
    for root, directories, files in os.walk(data_directory):
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                file_parts = file_name.split('_')
                antibiotic_name = file_parts[0]
                antibiotic_concentration = file_parts[1]
                
                current = pd.read_csv(file_path, index_col=0)['Current, A'].tolist()
                current.append(antibiotic_name)
                current.append(antibiotic_concentration)
                data_rows.append(current)
    
    # Берём столбцы из последнего файла (предполагается, что у всех одинаковые Voltage)
    new_columns = pd.read_csv(file_path, index_col=0)['Voltage, V'].tolist()
    new_columns.extend(['antibiotic', 'concentration'])
    data_frame = pd.DataFrame(data_rows, columns=new_columns)
    #Анализ данных
    voltage_values = pd.read_csv(file_path)['Voltage, V'].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(28, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    for idx, antibiotic_name in enumerate(data_frame['antibiotic'].unique()):
        antibiotic_data = data_frame[data_frame['antibiotic'] == antibiotic_name].iloc[0]
        current_values = antibiotic_data.iloc[:-2].astype(float).values
        ax = axes[idx // 3, idx % 3]
        ax.plot(voltage_values, current_values)
        ax.set_title(antibiotic_name, fontsize=10)
        ax.set_xlabel('Voltage, V', fontsize=8)
        ax.set_ylabel('Current, A', fontsize=8)
        ax.grid(True)
    plt.suptitle('Графики зависимости тока от напряжения для разных антибиотиков', fontsize=14)
    plt.show()
    
    return data_frame
    

