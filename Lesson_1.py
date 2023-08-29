import glob
import pandas as pd
import numpy as np
import datetime
import re
import os

# Зададим директорию с результатами анализов
folder_path = r'C:\Projects\geochem_python_lessons\Lessons\1 Load tables\Analytics_Actlabs'

# Найдём все xlsx таблицы в директории
files_list = glob.glob(folder_path + '\*.xlsx')
print(*files_list, sep = '\n')

# Напишем функцию-шаблон для чтения таблицы с результатами аналищов
def read_excel(filepath):
    # читаем xlsx таблицу
    report_df = pd.read_excel(filepath, sheet_name = 'Results', header = None)
    
    # читаем номер и дату отчёта в отдельные переменные
    report_number = report_df.iloc[0, 0].split(': ')[1]
    report_date_str = report_df.iloc[1, 0].split(': ')[1]
    report_date = datetime.datetime.strptime(report_date_str, '%d/%m/%Y')
    
    # получим метаданные
    elements_metadata = report_df.iloc[2:6, :]
    elements_metadata = elements_metadata.set_index(0)
    elements_metadata.index.name = None
    full_names = []
    for col in elements_metadata:
        name = elements_metadata.loc['Analyte Symbol', col] + '_' + \
                elements_metadata.loc['Unit Symbol', col].replace('%', 'pct') + '_' + \
                elements_metadata.loc['Analysis Method', col].replace('-','_')
        full_names.append(name)
    elements_metadata.columns = full_names
    elements_metadata
    
    # прочитаем результаты анализов
    df = report_df.iloc[6:].copy()
    df.columns = ['SampleID'] + full_names
    df['Report_Number'] = report_number
    df['Report_Date'] = report_date
    
    return elements_metadata, df

# прочитаем все таблицы и добавим в список
tables_list = []
for file in files_list:
    elements_metadata, file_df = read_excel(file)
    tables_list.append(file_df)

# объединим таблицы в сводную
df = pd.concat(tables_list, ignore_index = True)


# Найдём дубликаты
all_ids = list(df['SampleID'])
duplicate_ids = [i for i in set(all_ids) if all_ids.count(i) > 1]

# исключим дубликаты
df = df.sort_values(by = 'Report_Date').drop_duplicates(subset = 'SampleID', keep = 'last').sort_index()

# проверим наличие дубликатов
print(df.loc[df['SampleID'].isin(duplicate_ids), ['SampleID', 'Au_ppb_AR_MS', 'Report_Date']])

# прочитаем таблицу soil_samples
points_link = r'C:\Projects\geochem_python_lessons\Lessons\1 Load tables\Lesson_1.gdb\soil_samples'
fields  = ['SampleID', 'X_coord', 'Y_coord']
points_list = [i for i in arcpy.da.SearchCursor(points_link, fields)]
points_df = pd.DataFrame(points_list, columns = fields)

# объединим координаты и результаты анализов
df['SampleID'] = df['SampleID'].apply(lambda x: x.upper())
df_merged = pd.merge(points_df, df, on = 'SampleID', how='inner')


# найдём все текстовые значения по столбцам
def show_string_values(columns, df):
    all_string_list = []
    for col in columns:
        index = df[col].apply(lambda x: isinstance(x, str))
        string_values_list = df.loc[index, col].unique()
        if len(string_values_list) > 0:
            string = col + ': ' + ', '.join(string_values_list)
            all_string_list.append(string)
    if len(all_string_list) > 0:
        print(*all_string_list, sep = '\n')
    else:
        print('Текстовые значения не найдены')

show_string_values(elements_metadata, df_merged)

# напишем функцию для замены текстовых значений
def replace_text_value(value):
    # определим тип значения
    if any([type(value) == int,
           type(value) == float,
           value is None]):
        return value
    
    # если значение не числовое, то попробуем найти номер
    pattern = re.compile(r'[<>]?\s*(\d+(\.\d+)?)')
    match = pattern.search(value)
    if bool(match):
        number = float(match.group(1))
        if '<' in value:
            number /= 2
        elif '>' in value:
            number *= 1.1
        return number
    elif bool(re.search(r'n.*a', value, re.IGNORECASE)):
        return None

# выполним замену текстовых значений
for col in elements_metadata:
    df_merged[col] = df_merged[col].apply(lambda x: replace_text_value(x))
    
show_string_values(elements_metadata, df_merged)

# заполним таблицу метаданных
for col in elements_metadata:
    # вычислим количество числовых и NaN значений
    num_idx = df_merged[col].notna()
    numeric_count = len(df_merged.loc[num_idx, col])
    nan_count = len(df_merged.loc[~num_idx, col])
    elements_metadata.loc['N', col] = numeric_count
    elements_metadata.loc['N/A', col] = nan_count

    # вычислим значения меньше предела обнаружения
    dl = elements_metadata.loc['Detection Limit', col]
    less_dl_idx = df_merged[col] < dl
    less_dl_count = len(df_merged.loc[less_dl_idx, col])
    elements_metadata.loc['Less than D.L.', col] = less_dl_count

    # посчитаем количество значений равное пределу обнаружения
    equal_dl_idx = df_merged[col] == dl
    equal_dl_count = len(df_merged.loc[equal_dl_idx, col])
    elements_metadata.loc['Equal D.L.', col] = equal_dl_count

    # вычислим коэффициент чувствительности
    sensitivity_k = (less_dl_count + equal_dl_count/2) / numeric_count
    elements_metadata.loc['Sensitivity k', col] = sensitivity_k

    # базовая описательная статистика
    mean_value = df_merged[col].mean()
    std_value = df_merged[col].std()
    mean_geom_value = np.log10(df_merged[col]).mean()
    log10_std_value = np.log10(df_merged[col]).std()
    elements_metadata.loc['Mean value', col] = mean_value
    elements_metadata.loc['STD value', col] = std_value
    elements_metadata.loc['Mean geom value', col] = mean_geom_value
    elements_metadata.loc['STD log10 value', col] = log10_std_value


# подготовим путь для экспорта таблиц
folder_path = r'C:\Projects\geochem_python_lessons\Lessons\1 Load tables\DATA'
file_name = 'final_table.xlsx'
file_path = os.path.join(folder_path, file_name)

# запишем две таблицы в один файл, но разные листы
with pd.ExcelWriter(file_path) as writer:
    df_merged.to_excel(writer, sheet_name = 'Results', index=False)
    elements_metadata.to_excel(writer, sheet_name = 'Metadata')

# Визуализация в ArcGIS
# Вариант 1

# настроим окружение
arcpy.env.workspace = r"C:\Projects\geochem_python_lessons\Lessons\1 Load tables\Lesson_1.gdb"
arcpy.env.overwriteOutput = True

# импортируем таблицу excel
arcpy.conversion.ExcelToTable(file_path, 'Lab_results', 'Results')

# изменим псевдонимы переменных таблицы
for col in elements_metadata:
    alias = col.split('_')[0] + ' (' + col.split('_')[1] + ')'
    arcpy.management.AlterField('Lab_results', col, new_field_alias = alias)

# сделаем копию класса с точками опробования
arcpy.management.CopyFeatures('soil_samples', 'soil_samples_with_analysis_option_1')

# создадим выборку столбцов для объединения
soil_samples_fields = [i.name for i in arcpy.ListFields('soil_samples_with_analysis_option_1')]
fields_to_join = [i for i in df_merged.columns if i not in soil_samples_fields]

# сделаем постоянное объединение таблицы и точечного класса
arcpy.management.JoinField(in_data = 'soil_samples_with_analysis_option_1', 
                           in_field = 'SampleID', 
                           join_table = 'Lab_results', 
                           join_field = 'SampleID', 
                           fields = fields_to_join)

# Вариант 2
# Экспорт таблицы в Numpy array и экспорт его в класс данных

# проверим типы данных (object -> string)
df_update = df_merged.copy()
for col in df_update:
    if df_update[col].dtype == 'object':
        df_update[col] = df_update[col].astype('|S')

# конвертируем pandas DF в Numpy array. Последний в класс данных
structured_array = df_update.to_records()
sr = arcpy.Describe('soil_samples').spatialReference
output_feature_class = r'C:\Projects\geochem_python_lessons\Lessons\1 Load tables\Lesson_1.gdb\soil_samples_with_analysis_option_2'
arcpy.da.NumPyArrayToFeatureClass(structured_array, output_feature_class, ('X_coord', 'Y_coord'), sr)
arcpy.management.MakeFeatureLayer(output_feature_class, 'soil_samples_with_analysis_option_2')

# Вариант 3
# Обновим напрямую класс данных

# сделаем копию класса данных
output_fc = 'soil_samples_with_analysis_option_3'
arcpy.management.CopyFeatures('soil_samples', output_fc)

# сделаем выборку полей
soil_samples_fields = [i.name for i in arcpy.ListFields('soil_samples_with_analysis_option_3')]
fields_to_join = [i for i in df_merged.columns if i not in soil_samples_fields]

# Добавим поля
for col in fields_to_join:
    # подготовим пседоним
    alias = col.split('_')[0] + ' (' + col.split('_')[1] + ')'
    
    # подготовим тип данных
    df_type = df_merged[col].dtype
    if df_type == 'object':
        field_type = 'TEXT'
    elif df_type == 'float64':
        field_type = 'FLOAT'
    elif df_type == 'int64':
        field_type = 'LONG'
    elif df_type == '<M8[ns]':
        field_type = 'DATE'
    
    # добавим поле
    arcpy.management.AddField(in_table = output_fc,
                             field_name = col,
                             field_type = field_type,
                             field_alias = alias)    
    
# обновим значения
df_update = df_merged.set_index('SampleID')
for idx, item in df_update.loc[:, fields_to_join].iterrows():
    with arcpy.da.UpdateCursor(output_fc, fields_to_join, f"SampleID = '{idx}'") as cur:
        for row in cur:
            cur.updateRow(list(item))

