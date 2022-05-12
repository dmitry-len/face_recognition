import face_recognition
import os.path
import pickle

#директория с фотографиями

folder = input('Хотите изменить путь до директории с фотографиями?,("да"/"Enter")\n')

if folder == 'да':
    path = input('Введите полный путь до вашей директории с фотографиями, например: "D:/folder/FolderInFolder/faces"\n')
    with open('./dat/path.dat', 'wb') as f:
        pickle.dump(path, f)
        print('Директория обновлена')
else:
    with open('./dat/path.dat', 'rb') as f:
        path = pickle.load(f)


#получаем список с фотографиями
print('Считывание фотографий из директории...')

image_paths = [os.path.join(path, f) for f in os.listdir(path)]

all_face_encodings = {}
print('Обработка фотографий...')

y=0
for image_path in image_paths:
    #имя человека на фотографии
    subject_number = str(os.path.split(image_path)[1].split(".")[0])
    #данные для распознавания
    x=face_recognition.load_image_file(image_path)
    all_face_encodings[subject_number]=face_recognition.face_encodings(x)[0]
    y+=1
    print(f'Фотография №{y}: {subject_number}')

#сохранение в БД
print('Загрузка данных в БД')
with open('./dat/faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)
    print('Данные загружены в БД')
