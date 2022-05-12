import face_recognition
import cv2
import numpy as np
import pickle

all_face_encodings = {}
known_face_encodings = []
known_face_names = []
ChangedSettings = False

with open('./dat/faces.dat', 'rb') as f:
    all_face_encodings = pickle.load(f)

known_face_encodings = np.array(list(all_face_encodings.values()))
known_face_names = list(all_face_encodings.keys())


def SplashScreen():
    splash_screen = cv2.imread('splash_screen.jpg')
    while True:
        cv2.imshow('main', splash_screen)
        if cv2.waitKey(0):
            break
    return splash_screen

def ipcam():
    camera = input(
        '\nВыберите камеру ("0" - встроенная камера, "1" - внешняя камера,"2" - другая вняшеняя камера)\nДля подключения к ip-камере введите "ip" и следуйте инструкции\n(Вы так же можете использовать свой телефон в качестве ip-камеры, установив на телефон приложение "IP Webcam"\n')
    if camera == 'ip':
        camera = str(input('Введите адресс вашей камеры, например:"http://***.***.*.**:****"\n')) + '/video'
    else:
        camera = int(camera)

    with open('./dat/ipcam.dat', 'wb') as f:
        pickle.dump(camera, f)
        print('Камера сохранена')

    return camera

def defoultMode(video_capture,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality):
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    while True:
        ret, frame = video_capture.read()

        # Resize frame of video to 1/10 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=resizexy, fy=resizexy)

        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=RecognitionQuality)
                name = "Неизвестно"
                r, g, b = 0, 0, 0

                # использовать первое совпадение
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # использовать лучшее совпадение
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected
            top *= factor
            right *= factor
            bottom *= factor
            left *= factor

            r, g, b = 0, 200, 0
            if name == "Неизвестно":
                r, g, b = 0, 0, 200

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (r, g, b), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - rectangleSizeAdd), (right, bottom), (r, g, b), cv2.FILLED)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 3), font, FontSize, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('main', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == 27:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    exit()

def OnePersoneMode(video_capture,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality,known_face_encodings,known_face_names):
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    while True:
        ret, frame = video_capture.read()

        # Resize frame of video to 1/10 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=resizexy, fy=resizexy)

        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=RecognitionQuality)
                name = "Неизвестно"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/10 size
            if name != 'Неизвестно':
                top *= factor
                right *= factor
                bottom *= factor
                left *= factor

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - rectangleSizeAdd), (right, bottom), (0, 200, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, FontSize, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('main', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == 27:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    exit()

def settings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality):
    ChangedSettings = True
    setting = input('\nВыберите настройку, введите соответствующую цифру:\nИзменить размер окна - "1"\nУказать разрешение передаваемое камерой - "2"\nВыбрать режим распознавания - "3"\nИзменить камеру - "4"\nИзменить точность распознавания - "5"\nИзменить размер рамки и шрифта - "6"\nЗакончить - Enter\n')
    if setting == '1':
        sizey,sizex = ChangeSize()
        settings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality)
    if setting == '2':
        resizexy,factor,rectangleSizeAdd,FontSize = ChangeResolution()
        settings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality)
    if setting == '3':
        mode = SelectMode()
        settings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality)
    if setting == '4':
        camera = ipcam()
        settings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality)
    if setting == '5':
        RecognitionQuality = ChangeRecognitionQuality()
        settings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality)
    if setting == '6':
        rectangleSizeAdd,FontSize = FontRectangleSize()
        settings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality)
    ChangeSettings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality,ChangedSettings)


def ChangeSettings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality,ChangedSettings):
    if ChangedSettings:
        StartMode(mode, sizey, sizex, camera, resizexy, factor, rectangleSizeAdd, FontSize, RecognitionQuality, known_face_encodings, known_face_names)
    else:
        changeSetting = input('\nИзменить настройки? ("да"/Enter)\n')
        if changeSetting == 'да':
            settings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality)
        else:
            StartMode(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality,known_face_encodings,known_face_names)

def ChangeResolution():
    resize = int(input('\nВыберите разрешение которое транслирует ваша камера:\n1280x720 - введите "1"\n1920x1080 - введите "2"\n640х480 - введите "3"\nуниверсальное (менее производительный вариант) - введите "4"\n'))
    with open('./dat/resolution.dat', 'wb') as f:
        pickle.dump(resize, f)
        print('Разрешение сохранено')
    return SelectResolution(resize)

def SelectResolution(resize):
    resol = ''
    if resize == 2:
        resizexy = 0.25
        rectangleSizeAdd = 30
        FontSize = 0.7
        resol = '1920x1080'
    elif resize == 1:
        resizexy = 0.5
        rectangleSizeAdd = 6
        FontSize = 0.3
        resol = '1280x720'
    elif resize == 3:
        resizexy = 0.25
        rectangleSizeAdd = 15
        FontSize = 0.3
        resol = '640х480'
    elif resize == 4:
        resizexy = 1
        rectangleSizeAdd = 6
        FontSize = 0.3
        resol = 'универсальное'
    else:
        print('\nОжидался ввод "1", "2", "3" или "4"')
        ChangeResolution()
    factor = int(1 / resizexy)
    print(f'Разрешение изображения передаваемого камерой установлено - {resol}')
    return resizexy,factor,rectangleSizeAdd,FontSize

def ChangeRecognitionQuality():
    RecognitionQuality = int(input('Укажите на сколько точным должно быть распознавание, введите число от 1 до 10\n(чем меньше значение, тем точнее распознавание, но тем больше ресурсов требуется от системы)\n'))/10
    with open('./dat/RecognQuality.dat', 'wb') as f:
        pickle.dump(RecognitionQuality, f)
        print('Точность распознавания сохранена')
    return RecognitionQuality

def FontRectangleSize():
    rectangleSizeAdd = int(input("Введите размер рамки\n(Рукомендуемое занчение от 0 до 50)\n"))
    FontSize = int(input("\nВведите размер шрифта\n(Рукомендуемое занчение от 1 до 10)\n"))/10
    return rectangleSizeAdd,FontSize

def ChangeSize():
    sizey = input('\nВведите ширину окна\n')
    sizex = input('Введите высоту окна\n')
    size = str(sizex) + ':' + str(sizey)
    with open('./dat/windowsize.dat', 'wb') as f:
        pickle.dump(size, f)
        print('Размер сохранен')
    return sizey, sizex

def SelectMode():
    mode = input('\nВключить режим распознавания определенного человека?("да"/Enter)\n')
    with open('./dat/mode.dat', 'wb') as f:
        pickle.dump(mode, f)
        print('Режим сохранен')
    return mode

def StartMode(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality,known_face_encodings,known_face_names):
    if isinstance(camera, int):
        video_capture = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    else:
        video_capture = cv2.VideoCapture(camera)

    if video_capture is None or not video_capture.isOpened():
        print('Выбранная камера недоступна')
        camera = ipcam()
        StartMode(mode, sizey, sizex, camera, resizexy, factor, rectangleSizeAdd, FontSize, RecognitionQuality, known_face_encodings, known_face_names)

    if mode == 'да':
        name = input('Введите имя человека, которого необходимо распознать\n')
        if name in known_face_names:
            known_face_encodings = [known_face_encodings[known_face_names.index(name)]]
            known_face_names = [name]
            print('Включен режим распознавания определенного человека')
            cv2.namedWindow('main', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('main', int(sizey), int(sizex))
            print('Готово')
            SplashScreen()
            OnePersoneMode(video_capture,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality,known_face_encodings,known_face_names)
        else:
            print('\nЧеловека с таким именем нет в базе данных\n')
            StartMode(mode, sizey, sizex, camera, resizexy, factor, rectangleSizeAdd, FontSize, RecognitionQuality, known_face_encodings, known_face_names)
    else:
        cv2.namedWindow('main', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('main', int(sizey), int(sizex))
        print('Готово')
        SplashScreen()
        defoultMode(video_capture,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality)

with open('./dat/ipcam.dat', 'rb') as f:
    try:
        camera = pickle.load(f)
    except EOFError:
        camera = ipcam()

with open('./dat/windowsize.dat', 'rb') as f:
    try:
        sizex,sizey = pickle.load(f).split(':')
        print(f'Размер окна {sizey}x{sizex}')
    except EOFError:
        print('\nУкажите размер окна приложения, например: Ширина - 1280, Высота - 720')
        sizey,sizex = ChangeSize()

with open('./dat/resolution.dat', 'rb') as f:
    try:
        resize = pickle.load(f)
        resizexy,factor,rectangleSizeAdd,FontSize = SelectResolution(resize)
    except EOFError:
        resizexy,factor,rectangleSizeAdd,FontSize = ChangeResolution()

with open('./dat/RecognQuality.dat', 'rb') as f:
    try:
        RecognitionQuality = pickle.load(f)
    except EOFError:
        RecognitionQuality = ChangeRecognitionQuality()
    print('Установленное качество распознавания - ', RecognitionQuality*10)

with open('./dat/mode.dat', 'rb') as f:
    try:
        mode = pickle.load(f)
        if mode == 'да':
            print('Режим распознавания определенного человека активен')
        else:
            print('Активен стандартный режим распознавания')
        ChangeSettings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality,ChangedSettings)
    except EOFError:
        mode = SelectMode()
        ChangeSettings(mode,sizey,sizex,camera,resizexy,factor,rectangleSizeAdd,FontSize,RecognitionQuality,ChangedSettings)
