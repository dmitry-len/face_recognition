# coding: utf-8
from cx_Freeze import setup, Executable

executables = [Executable('main.py', targetName='face_recognition.exe', icon="camera.ico"),
               Executable('train.py', targetName='save_faces_to_database.exe', icon="database.ico")]

include_files = ['dat',
                 'faces_f',
                 '.idea',
                 'splash_screen.jpg'
                 ]

options = {
    'build_exe': {
        'include_msvcr': True,
        'build_exe': 'face_recognition_2_3',
        'include_files': include_files,
    }
}

setup(name='recognizer',
      version='2.3.0',
      description='camera face recognition',
      executables=executables,
      options=options)