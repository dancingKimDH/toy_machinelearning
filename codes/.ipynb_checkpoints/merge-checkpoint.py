# 파일 리스트
filenames = ['codes/혜인_RecurrenceOfSurgery_preprocess.ipynb', 'codes/동현_RecurrenceOfSurgery_학습.ipynb']

with open('merge.ipynb', 'w') as mergefile:
    for filename in filenames:
        with open(filename, encoding='cp949') as file:
            mergefile.write(file.read())