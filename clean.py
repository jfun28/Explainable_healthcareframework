from nbformat import read, write

def strip_output(nb):
    for cell in nb.cells:
        if hasattr(cell, "outputs"):
            cell.outputs = []
        if hasattr(cell, "prompt_number"):
            del cell["prompt_number"]
problem_file_name = 'C:\\jupyter\\Explainable Healthcare framework\\Explainable_healthcareframework\\DatasetMake\\1. 각 ID별 불균형한 날짜값 균일.ipynb'
save_file_name = 'C:\\jupyter\\Explainable Healthcare framework\\Explainable_healthcareframework\\DatasetMake\\1. 각 ID별 불균형한 날짜값 균일2.ipynb'
import json
from nbformat import write, v4

try:
    with open(problem_file_name, 'r', encoding='utf8') as f:
        content = f.read()
    
    # 가능한 문제 해결: 줄바꿈 문자 등 정리
    content = content.replace('\r\n', '\n')
    
    # JSON으로 변환 시도
    nb_dict = json.loads(content)
    
    # nbformat v4 노트북으로 변환
    nb = v4.upgrade(nb_dict)
    
    # 출력 제거 함수 적용
    strip_output(nb)
    
    # 저장
    write(nb, open(save_file_name, "w", encoding='utf8'))
    print("성공적으로 저장되었습니다.")
except Exception as e:
    print(f"오류: {e}")