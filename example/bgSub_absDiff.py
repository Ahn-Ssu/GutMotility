import cv2
import utilz
import sys

src_path = 'E:\git\GutMotility\data\Control_7 dpf\Control_7dpf_04.mp4'
cap = cv2.VideoCapture(utilz.path_processing(src_path))
if not cap.isOpened():
    print('Video open failed!')
    sys.exit()
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

# 배경 영상 등록
ret, back = cap.read()
if not ret:
    print('Background image registration failed!')
    sys.exit()
back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
back = back[160:960, :]

# video writer cfg

fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
writer = cv2.VideoWriter(f'absdiff_output.avi', fourcc, fps, (1920,800), isColor=False)
start = utilz.get_time()
# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 현재 프레임 영상 그레이스케일 변환
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[160:960, :] # crop

    # 노이즈 제거
    
    diff = cv2.absdiff(frame, back)
    diff = cv2.GaussianBlur(diff, (0, 0), 1.0)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    cv2.putText(diff, utilz.calc_time_by_sec(start), (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255))
    
    # 레이브링을 이용하여 바운딩 박스 표시
    # cnt, _, stats, _ = cv2.connectedComponentsWithStats(diff)
    # for i in range(1, cnt):
    #     x, y, w, h, s = stats[i]
    #     if s < 100:
    #         continue
            
    #     cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)
    writer.write(diff)
    
    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)

    if cv2.waitKey(30) == 27:
        break

writer.release()
cap.release()
cv2.destroyAllWindows()