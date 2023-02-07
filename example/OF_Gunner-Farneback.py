import cv2, numpy as np
import utilz

# 플로우 결과 그리기 ---①
def drawFlow(img,flow,step=16):
  h,w = img.shape[:2]
  h = 480
  
  # 16픽셀 간격의 그리드 인덱스 구하기 ---②
  idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int)
  indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)
  
  for x,y in indices:   # 인덱스 순회
    y += 140
    # 각 그리드 인덱스 위치에 점 그리기 ---③
    cv2.circle(img, (x,y), 1, (0,255,0), -1)
    # 각 그리드 인덱스에 해당하는 플로우 결과 값 (이동 거리)  ---④
    dx,dy = flow[y, x].astype(np.int)
    # 각 그리드 인덱스 위치에서 이동한 거리 만큼 선 그리기 ---⑤
    if dx > 0 or dy > 0:
        cv2.line(img, (x,y), (x+dx, y+dy), (0,0, 255),2, cv2.LINE_AA )
    else:
        cv2.line(img, (x,y), (x+dx, y+dy), (0,255, 0),2, cv2.LINE_AA )


prev = None # 이전 프레임 저장 변수

cap = cv2.VideoCapture('E:/git/GutMotility/data/Control_7 dpf/Control_7dpf_04.mp4')
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(1000/fps)
start = utilz.get_time()
fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
writer = cv2.VideoWriter(f'OF_GF_output.avi', fourcc, fps, (1920,800))

while cap.isOpened():
  ret,frame = cap.read()
  if not ret: break
  frame = frame[160:960, :, :]
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

  # 최초 프레임 경우 
  if prev is None: 
    prev = gray # 첫 이전 프레임 --- ⑥
  else:
    # 이전, 이후 프레임으로 옵티컬 플로우 계산 ---⑦
    flow = cv2.calcOpticalFlowFarneback(prev,gray,None,\
                0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN) 
    # 계산 결과 그리기, 선언한 함수 호출 ---⑧
    drawFlow(frame,flow)
    # 다음 프레임을 위해 이월 ---⑨
    prev = gray
  
  cv2.putText(frame, utilz.calc_time_by_sec(start), (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255))
  cv2.imshow('OpticalFlow-Farneback', frame)
  writer.write(frame)
  if cv2.waitKey(delay) == 27:
      break
cap.release()
writer.release()
cv2.destroyAllWindows()