def kalman(z_measure,x_last=0,p_last=0,Q=0.5,R=0.5):
    x_mid = x_last
    p_mid = p_last + Q
    kg = p_mid/(p_mid + R)
    x_now = x_mid + kg*(z_measure - x_mid)
    p_now = (1-kg)*p_mid
    p_last = p_now
    x_last = x_now
    return x_now,p_last,x_last

x_last = 0; p_last = 0
Q = 0.5  #系统噪声
Rk = 0.5  #测量噪声
yz_tmp = []
yzs = []

for i in range(len(yzs)):
    pred,p_last,x_last = kalman(yzs[i],x_last,p_last,Q,Rk)
    yz_tmp.append(pred)
