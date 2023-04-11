# ackermann_car_nav
学习小车导航ROS代码实现


# 功能包介绍
## mmwave_radar
- IWR烧写官方demo，调成function mode
- 由于实现逻辑的原因，现在代码里的buffer size和packet size是hard coded
    - rf_api_internal.h: buffer size需要修改成1帧数据的字节长度
    - defines.h: packet size需要修改成buffer size的公约数+10，并且不能超过1462，可以设置为1个chirp的字节长度
- 把xWR1xxx雷达的配置文件放到src/mmwave_radar/config目录，修改launch文件的相应参数
- 启动雷达：roslaunch mmwave_radar start_mmwave_radar.launch
- 查看雷达话题：rostopic list, rostopic echo /mmwave_adc_data
- 查看雷达消息格式：rosmsg info mmwave_radar/adcData
- 关闭雷达：roslaunch mmwave_radar stop_mmwave_radar.launch
