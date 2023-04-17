本质上，为什么DCA能从1443取数据呢？
1443的代码把数据从ADCbuffer搬运到60pin，再从60pin通过网口传输；
如果当read的大小达到一个一帧的整数倍时，我就开始传输操作

能否在先有逻辑上增加，而非替换？
```
frameIndex = 0
frameBufPtr = 0
part_frame = 0

if part_frame > 0:
    if ReadPtrBufIndex >= frame_len - part_frame
        取出当前 (0, frame_len-part_frame) 数据，开始发布
        frameIndex++
        frameBufPtr = frame_len - part_frame
        part_frame = 0
else:
    if ReadPtrBufIndex >= frameBufPtr + frame_len
        就说明收集满1帧了，开始发布数据
        取出 (frameBufPtr, frame_len) 数据，开始发布
        frameBufPtr += frame_len
        frameIndex++

if ReadPtrBufIndex+u32Size > buffer size:
    part_frame = ReadPtrBufIndex - frameBufPtr
    if part_frame > 0:
        取出 (frameBufPtr, part_frame) 的数据，暂时不发布
    else:
        frameBufPtr = 0
``` 

- 先验证一下，包顺序不对的情况是否会出现，如果不会出现，那我就可以放心的修改writeDataToBuffer_Inline
    - 没有包顺序不对的情况
- 我就不要写文件这个线程了，而是直接在接收数据的线程中，把数据保存下来
    - 关闭写文件线程；done
    - 停止线程中，不再写文件；done
    - 接收数据线程中，不再等待写文件的信号，因为已经不再写文件了；done
    - setFileName停掉；done
    - 在接收数据线程中，增加保存文件函数；done
    - ReorderAlgorithm是干什么用的？必须要有吗？有就有吧；done
- 预期效果：
    - 每一帧保存一个文件；成功了
- 当填满buffer后，重开呢？也就是sensor stop, sensor start，这样的好处是：
    - 避免错误累积
    - 简化取数据的逻辑，因为不用考虑断层。
    - 如果不重开，我现在就实现完了——那就把目前的实现完了吧
    - 怎么实现？
        - 当sensor stop时，不删除内存，而是全部初始化
- 把frameLen、nh作为参数传进去；done
- 组织数据格式，发布数据；done
- 目前发布数据的流程已经走通，再检查最后一遍整体逻辑。
    - 发现错误，要reorder再保存，而reorder必须用数组，所以还是得先赋值给数组，再赋值给vector
- 现在代码没问题了，但是数据处理有问题，即便是原始的数据，也有问题，是参数不对的原因吗？

### 代码走读
```
main():
    读取输入命令，s8Command := start_record，s8CommandArg := cf.json
    验证json文件的正确性
    判断是否有record process，如果有就直接return，没有就创建用于状态记录的共享内存
    // 建立主机端的配置和数据的Socket通信
    ConnectRFDCCard_RecordMode(gsEthConfigMode)
    // 这里包含接收和保存数据的全部逻辑
    StartRecordData(gsStartRecConfigMode, pub): 
        验证数据保存路径、RecordStopMode的正确性
        创建配置接收端objUdpConfigRecv
        启动配置读取进程，thread tConfigData([&]{objUdpConfigRecv.readConfigDatagrams();})
        创建数据接收端objUdpDataRecv
        启动数据读取进程，thread tRawData([&]{objUdpDataRecv.readData()})
            while bSocketState and ros::ok():
                // 从SenderAddr这个地址，接收MAX_BYTES_PER_PACKET多字节，到s8ReceiveBuf中
                s32CtPktRecvSize = recvfrom(socket, s8ReceiveBuf, MAX_BYTES_PER_PACKET, 0, SenderAddr)
                if s32CtPktRecvSize > 0:
                    解析packet，前4个字节表示第几个包，接下来6个字节表示raw data的字节数
                    if bFirstPktSet:
                        初始化一堆变量
                        // 创建并打开文件，准备写，而且是二进制、接着写
                        setFileName():
                            pRecordDataFile = fopen(strFileName1, "wb+")
                    // 如果当前包就是我预期的包
                    if u32CtPktNum == u32NextPktNum:
                        NumOfRecvPackets++
                        writeDataToBuffer_Inline(s8ReceiveBuf[10:], s32CtPktRecvSize-10):
                            继续在之前用的buffer的位置，写新收到的那么多数据
                            如果buffer1满了：
                                等待buffer2写完的信号
                                切换到buffer2，从零开始写
                                通知写线程，你可以写buffer1了
                            记录当数据包out of sequence的状态
                            // 这就是所谓的乒乓操作吧，好处是可以一边读buffer1，一边写buffer2
                        设置一些统计变量: 下一个包num、直到目前发送的包的byte、上次发的包的大小
                            u32NextPktNum = u32CtPktNum + 1
                            u64BytesSentTillPrevPkt = u64BytesSentTillCtPkt
                            s32PrevPktRecvSize = s32CtPktRecvSize
                    // 如果当前包是我预期的包之前的包
                    else if u32CtPktNum < u32NextPktNum:
                        大概是回退当前buffer的指针，然后开始写
                    // 如果当前包是我预期的包之后的包；因为网络良好，后两种情况大概率不会出现
                    else
                        大概是把中间没收到的包的位置填零，那么之后有可能再次收到这个包，也有可能永远没收到，就是0了
        启动数据保存进程，thread tRawData2([&] { objUdpDataRecv.Thread_WriteDataToFile();})
            writeDataToFile_Inline(s8RecBuf2, u32WritePtrSize)
                等待buffer1或buffer2填满的信号
                收到信号后，把满的buffer写到文件里
```

### 自己的修改
1. 先仿照激光雷达，把大框架搭起来，别管对不对
    - 创建消息发布者、数据接收线程、消息发布线程
    - 开始数据接收线程
        - 组织消息格式
            - 怎么把数组付给msg啊？这个大小应该是变换的；
                - 编译出来的结果是ector，所以就是vector相互赋值
    - 开始数据发布线程
2. 再把大框架分解，每个部分逐步完成
3. 剩下的就是编译，查看一下topic，再新建一个功能包，解析topic
4. 我这个功能包里有两个node：cli_control_main.cpp是用来配置DCA1000的，cli_record_main.cpp是用来录制数据的。这两个主函数，每个都依赖一些文件，每个文件有自己的header，怎么编译和链接？
我现在想干什么？遇到了什么问题？
- 现在编译通过了，但是说我在NodeHandle后才ros.init()
- 猜想：编译RF_API这个库时，没有调用init，就直接用了NodeHandle，可能是这个导致的
    - 当作参数传给函数，这个可以

为了实现实时收数据，DCA1000源代码我修改了哪里？
- 主函数增加了ros.init和nh，把nh传给写StartRecordData函数
- StartRecordData用nh声明publisher给UdpReceiver
- 把UdpReceiver写文件的函数，改成发布文件的函数；停止函数里的写文件注释掉，因为我不要最后一帧，是不完整的
- ReadData函数中的setFileName注释，因为我不再写文件了
- 重要：buffer size改成1帧大小；packet size改成1帧的公约数；文件大小也改成1帧大小
    - 现在buffer size肯定是1帧大小，但packet size一定是一个chirp的大小吗？万一一次读取的包没有这么多呢？但大概率是，不然buffer怎么能填满呢？
    - 我不能这么想，我设置的buffer size=196608，packet size=1024，所以结果是这样的，问题应该是：一帧是这么大么？
    - 如果我把pub变成write，理论上，我现在的代码和源代码逻辑一样，只是buffer和packet size改变了；如果这样做是没错的，那我pub了，理论上也是没错把三
    - 用原始代码收一波，看看结果是不是对的；
        - 虽然不敢说一定对，但至少数据不会报错，明天再好好做一组实验
        - 有数据但是不对——这个说明matlab的代码存在问题TODO:
    - 再用修改后的、保存每帧的代码收一波，看看结果是不是一样的；
        - 发了10帧，怎么只有8帧结果保存了？
            - 怀疑是还没保存完，就发出了停止信号，DCA就停止了？
            - 也就是说，我只是把buffer、packet、file size一改，就会收不到10帧了？并且收到的数据还会报错
        - 这个无论是保存文件，还是pub，结果没区别，都会报错
        - 我现在甚至能够做到启动-停止-启动的收数据版本，但是数据量不对
        - 是丢包吗？把时延增大试试
    - 单纯把packet增大，结果是不影响的

### 19:34
- 真的是收的数据错了吗？会不会是matlab解析错了？
- 如果不修改buffer size和file size，就不会出现
- 

### 13:09
我现在要干什么？
- 改代码吗？如果现在功能实现是正确的，就可以直接使用吧，毕竟这些东西太多了，万一哪里没有理解正确，就可能改错了
    - 可是这个逻辑你真的理解了吗？
        - 每次接收的一波数据是一个UDP packet还是多个packet？
            - recvfrom返回的就是1个UDP packet，最大是1462
        - 每个packet是包含1帧数据吗？如果不是一帧，怎么知道一帧的开始和结束？不然的话没法解析数据
            - 每个packet不管你的帧的大小，并不是整数个packet凑成1帧
        - buffer里，数据是接着存，还是从头开始存？这个问题很严重，因为你每次都发布u32buffer，万一buffer里的数据不是当前时刻的帧呢？
            - buffer里的数据是用一个指针按顺序存放的，1个buffer约75M，75e6个int8
            - 如果没理解错的话，当一个buffer满了（>75e6），就会给写进程一个信号，就开始写文件，也即发布消息
        - 25us这个事能硬件配置吗？还是默认的？
            - 默认是25us，软件可以设置；delay越小，吞吐量越大，但丢包概率越大
- pub没法传参，或者说传参很不方便，想把pub作为receiver的属性，等到StartRecordData时，把nh传进来
    - 没有问题，按计划执行

### 明天要做的
- 现在数据是发布了，但是数据对不对啊，怎么解析啊？
    - 我可以解析端给保存成文件，再用matlab去生成点云，目的就是为了看看，数据对不对
- 收集75M的数据再写文件，对于离线处理来说无所谓，但我实时处理，我该怎么发布数据？
    - 这个可能要计算一下，一帧的数据量，比如120x512x8x3=1474560 个int16，如果我想一次传输一帧，那就把buffer的大小设置成1474560x2？
    - 一个packet最大是1452字节，大概1.5kb，而1帧需要3Mb，所以一个packet不可能有1帧数据！那么packet是什么打包机制呢？
    - 1帧发布一次，好处是我每次处理的数据不需要分离帧，直接生成1帧点云
        - 那么我就要修改写文件线程的逻辑？看一下frameMode是怎么做到的，我只要把frame改成1即可
            - 草，frameMode好像只能在multimode下用，DCA的CLI能做到这件事吗？mmwave studio是怎么做到收固定帧数就停止的？
    - 区分不同帧这事，确实不好做吧，是不是通过控制红色板子实现的，绿色板子只负责收，不管你有多少帧：那么，我只要让红色板子的代码只发送固定的帧，那么绿色板子也就只能接收这么多了。没错，我觉得是这样做的。
    - 那么我现在该怎么办？把buffer的大小，设置成一帧的大小，这样就是一帧发一次，可是能这么完美吗？
    - 本质是：一个packet是不是包含整数个chirp？如果DCA端并不负责区分帧或chirp；那我怎样判断一帧数据的开始？
        - 如果注定只能通过数据量来判断，那也就是说，我每次要把上次不足的数据给记住，再和下次的数据拼到一起——可行，但是一旦错位一点，就全都错了
    - frameMode到底是什么，DCA的CLI到底能不能控制帧？不能
        - 不能，感觉DCA确实不该负责这件事，因为它只是用来传输原始数据的，而原始数据包含多少帧，哪里是一帧的开始，这些确实应该是IWR负责吧
    - 要不先保存一下数据，看看长什么样？可是保存了你也没法验证对不对？
    - 如果每个包，一定是1456字节，那么假设不丢包，是可以根据帧长度来解析帧数据的
- 开启写文件线程后的几行在干什么？猜应该是判断何时停止线程？
    - 不是，停止命令通过Callback的形式，很早就声明了，最后只是判断一下命令是否成功，以及是否timeout了

### 18:02
我想干什么？用DCA1000和IWR实时提取原始数据，并实时处理。现在的问题是，IWR的demo运行的同时，我已经可以用DCA1000持续地收集原始数据了，现在的问题是如何将数据解析成多帧？
- 现在的逻辑是：一次发送的数据是75e6个字节，里面包含了很多帧
    - 首先，这75e6个字节大概率不是整数个帧
    - 其次，当等来下一波数据时，下一波数据我就找不到一帧的开始
    - 最后，帧之间没有标记，只能通过数据量来分离吗？万一少了点数据，不就全都错位了吗？可是mmwave studio也存在这个问题啊？
- 首先，写接收端，赶紧把数据保存下来，打通数据保存的链路done
    - 理论上，原来是把数据直接保存文件；现在是先把数据发布，再接收后保存；理论上是一样的
- Big!!! IWR可以控制发多少帧的数据，那我就以10Hz收集毫米波雷达数据，收完就发布，发布完就计算；100ms，收发数据占40ms，那么计算就必须60ms内完成
    - 首先，试试IWR，一次发送10帧，把数据保存下来，解析一下，看看是不是10帧
        - 不对啊，原来的逻辑是，等到buffer满了才开始发数据，现在10帧，显然没满啊，难道就是因为时间间隔，认为发完了？
        - 数据量不对，查找原因
            - 用原来的、保存成文件的代码，数据量非常准确没有问题。看来问题出在自己发布数据的代码
        - 现在有2种实现思路：
            - 不修改DCA代码，直接调用CLI工具，保存成文件；处理的进程读取文件后处理；
                - 没问题，每次保存文件都是清空上一次的，相当于我的Queue size是1？
                    - 这样应该不行，接收端在读的时候，发送端可能就在写了
                - 甚至，我可以在传输端，把文件保存，然后读取，并发布，相当于多了一步处理
                    - 可行性：需要看DCA是否10帧结束就停止信号。原因未知，但我觉得是可以的
            - 修改DCA代码，直接发布数据，而不是保存成文件，也就是我目前这样，但是没有成功

### 阶段性总结
目标：用DCA1000+IWR实时收集并处理原始数据
结果：
- 能用代码完全替代mmwave studio的功能
- 满足导航实时性的收集和处理数据仍面临问题

烧写了官方demo的情况下，为了能实时提取原始数据，采取的方案有：

方案一：让IWR持续地发数据，DCA端接收数据包
- 问题：如何区分不同的帧？目前IWR代码发送的LVDS数据是不包含header的
- 解决：
    - 修改IWR的代码，让他发送的LVDS数据包含header
    - 通过帧长度来区分不同的帧
- 既然现在每次buffer满了就触发数据发送机制，调整packet大小，让1帧=整数N个packet，让buffer size > N个packet，小于N+1个packet，那么每次buffer存满一帧就会发布
    - 用源代码试一下：修改buffer size、packet size、file size；预期结果是，每一帧保存成一个文件，大小都是1966080字节
        - 现在确实是1帧1个文件，接下来就是解析每一帧，看看每一帧是否正确

方案二：让IWR发送1帧数据后停止，DCA接收到，IWR再发送1帧数据，重复这个过程
- 问题：给IWR发送命令需要300ms，这个时间太长了，除非可以1s作一次决策
- 解决：
    - 修改IWR代码，让它只做LVDS streaming，别整这么多没用的
    - 不要重新发全部指令，就发送sensorStart -> DCA receive -> sensorStop；这样做的好处是，IWR一侧，我是不需要做任何修改的，把现有的当作黑盒就可以了。
while True:
    sensorStart 0
    receive and publish data
    sleep()
    sensorStop
- 但我现在对DCA的控制代码，还是不能完全理解。
    - stop_record命令执行时间太长了，我希望他立即结束，为啥会等这么久
    - 当IWR发送完10帧，为什么DCA端会返回Start Record command : Success？谁给他的信号？就是说，IWR停止时，DCA是怎么知道的？难不成IWR返回了什么信息？？？
    - DCA端，是怎么建立Socket的，怎么组织UDP包的？？？
    - 可是我让IWR持续地发数据，然后随机的结束+开始，但保存的文件大小却没有改变，我怀疑DCA只保存了第一次的文件结果，即Record is completed之前的

理想情况：
- IWR端的代码只做ADC数据搬运，不要做任何点云生成操作，给每一帧的开头加header，持续地发数据；
- DCA就把ADC数据以UDP packet传给PC
- PC把UDP packet按顺序存到buffer里，并根据header来区分和校验帧，每次提取一帧发布出去

建文的建议：
    - 赶紧用起来
    - 在别人的基础上实现，不要自己实现

### 终极目标
目标：在ROS下收集毫米波雷达、激光雷达、IMU数据，并且小车的运动要能够通过键盘控制
    - 还有什么信号？里程计？
    - 最终形态：启动一个launch文件，小车、激光雷达、毫米波雷达同时启动，发布多种话题信号；让小车控制节点单独启动
    - 需要两台主机，怎么通信？

任务1：毫米波雷达
- 配置DCA、启动DCA、启动IWR、关闭DCA
- 我现在DCA和IWR是割裂的，手动地先启动DCA，再启动IWR，再手动关闭IWR和DCA，能不能做到全自动化：当运行时，所有节点按顺序启动，当Ctrl+C结束时，IWR和DCA自行关闭
    - 通过shell脚本，先启动DCA launch，再运行python IWR；可以是可以，太麻烦了，何不让IWR先sleep一会呢哈哈哈
    - 把IWR python写成一个ros节点，在DCA后启动，先sleep一会，当Ctrl+C时，sensorStop；DCA执行stop_record命令，对啊，谁说DCA的停止命令非要DCA的代码执行啊，让IWR一块执行了不就好了？
        - 端口号作为参数输入
        - IWR的配置文件作为参数输入
        - DCA的配置文件作为参数输入
        - CLI位置作为参数输入
        - Ctrl+C时，sensorStop，stop_record
    - 写一个launch文件，启动DCA、IWR

任务2：激光雷达


任务3：IMU


任务4：小车运动

