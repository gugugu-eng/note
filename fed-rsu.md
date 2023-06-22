## 地图信息

-----------------------------------------------------
参考文献：ns3-ai: Fostering Artificial Intelligence Algorithms for Networking Research

FedVANET: Efficient Federated Learning with Non-IID Data for Vehicular Ad Hoc Networks

FedDQ: Communication-Efficient Federated Learning with Descending Quantization

PTMAC: A Prediction-based TDMA MAC Protocol for Reducing Packet Collisions in VANET

### step1：生成地图跟背景车辆

+ ./usr/share/sumo/tools

+ python osmWebWizard.py 

  ##### Summary:

  #####  Node type statistics:
    Unregulated junctions       : 0
    Dead-end junctions          : 382
    Priority junctions          : 545
    Right-before-left junctions : 209
    Traffic light junctions      : 26
   Network boundaries:
    Original boundary  : 103.83,1.29,103.86,1.35
    Applied offset     : -370060.99,143050.93
    Converted boundary : 0.00,-6051.07,3085.50,0.00

cars：count30 Through Traffic Factor 5

Trucks:count20 Through Traffic Factor 5

Through Traffic Factor指的是路线穿越所选区域的车辆数与路线在所选区域内车辆数的对比。

Count是规定每小时想要生成的车辆数，与车道数和道路长度有关。

问题:每次运行地图里的背景车辆轨迹都不固定，是否会对实验结果产生影响？（控制变量法）



### step2：生成具有固定轨迹路线的参与训练的20辆车辆

traci.start(["sumo-gui", "-c", "/location/map.sumo.cfg"], port=7911)

其中 "sumo-gui" 表示以 GUI（图形用户界面）模式启动 SUMO，"-c" 参数后面跟着 SUMO 的配置文件路径，当使用[sumo-gui](https://sumo.dlr.de/docs/sumo-gui.html)作为服务器时，必须 在处理 TraCI 命令之前使用[*播放* 按钮](https://sumo.dlr.de/docs/sumo-gui.html#usage_description)或设置选项**--start来启动模拟。**

traci for py里定义的一些类：

node、edge、junction、connection、lane

每个路口都是一个node，每条道路都是一个edge，然后两个道路在某个点交汇形成一个junction，每一个node里可能会有很多的connection，这个connection是用来连接两个edge的，lane是车道的意思

节点id：601709881

edge的id：-47228917#2   这里的2指的lane编号。

**findRoute**(self, fromEdge, toEdge, vType='', depart=-1.0, routingMode=0)

```
findRoute(string, string, string, double, int) -> StageComputes the fastest route between the given edges for the given vehicletype (defaults to DEFAULT_VEHTYPE)Returns a Stage object that holds the edge list and the travel timeWhen the depart time is not set, the travel times at the current timewill be used. The routing mode may be ROUTING_MODE_DEFAULT (loaded ordefault speeds) and ROUTING_MODE_AGGREGATED (averaged historical speeds)
```

生成一个stage对象

+ 使用randomTrips生成20个随机路线

  (base) yl@redpc:~/Sumo/2023-03-24-20-20-26$ python /usr/share/sumo/tools/randomTrips.py -n osm.net.xml -r routs.rou.xml -e 9000

#### 永久的方法，直接修改XML文件

+ 生成20条起始点和终点可以来回的且不同的路线,20辆车分别从不同的起点去5个工厂送货

[('1110964736', '640084811'), ('749775286', '645299431#1'), ('22308620', '644958416'), ('159629978#0', '-676869698'), ('796708880', '481676841#1'), ('740593158', '159629978#0'), ('639593624#0', '977900324'), ('658171030', '481618763#1'), ('159629974', '658187505'), ('34127246#1', '192940058#0'), ('644962221', '645299431#1'), ('481679774', '964469329'), ('481659316#1', '122727336#0'), ('658171037', '658140815'), ('735354218', '827218852'), ('796708879', '330213981'), ('481618765#1', '657689877'), ('639551592#1', '641073128#1'), ('658171028#1', '481679774'), ('749515798#1', '657691226#1'), ('481676844#0', '512918030#2')]

+ 建立5个停车场，分别在他们的起始点，使得他们完成一趟送货就回到出发地点直到仿真结束。

{'1110964736': '1110964736_0', '749775286': '749775286_0', '22308620': '22308620_0', '159629978#0': '159629978#0_0', '796708880': '796708880_0', '740593158': '740593158_0', '639593624#0': '639593624#0_0', '658171030': '658171030_0', '159629974': '159629974_0', '34127246#1': '34127246#1_0', '644962221': '644962221_0', '481679774': '481679774_0', '481659316#1': '481659316#1_0', '658171037': '658171037_0', '735354218': '735354218_0', '796708879': '796708879_0', '481618765#1': '481618765#1_0', '639551592#1': '639551592#1_0', '658171028#1': '658171028#1_0', '749515798#1': '749515798#1_0', '481676844#0': '481676844#0_0'}

        <parkingArea id="Factory0" lane="1110964736_0" startPos="0" endPos="10" roadsideCapacity="5"/>
        <parkingArea id="Factory1" lane="749775286_0" startPos="0" endPos="10" roadsideCapacity="5" angle="-90"/>
        <parkingArea id="Factory2" lane="22308620_0" startPos="0" endPos="10" roadsideCapacity="5" angle="-90" />
        <parkingArea id="Factory3" lane="159629978#0_0" startPos="0" endPos="10" roadsideCapacity="5" angle="-90"/>
        <parkingArea id="Factory4" lane="740593158_0" startPos="0" endPos="10" roadsideCapacity="5" angle="-90"/>


1-4辆车去工厂0送货：[('464150152#1', '1110964736'), ('259686649', '1110964736'), ('655594042', '1110964736'), ('258894417#2', '1110964736'), ('479377241#1', '1110964736')],

5-8辆车去工厂1送货：[('737507570#1', '749775286'), ('537045023#1', '749775286'), ('652427907', '749775286'), ('749775286', '749775286'), ('537045023#1', '749775286')]

8-12辆车去工厂2送货：[('644822458', '22308620'), ('479377241#0', '22308620'), ('752085396#1', '22308620'), ('644961063', '22308620'), ('644962370', '22308620')]

12-16辆车去工厂3送货：[('749773515#1', '159629978#0'), ('658171040#2', '159629978#0'), ('481676842#0', '159629978#0'), ('656789616#0', '159629978#0'), ('479377241#0', '159629978#0')]

16-20辆车去工厂4送货：[('132001471#1', '740593158'), ('796708881', '740593158'), ('464307259', '740593158'), ('669878118', '740593158'), ('481646420#4', '740593158')]

### step3：创建NS-3可读的移动文件

+ cd  /yl/ns-allinone-3.38
+ cd ns-3.38
+ cd src
+ cd mobility/
+ cd examples/

最终路径在：/home/yl/yl/ns-allinone-3.38/ns-3.38/src/mobility/examples

+ yl@redpc:~/yl/ns-allinone-3.38/ns-3.38/src/mobility/examples$ cp ns2-mobility-trace.cc ../../../scratch/

复制ns2-mobility-trace.cc到/home/yl/yl/ns-allinone-3.38/ns-3.38/scratch

+ 在/home/yl/Sumo/2023-03-26-14-51-26路径下运行sumo -c osm.sumocfg --fcd-output trace.xml
+ 接下来我们使用 SUMO 的跟踪导出文件（traceExporter.py）生成 mobility.tcl 文件，其中包含每个节点/车辆在模拟的每一秒的移动性：

python traceExporter.py -i /home/yl/Sumo/2023-03-26-14-51-26/trace.xml --ns2mobility-output=/home/yl/Sumo/2023-03-26-14-51-26/mobility.tcl

**mobility.tcl 文件中的节点以下列形式存在：**https://www.isi.edu/nsnam/ns/doc/node172.html

$node_(2) set X_ 2719.68
$node_(2) set Y_ 5063.2
$node_(2) set Z_ 0
$ns_ at 0.0 "$node_(2) setdest 2719.68 5063.2 0.00"

表示在第0秒，节点2将开始以定义的速度0.00从其初始位置（X,Y）移动到目的地（2719.68 5063.2 0.00）



### step3：在地图中创建RSU

+ 获得地图的大小

  1186.45,5345.33

  2933.46,5427.11

  1012.99,3952.69

  地图的大小为x:1204.63 -2961.14=1756.51

  ​                       y:5374.54-3935.51=1439.03

每180个unit创建一个RSU，总共有80个RSU

原地图共有7354个节点

### step4:建立节点之间的通信

+ gdb debug

  ./ns3 run --gdb scratch/CustomSumo/wave-project.cc

+ gdb设置断点

  b main

+ 执行代码

  ./ns3 run scratch/CustomSumo/wave-project.cc > abc.out 2>&1

+ 可视化（Netanim）

+ 安装Netanim

  （1）安装依赖项 apt install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools

  （2）cd netanim/
            qamke NetAnim.pro
            make
            ./NetAnim

```
$ ./waf clean
$ ./waf configure --build-profile=optimized --enable-examples --enable-tests
```

修改file=/home/yl/yl/ns-allinone-3.38/ns-3.38/src/network/utils/queue.h, line=590,561

修改WiF-mac-queue-container.cc

./waf configure --build-profile=optimized --enable-examples --enable-tests --disable-werror

### step5:通过强化学习决定是否发送数据包

assert failed. cond="m_nQueuedBytes[addressTidPair] >= item->GetSize ()", +1.005579000s 1 file=../src/wifi/model/wifi-mac-queue.cc, line=544
terminate called without an active exception

### BackGround

RSU:2 个

汽车数量：5辆

信道带宽：20 MHZ  （802.11ax）

​					10 MHZ	（802.11p）

***<!--guardInterval = 800;-->***

UdpEchoServerApplication：是一个网络应用程序（Network Application），实现了一个简单的UDPP(Echo)服务器，可以接收来自客户端的数据报文并将其回显到客户端。

 UdpEchoClientApplication：实现一个UDP Echo客户端，用于测试网络连接、网络延迟和带宽等参数。当客户端发送数据报文到UdpEchoServerApplication后，服务器会将接收到的数据报文原封不动地返回给客户端。

+ UdpEchoClientApplication和UdpEchoServerApplication的区别是什么？

UdpEchoClientApplication和UdpEchoServerApplication是两个不同的网络应用程序，它们的主要区别在于它们的作用和功能。

UdpEchoServerApplication是一个UDP Echo服务器，它用于接收客户端发送的数据，并将其原封不动地返回给客户端。它的作用是测试网络连接和性能，并且可以测试网络延迟和带宽等参数。UdpEchoServerApplication的工作方式是等待客户端发送数据报文到服务器，然后将接收到的数据报文原封不动地返回给客户端。因此，UdpEchoServerApplication是一个被动的网络应用程序，它只响应客户端的请求，而不主动向客户端发送数据。

UdpEchoClientApplication是一个UDP Echo客户端，它用于向服务器发送数据，并等待服务器将接收到的数据原封不动地返回给客户端。它的作用也是测试网络连接和性能，并且可以测试网络延迟和带宽等参数。UdpEchoClientApplication的工作方式是向服务器发送数据报文，然后等待服务器将接收到的数据报文原封不动地返回给客户端。因此，UdpEchoClientApplication是一个主动的网络应用程序，它主动向服务器发送数据并等待服务器的响应。

总的来说，UdpEchoServerApplication和UdpEchoClientApplication都是UDP Echo测试工具，用于测试网络连接和性能。它们的主要区别在于UdpEchoServerApplication是被动的网络应用程序，而UdpEchoClientApplication是主动的网络应用程序。



RandomWalk2dMobilityModel：

二维随机行走移动模型。

每个实例都以用户提供的随机变量随机选择的速度和方向移动，直到步行了固定的距离或固定的时间。如果我们击中模型的边界之一（由矩形指定），我们将以反射角和速度在边界上反弹。该模型通常被认为是布朗运动模型。

YansWifiPhyHelper：

YansWifiPhyHelper是ns-3网络仿真平台中的一个类，它用于配置WiFi网络中的物理层（Physical Layer）参数。

在无线网络中，物理层是网络中最底层的一层，主要负责将数据从传输层传输到无线介质中。YansWifiPhyHelper类提供了一系列方法和属性，用于配置物理层相关的参数，如数据速率、信号传输范围、功率控制等。

+ YansWifiPhyHelper类的具体作用如下：

1. 配置WiFi网络中的物理层参数，如数据速率、信号传输范围、功率控制等。
2. 设置WiFi设备的天线数量和类型。
3. 设置传输信号的误码率、抗干扰能力等参数。
4. 配置多个物理层设备之间的互相影响，如干扰等。

总的来说，YansWifiPhyHelper是一个非常有用的类，它提供了一系列方法和属性，可以帮助用户更好地模拟WiFi网络中的物理层参数和特性，从而更加准确地评估无线网络的性能和可靠性。



+ Config::SetDefault("ns3::WifiRemoteStationManager::NonUnicastMode", StringValue(m_phyMode));

在计算机网络中，非单播（non-unicast）是指一种数据传输模式，其中数据帧被发送给多个接收方或广播到所有节点，而不仅仅是一个特定的目的地。

单播（unicast）是指数据帧仅被发送到特定的目的地，只有一个接收方可以接收到该帧。与单播不同，非单播帧可以被多个接收方或所有节点共享。

非单播传输模式通常用于多播（multicast）和广播（broadcast）数据的传输。多播是指数据帧被发送到一组接收方，而广播则是指数据帧被发送到所有节点。

在网络仿真和网络协议设计中，非单播传输模式的特点需要得到考虑和模拟。例如，在路由协议的设计中，需要考虑非单播数据帧的路由选择和转发策略，以实现高效的数据传输。在网络仿真中，需要模拟多播和广播数据帧的传输行为，以更好地理解网络协议的性能和行为。



+ 损耗模型

  FriisPropagationLossModel、ItuR1411LosPropagationLossModel、TwoRayGroundPropagationLossModel、ns3::LogDistancePropagationLossModel

+ 频段

  802.11p是一种基于IEEE 802.11标准的无线通信协议，主要用于车联网和智能交通系统中的车辆之间或车辆与基础设施之间的通信。5.9 GHz是指该协议使用的频段，其频率范围为5.850 GHz到5.925 GHz，是一种专门为车联网应用设计的无线频段，因其带宽较宽，具有较好的抗干扰能力，被广泛应用于车联网应用中。

+ 在 NS-3 中，YansWifiChannelHelper 是用于创建无线信道的辅助类。其中，AddPropagationLoss 和 SetPropagationLossModel 是两个方法，它们的作用都是设置无线信道的传播损耗模型，但是它们的使用方式和作用有一些不同。

  具体来说：

  1. AddPropagationLoss 方法用于向无线信道中添加传播损耗模型。它的参数可以是一个已经实例化的传播损耗模型对象，也可以是一个字符串，表示要使用的传播损耗模型类型。通过该方法可以向无线信道添加多个传播损耗模型，这些模型会依次计算信号在信道中的传播损耗。
  2. SetPropagationLossModel 方法用于设置无线信道的传播损耗模型。它的参数必须是一个已经实例化的传播损耗模型对象。通过该方法只能设置一个传播损耗模型，它会完全替换原有的传播损耗模型，而不是添加一个新的模型。

  因此，如果需要在无线信道中使用多个传播损耗模型，可以使用 AddPropagationLoss 方法，而如果只需要使用单一的传播损耗模型，则可以使用 SetPropagationLossModel 方法。

  

+ YansWifiPhyHelper、YansWavePhyHelper、NqosWaveMacHelpe、WaveHelper、Wifi80211pHelper区别是什么？

  

  

+ SetRemoteStationManage

  远程站点管理器（Remote Station Manager，RSM）是一种用于管理节点与远程设备之间无线通信的模块或算法。它通常包括两个主要功能：调整节点发送和接收的数据速率以及处理各种误码纠正技术（如自动重传请求（Automatic Repeat reQuest，ARQ）和前向纠错码（Forward Error Correction，FEC））。

  在无线网络中，信道条件经常变化，包括信号强度、多径效应、干扰等，因此节点需要不断地调整其发送和接收的数据速率以适应变化的信道条件。RSM模块可以根据当前信道条件选择最佳的数据速率，并调整发送功率和传输速率，以保证可靠地传输数据。

  此外，RSM模块还可以实现各种纠错技术来提高数据传输的可靠性，例如通过自动重传请求来纠正丢失的数据包，或使用前向纠错码来检测和纠正数据包中的错误。

  总的来说，远程站点管理器是无线通信系统中非常重要的一个模块，可以提高数据传输的可靠性和效率，使系统在不同的信道条件下都能正常运行。

+  wifiPhy.Set("TxPowerStart", DoubleValue(m_txp));

默认值为16 

+ 802.11p

m_adhocTxDevices = wifi80211p.Install(wifiPhy, wifi80211pMac, m_adhocTxNodes);

YansWifiPhyHelper wifiPhy，ns3::NqosWaveMacHelper wifi80211pMac，ns3::NodeContainer VanetRoutingExperiment::m_adhocTxNodes

+ 802.11b

m_adhocTxDevices = wifi.Install(wifiPhy, wifiMac, m_adhocTxNodes);

ns3::YansWifiPhyHelper wifiPhy,ns3::WifiMacHelper wifiMac,ns3::NodeContainer VanetRoutingExperiment::m_adhocTxNodes

+ WAVE-PHY

m_adhocTxDevices = waveHelper.Install(wavePhy, waveMac, m_adhocTxNodes);

ns3::YansWavePhyHelper wavePhy,ns3::QosWaveMacHelper waveMac,ns3::NodeContainer VanetRoutingExperiment::m_adhocTxNodes

+ ​    m_routingHelper->Install(m_adhocTxNodes,

  ​                             m_adhocTxDevices,

  ​                             m_adhocTxInterfaces,

  ​                             m_TotalSimTime,

  ​                             m_protocol,

  ​                             m_nSinks,

  ​                             m_routingTables);





## protobuf版本冲突

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

设置临时环境变量

+ ns3中debug

./ns3 run --gdb scratch/vanet-routing-compare

+ 终端对Python文件debug

python -m pdb tf_agent_training.py 

ns3gym 与ns3中的交互

```python
class EnvWrapper:
    def __init__(self, no_threads, **params):
        self.params = params       #{'simTime': 60, 'envStepTime': 0.01, 'historyLength': 300, 'agentType': 'discrete', 									'scenario': 'convergence', 'nWifi': 15}
        self.no_threads = no_threads   #1，开启一个线程
        self.ports = [13968+i+np.random.randint(40000) for i in range(no_threads)] #为每一个线程分配端口号
        self.commands = self._craft_commands(params)

        self.SCRIPT_RUNNING = False
        self.envs = []

        self.run()
        for port in self.ports:
            env = ns3env.Ns3Env(port=port, stepTime=params['envStepTime'], startSim=0, simSeed=0, simArgs=params, debug=False)
            self.envs.append(env)

        self.SCRIPT_RUNNING = True

    def run(self):
        if self.SCRIPT_RUNNING:
            raise AlreadyRunningException("Script is already running")

        for cmd, port in zip(self.commands, self.ports):
            print("###############"+cmd)
            subprocess.Popen(['bash', '-c', cmd])
        self.SCRIPT_RUNNING = True

    def _craft_commands(self, params):
        try:
            waf_pwd = find_ns3_path("./")
        except FileNotFoundError:
            import sys
            sys.path.append("../../")
            waf_pwd = find_waf_path("../../")

        command = f'{waf_pwd} --run "linear-mesh'
        for key, val in params.items():
            command+=f" --{key}={val}"

        commands = []
        for p in self.ports:
            commands.append(command+f' --openGymPort={p}"')

        return commands    #./ns3 run scratch/vanet/cw -- --simTime=60 --envStepTime=0.01 --historyLength=300 --agentType=discrete --scenario=convergence --nRSU=15 --openGymPort=15988


    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())

        return obs

    def step(self, actions):
        next_obs, reward, done, info = [], [], [], []

        for i, env in enumerate(self.envs):
            no, rew, dn, inf = env.step(actions[i].tolist())
            next_obs.append(no)
            reward.append(rew)
            done.append(dn)
            info.append(inf)

        return np.array(next_obs), np.array(reward), np.array(done), np.array(info)

    @property
    def observation_space(self):
        dim = repr(self.envs[0].observation_space).replace('(', '').replace(',)', '').split(", ")[2]
        return (self.no_threads, int(dim))

    @property
    def action_space(self):
        dim = repr(self.envs[0].action_space).replace('(', '').replace(',)', '').split(", ")[2]
        return (self.no_threads, int(dim))

    def close(self):
        time.sleep(5)
        for env in self.envs:
            env.close()
        # subprocess.Popen(['bash', '-c', "killall linear-mesh"])

        self.SCRIPT_RUNNING = False

    def __getattr__(self, attr):
        for env in self.envs:
            env.attr()

```

```python
class Teacher:
    """Class that handles training of RL model in ns-3 simulator

    Attributes:
        agent: agent which will be trained
        env (ns3-gym env): environment used for learning. NS3 program must be run before creating teacher
        num_agents (int): number of agents present at once
    """
    def __init__(self, env, num_agents, preprocessor):
        self.preprocess = preprocessor.preprocess
        self.env = env
        self.num_agents = num_agents
        self.CW = 16
        self.action = None              # For debug purposes

    def dry_run(self, agent, steps_per_ep):
        obs = self.env.reset()
        obs = self.preprocess(np.reshape(obs, (-1, len(self.env.envs), 1)))

        with tqdm.trange(steps_per_ep) as t:
            for step in t:
                self.actions = agent.act()
                next_obs, reward, done, info = self.env.step(self.actions)

                obs = self.preprocess(np.reshape(next_obs, (-1, len(self.env.envs), 1)))

                if(any(done)):
                    break

    def eval(self, agent, simTime, stepTime, history_length, tags=None, parameters=None, experiment=None):
        agent.load()
        steps_per_ep = int(simTime/stepTime + history_length)

        logger = Logger(True, tags, parameters, experiment=experiment)
        try:
            logger.begin_logging(1, steps_per_ep, agent.noise.sigma, agent.noise.theta, stepTime)
        except  AttributeError:
            logger.begin_logging(1, steps_per_ep, None, None, stepTime)
        add_noise = False

        obs_dim = 1
        time_offset = history_length//obs_dim*stepTime

        try:
            self.env.run()
        except AlreadyRunningException as e:
            pass

        cumulative_reward = 0
        reward = 0
        sent_mb = 0

        obs = self.env.reset()
        obs = self.preprocess(np.reshape(obs, (-1, len(self.env.envs), obs_dim)))

        with tqdm.trange(steps_per_ep) as t:
            for step in t:
                self.debug = obs
                self.actions = agent.act(np.array(logger.stations, dtype=np.float32), add_noise)
                #self.actions = agent.act(np.array(obs, dtype=np.float32), add_noise)
                next_obs, reward, done, info = self.env.step(self.actions)

                next_obs = self.preprocess(np.reshape(next_obs, (-1, len(self.env.envs), obs_dim)))

                cumulative_reward += np.mean(reward)

                if step>(history_length/obs_dim):
                    logger.log_round(obs, reward, cumulative_reward, info, agent.get_loss(), np.mean(obs, axis=0)[0], step)
                t.set_postfix(mb_sent=f"{logger.sent_mb:.2f} Mb", curr_speed=f"{logger.current_speed:.2f} Mbps")

                obs = next_obs

                if(any(done)):
                    break

        self.env.close()
        self.env = EnvWrapper(self.env.no_threads, **self.env.params)

        print(f"Sent {logger.sent_mb:.2f} Mb/s.\tMean speed: {logger.sent_mb/(simTime):.2f} Mb/s\tEval finished\n")

        logger.log_episode(cumulative_reward, logger.sent_mb/(simTime), 0)

        logger.end()
        return logger


    def train(self, agent, EPISODE_COUNT, simTime, stepTime, history_length, send_logs=True, experimental=True, tags=None, parameters=None, experiment=None):
        steps_per_ep = int(simTime/stepTime + history_length)

        logger = Logger(send_logs, tags, parameters, experiment=experiment)
        try:
            logger.begin_logging(EPISODE_COUNT, steps_per_ep, agent.noise.sigma, agent.noise.theta, stepTime)
        except  AttributeError:
            logger.begin_logging(EPISODE_COUNT, steps_per_ep, None, None, stepTime)

        add_noise = True

        obs_dim = 1
        time_offset = history_length//obs_dim*stepTime

        for i in range(EPISODE_COUNT):
            print(i)
            try:
                self.env.run()
            except AlreadyRunningException as e:
                pass

            if i>=EPISODE_COUNT*4/5:
                add_noise = False
                print("Turning off noise")

            cumulative_reward = 0
            reward = 0
            sent_mb = 0

            obs = self.env.reset()
            obs = self.preprocess(np.reshape(obs, (-1, len(self.env.envs), obs_dim))) #返回滑动窗口内的均值和方差    

            self.last_actions = None

            with tqdm.trange(steps_per_ep) as t:   #库创建一个进度条tqdm来跟踪循环中的进度
                for step in t:
                    self.debug = obs

                    self.actions = agent.act(np.array(obs, dtype=np.float32), add_noise)
                    next_obs, reward, done, info = self.env.step(self.actions)
                    # reward = 1-np.reshape(np.mean(next_obs), reward.shape)
                    next_obs = self.preprocess(np.reshape(next_obs, (-1, len(self.env.envs), obs_dim)))

                    if self.last_actions is not None and step>(history_length/obs_dim) and i<EPISODE_COUNT-1:
                        agent.step(obs, self.actions, reward, next_obs, done, 2)

                    cumulative_reward += np.mean(reward)

                    self.last_actions = self.actions

                    if step>(history_length/obs_dim):
                        logger.log_round(obs, reward, cumulative_reward, info, agent.get_loss(), np.mean(obs, axis=0)[0], i*steps_per_ep+step)
                    t.set_postfix(mb_sent=f"{logger.sent_mb:.2f} Mb", curr_speed=f"{logger.current_speed:.2f} Mbps")

                    obs = next_obs

                    if(any(done)):
                        break

            self.env.close()
            if experimental:
                self.env = EnvWrapper(self.env.no_threads, **self.env.params)

            agent.reset()
            print(f"Sent {logger.sent_mb:.2f} Mb/s.\tMean speed: {logger.sent_mb/(simTime):.2f} Mb/s\tEpisode {i+1}/{EPISODE_COUNT} finished\n")

            logger.log_episode(cumulative_reward, logger.sent_mb/(simTime), i)

        logger.end()
        print("Training finished.")
        return logger
```

```python
#%%
import matplotlib.pyplot as plt
import numpy as np

class Preprocessor:
    def __init__(self, plot=False):
        self.plot = plot
        if plot:
            self.fig, self.ax = plt.subplots(2, sharex=True)   
            # 创建一个包含两个子图的 matplotlib 图形，并共享它们的 x 轴。
            #plt.subplots()是 Matplotlib 库中的一个函数，它创建一个新图形并返回一个包含图形对象和轴对象数组的元组。元组被解包为两个变量self.fig和self.ax。这些变量可用于在该类的其他方法中操作图形及其子图。

    def normalize(self, sig):
        return np.clip((sig-np.min(sig))/(np.max(sig)-np.min(sig)+1e-6), 0, 1)

    def preprocess(self, signal):   #返回滑动窗口的均值和方差
        window = 150
        res = []

        for i in range(0, len(signal), window//2):
            res.append([
                [np.mean(signal[i:i+window, batch]),
                np.std(signal[i:i+window, batch])] for batch in range(0, signal.shape[1])])
        res = np.array(res)
        res = np.clip(res, 0, 1)

        if self.plot:
            plot_len = len(signal[:, 0, 0])
            plot_range = [i for i in range(plot_len, 0, -1)]

            res_0 = self.normalize(res[:, :, 0].repeat(window//2, 0))

            self.ax[0].clear()
            self.ax[0].plot(np.array(plot_range), self.normalize(signal[:, 0, 0]), c='b')
            self.ax[0].plot(np.array(plot_range), res_0, c='g')
            self.ax[0].plot(np.array(plot_range), res_0 + res[:, :, 1].repeat(window//2, 0), c='r')

            self.ax[1].clear()
            self.ax[1].plot(np.array(plot_range), signal[:, 0, 1], c='b')
            self.ax[1].plot(np.array(plot_range), res[:, :, 2].repeat(window//2, 0), c='g')

            plt.pause(0.001)
        return res

```

```python
class Ns3Env(gym.Env):
    def __init__(self, stepTime=0, port=0, startSim=True, simSeed=0, simArgs={}, debug=False):
        self.stepTime = stepTime
        self.port = port
        self.startSim = startSim
        self.simSeed = simSeed
        self.simArgs = simArgs
        self.debug = debug

        # Filled in reset function
        self.ns3ZmqBridge = None
        self.action_space = None
        self.observation_space = None

        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        self.ns3ZmqBridge = Ns3ZmqBridge(self.port, self.startSim, self.simSeed, self.simArgs, self.debug)
        self.ns3ZmqBridge.initialize_env(self.stepTime)
        self.action_space = self.ns3ZmqBridge.get_action_space()
        self.observation_space = self.ns3ZmqBridge.get_observation_space()
        # get first observations
        self.ns3ZmqBridge.rx_env_state()
        self.envDirty = False
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        obs = self.ns3ZmqBridge.get_obs()
        reward = self.ns3ZmqBridge.get_reward()
        done = self.ns3ZmqBridge.is_game_over()
        extraInfo = self.ns3ZmqBridge.get_extra_info()
        return (obs, reward, done, extraInfo)

    def step(self, action):
        response = self.ns3ZmqBridge.step(action)
        self.envDirty = True
        return self.get_state()

    def reset(self):
        if not self.envDirty:
            obs = self.ns3ZmqBridge.get_obs()
            return obs

        if self.ns3ZmqBridge:
            self.ns3ZmqBridge.close()
            self.ns3ZmqBridge = None

        self.envDirty = False
        self.ns3ZmqBridge = Ns3ZmqBridge(self.port, self.startSim, self.simSeed, self.simArgs, self.debug)
        self.ns3ZmqBridge.initialize_env(self.stepTime)
        self.action_space = self.ns3ZmqBridge.get_action_space()
        self.observation_space = self.ns3ZmqBridge.get_observation_space()
        # get first observations
        self.ns3ZmqBridge.rx_env_state()
        obs = self.ns3ZmqBridge.get_obs()
        return obs

    def render(self, mode='human'):
        return

    def get_random_action(self):
        act = self.action_space.sample()
        return act

    def close(self):
        if self.ns3ZmqBridge:
            self.ns3ZmqBridge.close()
            self.ns3ZmqBridge = None

        if self.viewer:
            self.viewer.close()
```

```python
from comet_ml import Experiment
import numpy as np
from collections import deque
import glob
import pandas as pd
import json
from datetime import datetime

#Logger(False, tags, parameters, experiment=experiment)
class Logger:
    def __init__(self, comet_ml, tags=None, parameters=None, experiment=None):
        self.stations = 5
        self.comet_ml = comet_ml
        self.logs = pd.DataFrame(columns=["step", "name", "type", "value"])
        self.fname = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

        if self.comet_ml:
            if experiment is None:
                try:
                    json_loc = glob.glob("./**/comet_token.json")[0]
                    with open(json_loc, "r") as f:
                        kwargs = json.load(f)
                except IndexError:
                    kwargs = {
                        "api_key": "XXXXXXXXXXXX",
                        "project_name": "rl-in-wifi",
                        "workspace": "XYZ"
                    }
#api_key：这是用于通过 Comet.ml 服务进行身份验证的 API 密钥。它是识别用户帐户并允许用户访问数据并将数据记录到他们自己的 Comet.ml 工作区的唯一密钥。
#project_name：这是与实验关联的项目的名称。它用于将相关实验组织和分组在一起，可用于以后搜索和过滤实验。
#workspace：这是与实验关联的工作区的名称。工作区是相关项目和实验的集合，用于管理和组织实验数据。


                self.experiment = Experiment(**kwargs)
            else:
                self.experiment = experiment
        self.sent_mb = 0
        self.speed_window = deque(maxlen=100)
        self.step_time = None
        self.current_speed = 0
        if self.comet_ml:
            if tags is not None:
                self.experiment.add_tags(tags)
            if parameters is not None:
                self.experiment.log_parameters(parameters)

    def log_parameter(self, param_name, param_value):
        if self.comet_ml:
            self.experiment.log_parameter(param_name, param_value)

        entry = {"step": 0, "name": param_name, "type": "parameter", "value": param_value}
        self.logs = self.logs.append(entry, ignore_index=True)

    def log_metric(self, metric_name, value, step=None):
        if self.comet_ml:
            self.experiment.log_metric(metric_name, value, step=step)

        entry = {"step": step, "name": metric_name, "type": "metric", "value": value}
        self.logs = self.logs.append(entry, ignore_index=True)

    def log_metrics(self, metrics, step):
        for metric in metrics:
            self.log_metric(metric, metrics[metric], step=step)

    def begin_logging(self, episode_count, steps_per_ep, sigma, theta, step_time):   #15，6300
        self.step_time = step_time
        self.log_parameter("Episode count", episode_count)
        self.log_parameter("Steps per episode", steps_per_ep)
        self.log_parameter("theta", theta)
        self.log_parameter("sigma", sigma)
        self.log_parameter("Step time", step_time)

    def log_round(self, states, reward, cumulative_reward, info, loss, observations, step):
        if self.comet_ml:
            self.experiment.log_histogram_3d(states, name="Observations", step=step)
        info = [[j for j in i.split("|")] for i in info]
        info = np.mean(np.array(info, dtype=np.float32), axis=0)
        try:
            round_mb = info[0]
        except Exception as e:
            print(info)
            print(reward)
            raise e
        self.speed_window.append(round_mb)
        self.current_speed = np.mean(np.asarray(self.speed_window)/self.step_time)
        self.sent_mb += round_mb
        CW = info[1]
        self.stations = info[2]
        fairness = info[3]

        self.log_metric("Round reward", np.mean(reward), step=step)
        self.log_metric("Per-ep reward", np.mean(cumulative_reward), step=step)
        self.log_metric("Megabytes sent", self.sent_mb, step=step)
        self.log_metric("Round megabytes sent", round_mb, step=step)
        self.log_metric("Chosen CW", CW, step=step)
        self.log_metric("Station count", self.stations, step=step)
        self.log_metric("Current throughput", self.current_speed, step=step)
        self.log_metric("Fairness index", fairness, step=step)

        for i, obs in enumerate(observations):
            self.log_metric(f"Observation {i}", obs, step=step)
            self.log_metrics(loss, step=step)
        
    def log_episode(self, cumulative_reward, speed, step):
        self.log_metric("Cumulative reward", cumulative_reward, step=step)
        self.log_metric("Speed", speed, step=step)

        self.sent_mb = 0
        self.last_speed = speed
        self.speed_window = deque(maxlen=100)
        self.current_speed = 0
        self.logs.to_csv(self.fname)

    def end(self):
        if self.comet_ml:
            self.experiment.end()
        
        self.logs.to_csv(self.fname)
```

```Python
class Ns3Env(gym.Env):
    def __init__(self, stepTime=0, port=0, startSim=True, simSeed=0, simArgs={}, debug=False):
        self.stepTime = stepTime
        self.port = port
        self.startSim = startSim
        self.simSeed = simSeed
        self.simArgs = simArgs
        self.debug = debug

        # Filled in reset function
        self.ns3ZmqBridge = None
        self.action_space = None
        self.observation_space = None

        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        self.ns3ZmqBridge = Ns3ZmqBridge(self.port, self.startSim, self.simSeed, self.simArgs, self.debug)
        self.ns3ZmqBridge.initialize_env(self.stepTime)
        self.action_space = self.ns3ZmqBridge.get_action_space()
        self.observation_space = self.ns3ZmqBridge.get_observation_space()
        # get first observations
        self.ns3ZmqBridge.rx_env_state()
        self.envDirty = False
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        obs = self.ns3ZmqBridge.get_obs()
        reward = self.ns3ZmqBridge.get_reward()
        done = self.ns3ZmqBridge.is_game_over()
        extraInfo = self.ns3ZmqBridge.get_extra_info()
        return (obs, reward, done, extraInfo)

    def step(self, action):
        response = self.ns3ZmqBridge.step(action)
        self.envDirty = True
        return self.get_state()

    def reset(self):
        if not self.envDirty:
            obs = self.ns3ZmqBridge.get_obs()
            return obs

        if self.ns3ZmqBridge:
            self.ns3ZmqBridge.close()
            self.ns3ZmqBridge = None

        self.envDirty = False
        self.ns3ZmqBridge = Ns3ZmqBridge(self.port, self.startSim, self.simSeed, self.simArgs, self.debug)
        self.ns3ZmqBridge.initialize_env(self.stepTime)
        self.action_space = self.ns3ZmqBridge.get_action_space()
        self.observation_space = self.ns3ZmqBridge.get_observation_space()
        # get first observations
        self.ns3ZmqBridge.rx_env_state()
        obs = self.ns3ZmqBridge.get_obs()
        return obs

    def render(self, mode='human'):
        return

    def get_random_action(self):
        act = self.action_space.sample()
        return act

    def close(self):
        if self.ns3ZmqBridge:
            self.ns3ZmqBridge.close()
            self.ns3ZmqBridge = None

        if self.viewer:
            self.viewer.close()
```

