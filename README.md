This project implemented the matrix decomposition targetting the **re-paraphrase** of Audio channel ASR corpus.

**Text feature decomposition for text paraphrase-constriction**

- **Theory**

    While our target is splitting our raw sentences to semantic paraphrases or specfic topic focusing sentences. The first thing we'd like 
to do is enumerate all the topic with raw sentences. The un-supervise techonology maybe the only choice--and i will prove it suits this task well. 

**How to enumerate the topics from the raw text?**

How to enumerate the topics, this is'nt something trivial. Basically, we dont know how many topics in a stage of broadcaste, so we inplemented the un-superwised algorithm, which is matrix decomposition here. But we also do'nt know how good is our result--is the sample topic are semantic paraphrase or something bullshit. So i will introduce two metric about the evaluation about our algorithm. Actually, they do a great help in my empirical experiments.
    - first, the average map rate, it denote the how close is the topic freqence words with each sentences(So its average).
    - second, the continuity rate, this metric reveal the how good each topic is close to a semantic sentences(context-sensitive so its readable).
```
 good topic numerated:
 
Topic #0: 婚姻 能够 明显 希望 树新风 讲文明 公益广告 下去 广播 心态
Topic #1: 岗位 听众 觉得 稀里糊涂 回家 起来 个头 过年 这笔 车车
Topic #2: 可能 来讲 重新 比如说 工作 或者说 朋友 当中 很多 肯定
Topic #3: 所谓 天空 正常 海底 责任 对方 正式 生完 四五个 是因为
Topic #4: 愿意 准备 焦虑 到底 孩子 知道 手机 不行 我要 辞职
Topic #5: 这种 知道 情况 孩子 需要 不想 踏踏实实 之后 照顾 请问
Topic #6: 特别 只能 高兴 就要 离婚 兰州 地方 发展 难听 老公
Topic #7: 老师 应该 快乐 小年 谢谢您 一步 谢谢 来看 工作 再次
Topic #8: 感觉 小时 之前 问题 希望 提到 双方 孩子 头发 资金
Topic #9: 当中 过程 对方 诚实 能够 如果说 聊天 打电话 欢迎 开心
Topic #10: 之前 里面 不要 经验 出来 夜话 兰山 留言 觉得 半年
why good?: we have ADs, jobs, broadcast channel intros, school stuff and marriage stuff seperated.


```

**How to find the best topics number?**
Topic number are like centroids, for we are doing the classfication according the topics. As I said the number of topics from stages vary. It's not a puzzle, and we donot have some prior knowledge to set this number. The only approach is expriment, or grid search to find the best centroid number. And the evalate metric are the two mention before: average map rate and continuity rate. i will present the empirical experimental below:
```
#I denote the topic number, count rate and map rate explained above
I:10 {'count rate': 0.24071122646442536, 'map rate': 0.09038461538461587} 0.3311
I:11 {'count rate': 0.2665704771582997, 'map rate': 0.09195804195804243} 0.3585
I:12 {'count rate': 0.20843997458889033, 'map rate': 0.10087412587412638} 0.3093
I:13 {'count rate': 0.2075082288660859, 'map rate': 0.09632867132867179} 0.3038
I:14 {'count rate': 0.18657498800069866, 'map rate': 0.09755244755244806} 0.2841
I:15 {'count rate': 0.17270018696751288, 'map rate': 0.10052447552447603} 0.2732
I:16 {'count rate': 0.14213337939158224, 'map rate': 0.10279720279720339} 0.2449
I:17 {'count rate': 0.1596169754776241, 'map rate': 0.10349650349650404} 0.2631
I:18 {'count rate': 0.12892016115650834, 'map rate': 0.1089160839160845} 0.2378
I:19 {'count rate': 0.14405133591486435, 'map rate': 0.11223776223776276} 0.2563
I:20 {'count rate': 0.13467529560086286, 'map rate': 0.11451048951049006} 0.2492
I:21 {'count rate': 0.15141817239292998, 'map rate': 0.11748251748251794} 0.2689
I:22 {'count rate': 0.11305568181318482, 'map rate': 0.11223776223776272} 0.2253
I:23 {'count rate': 0.1249637748198169, 'map rate': 0.11433566433566476} 0.2393
I:24 {'count rate': 0.1158039132717906, 'map rate': 0.12430069930069979} 0.2401
I:25 {'count rate': 0.12295342470271313, 'map rate': 0.12202797202797241} 0.245
I:26 {'count rate': 0.13639176588568072, 'map rate': 0.12202797202797254} 0.2584
I:27 {'count rate': 0.14215198303231744, 'map rate': 0.1288461538461541} 0.271
I:28 {'count rate': 0.10248705065734051, 'map rate': 0.13076923076923103} 0.2333
I:29 {'count rate': 0.09386451314196323, 'map rate': 0.127622377622378} 0.2215
I:30 {'count rate': 0.11840838888252683, 'map rate': 0.13391608391608437} 0.2523
I:31 {'count rate': 0.12138785314841653, 'map rate': 0.1354895104895109} 0.2569
I:32 {'count rate': 0.12288603265987558, 'map rate': 0.1381118881118883} 0.261
I:33 {'count rate': 0.08592291922314456, 'map rate': 0.12762237762237805} 0.2135
I:34 {'count rate': 0.10606509750894777, 'map rate': 0.13461538461538483} 0.2407
I:35 {'count rate': 0.11220131433753726, 'map rate': 0.13933566433566455} 0.2515
I:36 {'count rate': 0.08501783053869406, 'map rate': 0.14125874125874147} 0.2263
I:37 {'count rate': 0.08901904054389248, 'map rate': 0.1435314685314687} 0.2326
I:38 {'count rate': 0.07909983994455329, 'map rate': 0.14038461538461552} 0.2195
I:39 {'count rate': 0.09350929480831444, 'map rate': 0.14440559440559456} 0.2379
I:40 {'count rate': 0.09168578839557828, 'map rate': 0.14737762237762247} 0.2391
I:41 {'count rate': 0.12219244349275307, 'map rate': 0.14353146853146873} 0.2657
I:42 {'count rate': 0.08742254424995186, 'map rate': 0.14965034965034976} 0.2371
I:43 {'count rate': 0.12293488879623334, 'map rate': 0.15000000000000013} 0.2729
I:44 {'count rate': 0.052484238077977856, 'map rate': 0.1484265734265735} 0.2009
I:45 {'count rate': 0.09854205064098111, 'map rate': 0.1479020979020979} 0.2464
I:46 {'count rate': 0.10120636094148305, 'map rate': 0.14562937062937079} 0.2468
I:47 {'count rate': 0.10618053243053246, 'map rate': 0.1520979020979022} 0.2583
I:48 {'count rate': 0.08102887184894529, 'map rate': 0.14178321678321698} 0.2228
I:49 {'count rate': 0.10991281554408092, 'map rate': 0.14702797202797213} 0.2569
...
The global optimal parameter.....
[11, 0.3585] [47, 0.1521] [11, 0.2666]
```

**how to re-parapherase?**
Our target is to order the sentences into context-sensible groups. What is groups, the topics, but how to gurantee the context-sensible? The answer is continuity. Actually we can pick the highest continuity topic from each topic number-from grid search. I will present the demo directly:
```
re-parapherase with most high continuity(id before sentence denote time info):
topic number: 25
topic 4th
481	我，我觉得我一直找不到特别喜欢的人。
482	找不到特别喜欢的
484	是你从来没有遇到过自己，特别喜欢的还是说遇到过，喜欢的，但是人家不喜欢你，没找上
485	我觉得我天天都能遇到喜欢的人。
486	但是过一天之后，就觉得不喜欢了。
491	那如果是这样的话，那你找不着自己喜欢的人，很正常。
492	为啥很正常？
493	首先，你对一个人的喜欢，你说我天天都能够遇到自己喜欢的人，你这都喜欢来自于什么来自于外表。
495	那你每天都能遇到的，或者说这个么，天天可能夸张，那些，但是你经常都能遇到的不主要来自于外表，还能来自于什么，您对这个人你根本都还没来得及了解呢，你就喜欢了，这不是外表是什么呢，一个言行一个余只，你觉得就是我特别喜欢呢
499	对我觉得你说的特别有道理
501	当然，人都是第一呃，第一第一时间判断的话，靠的是市局是吧，靠的这个呃，看到了，哎，我比较喜欢这款，或者说呃那对哪一个类型的比较中意什么，包括这个月分期只一个动作，一个行为呀，我特别
503	他可能这个喜欢
505	也会觉得特别的好，但是那能像这个真正意象的，喜欢吗，可能看完了过去就忘了，甚至说你过两天，你连就是都想不起来了，就觉得你说得对，我觉得我的这个喜欢，可能停留的这个层次比较一比较全，都是只简单的乱叫言情一个细节
506	再把它沉淀一点，加厚对吧，真正容易喜欢一个人，是在对这个人了解基础之上，然后说你特别的喜欢他，那你就喜欢，绝对不会说我喜欢了，今天喜欢明天就忘绝对不会解压。
507	那哥儿子，你有没有遇到，就是看到一个人就特别喜欢，看到一个人就特别喜欢。
508	您的这个喜欢的，我要看您喜欢，怎么去定义了
517	对我明白了，一共一说，我就觉得扩展开了那好那特别渊博，感谢，我还有一个还有一个问题，那个我我不是他发票，我觉得真的您的说的特别好，就是前面跟您说就像飞机起飞一样的在跑道上肯定要冲刺吗？
518	比如你说这句话我就觉得
525	是这个道理吧，你不是直升飞机，你非得按照直升飞机那个要求，但是我觉得这个这个排量，这台的什么都是特别没水平吗，对吗，所以我们说去对比去去去这个参照的话那一定是以什么传到的，一定是以什么对比的是结合自身的这个条件去对比去参照的而不是说
528	今天快乐，新年快乐，再见

Topic number 49
topic 31th
8	一声汽笛吹响了回家的号角，一次起降，期待中换价的团员
9	过年回家，澄海的太多的含义，此刻的您正在回家的路上，还是已经安全到家，开始品味大妈到好的那胡日查
10	我们来听听大家对回家的那份期待，一年就回一次高兴，赵海居高十七年就是涨，让大家都过上幸福快乐的时候。
16	人民水壶之后，回家过年，这个幸福狗星了
18	在回家过年的大厅中，还有这样一个群体，他们就是在外就读的学生羽翼，还没有完全丰满，就绚丽的父母，他们对于回家的那份期待，也是不言而喻的。
20	哦，在这边上学，假期打工，昨天晚上都很激动，专员都没睡着，感觉不管在哪，爸爸妈妈在家里就是回家，就是需要温暖一些，我是西北师范大学大三的一名学生，报价不是，比较早嘛，然后就说打几天工再回家到快回家，他这几天
21	就特别着急，就想赶紧回去回家过年吧，回去看他爸妈，让他们放心吧，道声平安晚，我您回家，我们回家
24	嗯，您可以跟我们回家，现在先生都还听说
26	您先回家
28	啊，我是今年第一次，然后离开自己父母啊，然后他这边来过年，因为是结婚军演，然后要说公仆他家过年幸福的我是在兰州工大的河南人，今年嘛第一次带媳妇回家，所以皮带第四框架也期待的挺长时间呢，心情比较激动吗？
36	一切就等着这个数着日子，到了以后，赶紧上车回家，能去见见父母跟姐妹们团圆一下，或是在兰州上班的这个河南焦作人不是年年都能回家很高兴，几年都没见面，然后两三年健身运
38	我是静安静静平安的青岛工作呢，转车过来的啊，怎么说呢，回家过年嘛，肯定是很激动了，一直很激动啊，各种准备，没想到的东西尽量带的能带多少带多少春节过年，大家都一起回家吗，这个新建就是不一样
40	有钱没钱回家过年，不用考虑这爸妈为家人带什么礼物，因为啊回家就是最好的礼物，家销售逐行胡彦锁恭祝大家新春大吉吉祥如意。

And one of the most useful implementation is the ADs detection:
3	讲文明，树新风公益广告
8	本公益广告由甘肃省文明、磐安住商新闻出版广电局，登录人民广播电台联合制作播出
10	九七分，兰州综合广播讲文明树新风公益广告
14	人民阳泉从点滴做起事结束，文明养犬人
18	九积分兰州综合广播讲文明树新风公益广告
23	徐文明用语文明出行
24	秦文明，细
27	九、提升兰州综合广播讲文明树新风公益广告
```

Theory finished(tired

- *Project architecture*
...etc.


**TODO:**

- ASR improvemental for better corpus.
- Token embedding with more semantic vocabulary.



**server demo**
![avatar](../server.png)

